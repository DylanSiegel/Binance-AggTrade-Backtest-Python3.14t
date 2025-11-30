#!/usr/bin/env python3.14t
"""
update.py (Universal Auto-Updater for AGG2 data)

Target runtime:
- Python 3.14t, GIL disabled (-X gil=0)
- Windows 11
- AMD Ryzen 9 7900X (12C / 24T)

Constraints:
- Standard library only
- CPU-bound = threading / ThreadPoolExecutor (no multiprocessing)
- Compression via compression.zstd only

Purpose:
    1. Scans BASE_DIR to find all existing symbols (coins).
    2. For each symbol, computes the date range to sync.
    3. Fills gaps and appends new days in the same format as data.py.

Storage format (per symbol, per month):
    BASE_DIR/SYMBOL/YYYY/MM/data.quantdev   : concatenated compressed day blobs
    BASE_DIR/SYMBOL/YYYY/MM/index.quantdev  : sequence of <day, offset, length>
"""

import os
import re
import gc
import time
import struct
import datetime as dt
import http.client
import ssl
import signal
import zipfile
from io import BytesIO, TextIOWrapper
from csv import reader as csv_reader
from pathlib import Path

import sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# 0. Mandatory 3.14t / Free-threaded boilerplate
# ---------------------------------------------------------------------------

if sys._is_gil_enabled():
    raise RuntimeError("GIL must be disabled. Run Python 3.14t with -X gil=0")

CPU_THREADS = 24  # Ryzen 9 7900X logical threads

try:
    import compression.zstd as zstd
except ImportError as exc:
    raise RuntimeError("Native 'compression.zstd' (PEP 784) is required.") from exc

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

CONFIG: dict[str, object] = {
    "BASE_DIR": "data",

    # Network / source settings (must match data.py)
    "HOST_DATA": "data.binance.vision",
    "S3_PREFIX": "data/futures/um",   # UM futures
    "DATASET": "aggTrades",

    # Lower bound if genesis probing fails
    "FALLBACK_DATE": dt.date(2020, 1, 1),

    # Threading
    "WORKERS": CPU_THREADS,
    "ROWS_PER_CHUNK": 50_000,

    # Network policy
    "TIMEOUT": 15,
    "RETRIES": 5,
    "BACKOFF": 0.5,
    "VERIFY_SSL": True,
    "USER_AGENT": f"QuantEngine/3.14t Updater ({CPU_THREADS} threads)",
}

UTC = dt.timezone.utc
_thread_local = threading.local()
_stop_event = threading.Event()

_ssl_context = ssl.create_default_context()
if not CONFIG["VERIFY_SSL"]:
    _ssl_context.check_hostname = False
    _ssl_context.verify_mode = ssl.CERT_NONE

# ---------------------------------------------------------------------------
# 2. Binary Schema (must match data.py)
# ---------------------------------------------------------------------------

PX_SCALE = 100_000_000
QT_SCALE = 100_000_000

AGG_HDR_MAGIC = b"AGG2"

AGG_ROW_STRUCT = struct.Struct("<QQQQHHqB3x")
AGG_ROW_SIZE = AGG_ROW_STRUCT.size

AGG_HDR_STRUCT = struct.Struct("<4sBBHQqq16x")
INDEX_ROW_STRUCT = struct.Struct("<HQQ")
INDEX_ROW_SIZE = INDEX_ROW_STRUCT.size

FLAG_IS_BUYER_MAKER = 1 << 0

# ---------------------------------------------------------------------------
# 3. Locks & Path Helpers (per symbol/month)
# ---------------------------------------------------------------------------

_month_locks: dict[tuple[str, int, int], threading.Lock] = {}
_guard = threading.Lock()


def _get_month_lock(symbol: str, year: int, month: int) -> threading.Lock:
    key = (symbol, year, month)
    with _guard:
        lock = _month_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _month_locks[key] = lock
        return lock


def _ensure_paths(symbol: str, year: int, month: int) -> tuple[str, str, str]:
    base_dir = str(CONFIG["BASE_DIR"])
    dir_path = os.path.join(base_dir, symbol, f"{year:04d}", f"{month:02d}")
    os.makedirs(dir_path, exist_ok=True)
    data_path = os.path.join(dir_path, "data.quantdev")
    index_path = os.path.join(dir_path, "index.quantdev")
    return dir_path, data_path, index_path


def _is_day_indexed(symbol: str, year: int, month: int, day: int) -> bool:
    """
    True if index has an entry for this (symbol, year, month, day) and that entry
    points entirely within the current data file.

    If an index row exists but offset+length > data_size, treat as NOT indexed
    so we can safely append a new valid block on rerun.
    """
    _, data_path, index_path = _ensure_paths(symbol, year, month)

    if not os.path.exists(index_path) or not os.path.exists(data_path):
        return False

    try:
        data_size = os.path.getsize(data_path)
        with open(index_path, "rb") as f:
            while True:
                raw = f.read(INDEX_ROW_SIZE)
                if not raw or len(raw) < INDEX_ROW_SIZE:
                    break
                d, offset, length = INDEX_ROW_STRUCT.unpack(raw)
                if d == day and (offset + length) <= data_size:
                    return True
    except OSError:
        return False
    return False

# ---------------------------------------------------------------------------
# 4. Networking
# ---------------------------------------------------------------------------


def _get_connection(host: str) -> http.client.HTTPSConnection:
    conns = getattr(_thread_local, "conns", None)
    if conns is None:
        conns = {}
        _thread_local.conns = conns

    conn = conns.get(host)
    if conn is None or conn.sock is None:
        conn = http.client.HTTPSConnection(
            host,
            context=_ssl_context,
            timeout=float(CONFIG["TIMEOUT"]),
        )
        conns[host] = conn
    else:
        conn.timeout = float(CONFIG["TIMEOUT"])
    return conn


def _close_connection(host: str) -> None:
    conns = getattr(_thread_local, "conns", None)
    if not conns:
        return
    conn = conns.pop(host, None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def _http_request(host: str, method: str, path: str) -> bytes | None:
    headers = {
        "User-Agent": str(CONFIG["USER_AGENT"]),
        "Accept-Encoding": "identity",
    }
    retries = int(CONFIG["RETRIES"])
    backoff = float(CONFIG["BACKOFF"])

    for attempt in range(1, retries + 1):
        if _stop_event.is_set():
            return None

        conn = _get_connection(host)
        try:
            conn.request(method, path, headers=headers)
            resp = conn.getresponse()
            try:
                if resp.status == 200:
                    return resp.read()
                if resp.status == 404:
                    resp.read()
                    return None
                resp.read()
            finally:
                resp.close()
        except (http.client.HTTPException, OSError, ssl.SSLError):
            _close_connection(host)

        if attempt < retries:
            time.sleep(backoff * (2 ** (attempt - 1)))

    return None

# ---------------------------------------------------------------------------
# 5. Genesis Detection (per symbol)
# ---------------------------------------------------------------------------


def _fetch_genesis_date(symbol: str) -> dt.date:
    """
    Try to detect earliest available date for this symbol by scraping
    the S3 listing from data.binance.vision.

    If anything fails, return FALLBACK_DATE.
    """
    host = str(CONFIG["HOST_DATA"])
    dataset = str(CONFIG["DATASET"])
    prefix = f"{CONFIG['S3_PREFIX']}/daily/{dataset}/{symbol}/"
    path = f"/?prefix={prefix}&delimiter=/"

    html_bytes = _http_request(host, "GET", path)

    fallback = CONFIG["FALLBACK_DATE"]
    if not isinstance(fallback, dt.date):
        fallback = dt.date(2020, 1, 1)

    if not html_bytes:
        return fallback

    text = html_bytes.decode("utf-8", "replace")
    pattern = rf"{re.escape(symbol)}-{re.escape(dataset)}-(\d{{4}}-\d{{2}}-\d{{2}})"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return fallback

    dates: list[dt.date] = []
    for date_str in matches:
        try:
            dates.append(dt.datetime.strptime(date_str, "%Y-%m-%d").date())
        except ValueError:
            continue

    return min(dates) if dates else fallback

# ---------------------------------------------------------------------------
# 6. Download + Transform (same as data.py semantics)
# ---------------------------------------------------------------------------


def _download_day_zip(symbol: str, day: dt.date) -> bytes | None:
    year = day.year
    month_str = f"{day.month:02d}"
    day_str = f"{day.day:02d}"

    dataset = str(CONFIG["DATASET"])
    prefix = str(CONFIG["S3_PREFIX"])

    path = (
        f"/{prefix}/daily/{dataset}/{symbol}/"
        f"{symbol}-{dataset}-{year}-{month_str}-{day_str}.zip"
    )
    return _http_request(str(CONFIG["HOST_DATA"]), "GET", path)


def _process_zip_to_agg2(day: dt.date, zip_bytes: bytes | None) -> bytes | None:
    if not zip_bytes:
        return None

    buf_size = AGG_ROW_SIZE * int(CONFIG["ROWS_PER_CHUNK"])

    buf = getattr(_thread_local, "buf", None)
    if buf is None or len(buf) != buf_size:
        buf = bytearray(buf_size)
        _thread_local.buf = buf

    view = memoryview(buf)
    pack_into = AGG_ROW_STRUCT.pack_into
    chunks: list[bytes] = []
    c_min, c_max, total_count = 2**63 - 1, -2**63, 0

    try:
        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return None

            with zf.open(csv_names[0], "r") as f_in:
                wrapper = TextIOWrapper(f_in, encoding="utf-8", newline="")
                reader = csv_reader(wrapper)
                first_row = next(reader, None)
                if not first_row:
                    return None

                headers = [h.replace("_", "").lower().strip() for h in first_row]
                idx: dict[str, int] = {
                    "id": 0,
                    "px": 1,
                    "qt": 2,
                    "fi": 3,
                    "li": 4,
                    "ts": 5,
                    "bm": 6,
                }

                if "aggtradeid" in headers or "id" in headers:
                    m = {name: i for i, name in enumerate(headers)}
                    idx["id"] = m.get("aggtradeid", m.get("id", 0))
                    idx["px"] = m.get("price", 1)
                    idx["qt"] = m.get("quantity", 2)
                    idx["fi"] = m.get("firsttradeid", 3)
                    idx["li"] = m.get("lasttradeid", 4)
                    idx["ts"] = m.get("transacttime", 5)
                    idx["bm"] = m.get("isbuyermaker", 6)

                offset = 0

                def _process(rows) -> None:
                    nonlocal offset, total_count, c_min, c_max
                    for row in rows:
                        if not row:
                            continue
                        try:
                            ts = int(row[idx["ts"]])
                            if ts < c_min:
                                c_min = ts
                            if ts > c_max:
                                c_max = ts

                            fi = int(row[idx["fi"]])
                            li = int(row[idx["li"]])
                            cnt = li - fi + 1
                            if cnt < 0:
                                continue
                            if cnt > 65535:
                                cnt = 65535

                            px = int(float(row[idx["px"]]) * PX_SCALE)
                            qt = int(float(row[idx["qt"]]) * QT_SCALE)
                            is_maker = row[idx["bm"]].lower() in ("true", "1", "t")

                            pack_into(
                                view,
                                offset,
                                int(row[idx["id"]]),
                                px,
                                qt,
                                fi,
                                cnt,
                                FLAG_IS_BUYER_MAKER if is_maker else 0,
                                ts,
                                0 if is_maker else 1,
                            )
                            offset += AGG_ROW_SIZE
                            total_count += 1

                            if offset >= buf_size:
                                chunks.append(view[:offset].tobytes())
                                offset = 0
                        except (ValueError, IndexError):
                            continue

                if "aggtradeid" not in headers and "price" not in headers:
                    _process([first_row])
                _process(reader)

                if offset > 0:
                    chunks.append(view[:offset].tobytes())

    except Exception:
        return None

    if total_count == 0:
        return b""

    hdr = AGG_HDR_STRUCT.pack(
        AGG_HDR_MAGIC,
        1,          # version
        day.day,    # day
        0,          # reserved
        total_count,
        c_min,
        c_max,
    )
    return hdr + b"".join(chunks)

# ---------------------------------------------------------------------------
# 7. Per-day worker (atomic-ish writer, no self-healing)
# ---------------------------------------------------------------------------


def _process_day(symbol: str, day: dt.date) -> str:
    if _stop_event.is_set():
        return "stop"

    y, m, d = day.year, day.month, day.day

    if _is_day_indexed(symbol, y, m, d):
        return "skip"

    zip_data = _download_day_zip(symbol, day)
    if _stop_event.is_set():
        return "stop"
    if zip_data is None:
        return "missing"

    agg_blob = _process_zip_to_agg2(day, zip_data)
    if agg_blob is None:
        return "error"
    if not agg_blob:
        return "missing"

    try:
        c_blob = zstd.compress(agg_blob, level=3)
    except Exception:
        return "error"

    blob_len = len(c_blob)
    lock = _get_month_lock(symbol, y, m)
    _, data_path, index_path = _ensure_paths(symbol, y, m)

    try:
        with lock:
            if _is_day_indexed(symbol, y, m, d):
                return "skip"

            # ensure data file exists
            if not os.path.exists(data_path):
                with open(data_path, "ab"):
                    pass

            # 1) append compressed block and get true offset via tell()
            with open(data_path, "ab") as f_data:
                start_offset = f_data.tell()
                f_data.write(c_blob)
                f_data.flush()
                os.fsync(f_data.fileno())

            # 2) append index row referencing fully written block
            with open(index_path, "ab") as f_idx:
                f_idx.write(INDEX_ROW_STRUCT.pack(d, start_offset, blob_len))
                f_idx.flush()
                os.fsync(f_idx.fileno())

        return "ok"
    except OSError:
        return "error"

# ---------------------------------------------------------------------------
# 8. Chunk worker and chunking utility (exactly 24 chunks)
# ---------------------------------------------------------------------------


def _worker_chunk(symbol: str, days: list[dt.date]) -> dict[str, int]:
    stats: dict[str, int] = {"ok": 0, "skip": 0, "missing": 0, "error": 0, "stop": 0}
    for day in days:
        if _stop_event.is_set():
            stats["stop"] += 1
            break
        res = _process_day(symbol, day)
        stats[res] = stats.get(res, 0) + 1
        if res == "stop":
            break
    return stats


def _split_into_chunks(seq: list[dt.date], n: int) -> list[list[dt.date]]:
    length = len(seq)
    if n <= 0:
        return [seq]

    base = length // n
    rem = length % n

    chunks: list[list[dt.date]] = []
    idx = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        if size > 0:
            chunk = seq[idx:idx + size]
        else:
            chunk = []
        chunks.append(chunk)
        idx += size
    return chunks

# ---------------------------------------------------------------------------
# 9. High-level per-symbol updater
# ---------------------------------------------------------------------------


def _update_symbol(symbol: str) -> None:
    print(f"\n=== Updating {symbol} ===")

    start_date = _fetch_genesis_date(symbol)
    today_utc = dt.datetime.now(UTC).date()
    end_date = today_utc - dt.timedelta(days=1)

    if end_date < start_date:
        print("   -> Up to date (no days after genesis).")
        return

    days: list[dt.date] = [
        start_date + dt.timedelta(days=i)
        for i in range((end_date - start_date).days + 1)
    ]
    print(f"   -> Date range: {start_date} to {end_date} ({len(days)} days)")

    # Split into exactly 24 chunks (some may be empty)
    chunks = _split_into_chunks(days, CPU_THREADS)

    stats: dict[str, int] = {
        "ok": 0,
        "skip": 0,
        "missing": 0,
        "error": 0,
        "stop": 0,
    }

    gc.disable()
    try:
        with ThreadPoolExecutor(max_workers=CPU_THREADS) as executor:
            futures = {
                executor.submit(_worker_chunk, symbol, chunk): idx
                for idx, chunk in enumerate(chunks)
            }

            completed_chunks = 0
            for fut in as_completed(futures):
                chunk_idx = futures[fut]
                try:
                    chunk_stats = fut.result()
                except Exception:
                    chunk_stats = {"error": 0}
                    chunk_stats["error"] = 1

                for k, v in chunk_stats.items():
                    stats[k] = stats.get(k, 0) + v

                completed_chunks += 1
                print(
                    f"   -> Chunk {completed_chunks}/{CPU_THREADS} done for {symbol} "
                    f"(ok={stats['ok']}, skip={stats['skip']}, "
                    f"missing={stats['missing']}, error={stats['error']})"
                )
    finally:
        gc.enable()

    print(f"   -> Final stats for {symbol}: {stats}")

# ---------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------


def _signal_handler(signum: int, frame) -> None:
    _stop_event.set()


def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)

    base = Path(str(CONFIG["BASE_DIR"]))
    if not base.exists():
        print(f"[error] Base data directory '{base}' not found.")
        return

    # Discover symbols (top-level dirs under BASE_DIR)
    symbols = [
        d.name for d in base.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    symbols.sort()

    if not symbols:
        print(f"[error] No symbols found under '{base}'. Run data.py first.")
        return

    print(f"=== Universal Updater ===")
    print(f"Found {len(symbols)} symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
    print(f"Using {CPU_THREADS} worker threads.\n")

    for idx, symbol in enumerate(symbols, start=1):
        if _stop_event.is_set():
            break
        print(f"[{idx}/{len(symbols)}] Symbol: {symbol}")
        _update_symbol(symbol)

    print("\n[Done] All symbols processed.")


if __name__ == "__main__":
    main()
