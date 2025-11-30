#!/usr/bin/env python3.14t
"""
data.py (v3.0 - Free-Threaded, Atomic-ish, No Self-Healing)

Target runtime:
- Python 3.14t, GIL disabled (-X gil=0)
- Windows 11
- AMD Ryzen 9 7900X (12C / 24T)

Constraints:
- Standard library only
- CPU-bound = threading / ThreadPoolExecutor (no multiprocessing)
- Compression via compression.zstd only

Properties:
- Uses file.tell() for index offsets under a per-month lock.
- Month-scoped locks serialize writes while still allowing parallel processing of many days.
- No self-healing/truncation: crashes may leave unreachable bytes at the end of data files,
  but index entries never point to non-existent bytes.
"""

import os
import time
import struct
import datetime as dt
import http.client
import ssl
import signal
import zipfile
from io import BytesIO, TextIOWrapper
from csv import reader as csv_reader

import sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# 0. Mandatory 3.14t / Free-threaded boilerplate
# ---------------------------------------------------------------------------

if sys._is_gil_enabled():
    raise RuntimeError("GIL must be disabled (run Python 3.14t with -X gil=0)")

CPU_THREADS = 24

try:
    import compression.zstd as zstd
except ImportError as exc:
    raise RuntimeError("Native 'compression.zstd' (PEP 784) is required.") from exc

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

CONFIG: dict[str, object] = {
    # Change this to the symbol you want
    "SYMBOL": "BTCUSDT",

    # Directory where data will be saved: ./data/SYMBOL/YYYY/MM/...
    "BASE_DIR": "data",

    # Binance Data Source Settings (UM Futures aggTrades)
    "HOST_DATA": "data.binance.vision",
    "S3_PREFIX": "data/futures/um",
    "DATASET": "aggTrades",

    # Start Date: you can change this to tighten the backfill window
    "FALLBACK_DATE": dt.date(2020, 1, 1),

    # Threads
    "WORKERS": CPU_THREADS,
    "ROWS_PER_CHUNK": 50_000,

    # Network
    "TIMEOUT": 10,
    "RETRIES": 5,
    "BACKOFF": 0.5,
    "VERIFY_SSL": True,
    "USER_AGENT": "QuantEngine/3.14t data.py",
}

# ---------------------------------------------------------------------------
# 2. Binary Schema (must match update.py)
# ---------------------------------------------------------------------------

PX_SCALE = 100_000_000
QT_SCALE = 100_000_000

# Per-row payload:
#   trade_id, price_scaled, qty_scaled, first_trade_id, count (H),
#   flags (H), timestamp (q), side(B), padding
AGG_ROW_STRUCT = struct.Struct("<QQQQHHqB3x")
AGG_ROW_SIZE = AGG_ROW_STRUCT.size

# File header:
#   magic(4s) = b"AGG2"
#   version(B)
#   day(B)
#   reserved(H)
#   row_count(Q)
#   min_ts(q)
#   max_ts(q)
#   padding(16x)
AGG_HDR_STRUCT = struct.Struct("<4sBBHQqq16x")
AGG_HDR_MAGIC = b"AGG2"

# Index row: day (H), offset (Q), length (Q)
INDEX_ROW_STRUCT = struct.Struct("<HQQ")
INDEX_ROW_SIZE = INDEX_ROW_STRUCT.size

FLAG_IS_BUYER_MAKER = 1 << 0

# ---------------------------------------------------------------------------
# 3. Global State & Locks
# ---------------------------------------------------------------------------

_thread_local = threading.local()
_stop_event = threading.Event()

_month_locks: dict[tuple[int, int], threading.Lock] = {}
_guard = threading.Lock()

_ssl_context = ssl.create_default_context()
if not CONFIG["VERIFY_SSL"]:
    _ssl_context.check_hostname = False
    _ssl_context.verify_mode = ssl.CERT_NONE


def _get_month_lock(year: int, month: int) -> threading.Lock:
    key = (year, month)
    with _guard:
        lock = _month_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _month_locks[key] = lock
        return lock


def _ensure_paths(year: int, month: int) -> tuple[str, str, str]:
    base_dir = str(CONFIG["BASE_DIR"])
    symbol = str(CONFIG["SYMBOL"])
    dir_path = os.path.join(base_dir, symbol, f"{year:04d}", f"{month:02d}")
    os.makedirs(dir_path, exist_ok=True)
    data_path = os.path.join(dir_path, "data.quantdev")
    index_path = os.path.join(dir_path, "index.quantdev")
    return dir_path, data_path, index_path


def _is_day_indexed(year: int, month: int, day: int) -> bool:
    """
    Returns True if index has an entry for this day AND that entry points to
    bytes fully contained in the data file. If an entry points beyond EOF,
    we treat it as 'not indexed' to allow overwrite on reruns.
    """
    _, data_path, index_path = _ensure_paths(year, month)

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
# 5. Transform Zip -> AGG2
# ---------------------------------------------------------------------------


def _download_day_zip(day: dt.date) -> bytes | None:
    year = day.year
    month_str = f"{day.month:02d}"
    day_str = f"{day.day:02d}"

    sym = str(CONFIG["SYMBOL"])
    dataset = str(CONFIG["DATASET"])
    prefix = str(CONFIG["S3_PREFIX"])

    path = (
        f"/{prefix}/daily/{dataset}/{sym}/"
        f"{sym}-{dataset}-{year}-{month_str}-{day_str}.zip"
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

                # If headers do not look like typical aggTrades, treat first row as data
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
        1,         # version
        day.day,   # day
        0,         # reserved
        total_count,
        c_min,
        c_max,
    )
    return hdr + b"".join(chunks)

# ---------------------------------------------------------------------------
# 6. Per-day worker (atomic-ish writer, no healing)
# ---------------------------------------------------------------------------


def _process_day(day: dt.date) -> str:
    if _stop_event.is_set():
        return "stop"

    y, m, d = day.year, day.month, day.day

    # Quick check before network
    if _is_day_indexed(y, m, d):
        return "skip"

    zip_data = _download_day_zip(day)
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
    lock = _get_month_lock(y, m)
    _, data_path, index_path = _ensure_paths(y, m)

    try:
        with lock:
            # Authoritative re-check under the month lock
            if _is_day_indexed(y, m, d):
                return "skip"

            # Ensure data file exists
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
# 7. Chunk worker (for exactly 24 chunks)
# ---------------------------------------------------------------------------


def _worker_chunk(days: list[dt.date]) -> dict[str, int]:
    stats: dict[str, int] = {"ok": 0, "skip": 0, "missing": 0, "error": 0, "stop": 0}
    for day in days:
        if _stop_event.is_set():
            stats["stop"] += 1
            break
        res = _process_day(day)
        stats[res] = stats.get(res, 0) + 1
        if res == "stop":
            break
    return stats

# ---------------------------------------------------------------------------
# 8. Utility: split work into exactly 24 chunks
# ---------------------------------------------------------------------------


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
        chunk = seq[idx:idx + size] if size > 0 else []
        chunks.append(chunk)
        idx += size
    return chunks

# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------


def _signal_handler(signum: int, frame) -> None:
    _stop_event.set()


def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)

    symbol = str(CONFIG["SYMBOL"])
    print(f"--- data.py (3.0) | Symbol: {symbol} ---")

    start_date = CONFIG["FALLBACK_DATE"]
    if not isinstance(start_date, dt.date):
        start_date = dt.date(2020, 1, 1)

    today_utc = dt.datetime.now(dt.timezone.utc).date()
    end_date = today_utc - dt.timedelta(days=1)

    if end_date < start_date:
        print("[init] Nothing to do: end_date < start_date")
        return

    total_days = (end_date - start_date).days + 1
    days: list[dt.date] = [
        start_date + dt.timedelta(days=i) for i in range(total_days)
    ]

    print(f"[job] Downloading {len(days)} days from {start_date} to {end_date}.")

    # Split into exactly 24 chunks (some may be empty)
    chunks = _split_into_chunks(days, CPU_THREADS)

    global_stats: dict[str, int] = {
        "ok": 0,
        "skip": 0,
        "missing": 0,
        "error": 0,
        "stop": 0,
    }

    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=CPU_THREADS) as executor:
        futures = [executor.submit(_worker_chunk, chunk) for chunk in chunks]

        completed_chunks = 0
        for fut in as_completed(futures):
            chunk_stats = fut.result()
            for k, v in chunk_stats.items():
                global_stats[k] = global_stats.get(k, 0) + v
            completed_chunks += 1
            # Optional simple progress indicator by chunk:
            print(f"[chunk] {completed_chunks}/{CPU_THREADS} completed")

    elapsed = time.perf_counter() - t0
    print(f"[done] {elapsed:.2f}s | stats={global_stats}")


if __name__ == "__main__":
    main()
