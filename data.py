#!/usr/bin/env python3.14t
"""
data.py (v3.3 - Free-Threaded, AGG3, Index v1 + Checksums)

Target runtime:
- Python 3.14t, GIL disabled (-X gil=0)
- Windows 11
- AMD Ryzen 9 7900X (12C / 24T)

Features:
- AGG3 row format:
    * 48 bytes, fixed width, 8-byte aligned row size
    * trade_id(Q), px_scaled(Q), qty_scaled(Q), first_id(Q),
      trade_count(I, 32-bit), flags(H), ts_ms(q), pad(2x)
- Header per day with magic/version/day/zstd_level/row_count/min_ts/max_ts.
- Index v1 with header + 64-bit checksum of uncompressed day blob.
- Backward compatibility with legacy v0 index.
- Free-threaded Python 3.14t only, using compression.zstd.
"""

import os
import time
import struct
import datetime as dt
import http.client
import ssl
import signal
import zipfile
import hashlib
import sys
import threading
from io import BytesIO, TextIOWrapper
from csv import reader as csv_reader
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# 0. Mandatory 3.14t / Free-threaded boilerplate
# ---------------------------------------------------------------------------

if sys._is_gil_enabled():
    raise RuntimeError("Performance Critical: Must run on Python 3.14t with -X gil=0")

CPU_THREADS = 24

try:
    import compression.zstd as zstd
except ImportError as exc:
    raise RuntimeError("Native 'compression.zstd' (PEP 784) is required.") from exc

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

CONFIG: dict[str, object] = {
    "SYMBOL": "BTCUSDT",
    "BASE_DIR": "data",
    
    # Binance Data Source
    "HOST_DATA": "data.binance.vision",
    "S3_PREFIX": "data/futures/um",
    "DATASET": "aggTrades",

    "FALLBACK_DATE": dt.date(2020, 1, 1),
    
    "WORKERS": CPU_THREADS,
    "ROWS_PER_CHUNK": 50_000,
    
    "TIMEOUT": 10,
    "RETRIES": 5,
    "BACKOFF": 0.5,
    "VERIFY_SSL": True,
    "USER_AGENT": "QuantEngine/3.14t data.py",
    
    "ZSTD_LEVEL": 3,
}

# ---------------------------------------------------------------------------
# 2. Binary Schema (AGG3)
# ---------------------------------------------------------------------------

PX_SCALE = 100_000_000
QT_SCALE = 100_000_000

# Row: trade_id(Q), px(Q), qt(Q), fi(Q), count(I), flags(H), ts(q), pad(2x)
AGG_ROW_STRUCT = struct.Struct("<QQQQIHq2x")
AGG_ROW_SIZE = AGG_ROW_STRUCT.size
assert AGG_ROW_SIZE == 48

# Header: magic(4s), ver(B), day(B), z_lvl(H), count(Q), min_ts(q), max_ts(q), pad(16x)
AGG_HDR_STRUCT = struct.Struct("<4sBBHQqq16x")
AGG_HDR_MAGIC = b"AGG3"
assert AGG_HDR_STRUCT.size == 48

# Index Header: magic(4s), ver(I), count(Q)
INDEX_MAGIC = b"QIDX"
INDEX_VERSION = 1
INDEX_HDR_STRUCT = struct.Struct("<4sIQ")
INDEX_HDR_SIZE = INDEX_HDR_STRUCT.size

# Index Rows
INDEX_ROW_STRUCT_V0 = struct.Struct("<HQQ")    # Legacy
INDEX_ROW_STRUCT_V1 = struct.Struct("<HQQQ")   # v1 with checksum
INDEX_ROW_SIZE_V0 = INDEX_ROW_STRUCT_V0.size
INDEX_ROW_SIZE_V1 = INDEX_ROW_STRUCT_V1.size

FLAG_IS_BUYER_MAKER = 1 << 0

# ---------------------------------------------------------------------------
# 3. Global State & Locks
# ---------------------------------------------------------------------------

_thread_local = threading.local()
_stop_event = threading.Event()

# Fine-grained locking: (Year, Month) -> Lock
_month_locks: dict[tuple[int, int], threading.Lock] = {}
_guard = threading.Lock()

_ssl_context = ssl.create_default_context()
if not CONFIG["VERIFY_SSL"]:
    _ssl_context.check_hostname = False
    _ssl_context.verify_mode = ssl.CERT_NONE


def _get_month_lock(year: int, month: int) -> threading.Lock:
    key = (year, month)
    # Optimistic check
    if key in _month_locks:
        return _month_locks[key]
    with _guard:
        # Double-check
        if key not in _month_locks:
            _month_locks[key] = threading.Lock()
        return _month_locks[key]


def _ensure_paths(year: int, month: int) -> tuple[str, str, str]:
    base_dir = str(CONFIG["BASE_DIR"])
    symbol = str(CONFIG["SYMBOL"])
    dir_path = os.path.join(base_dir, symbol, f"{year:04d}", f"{month:02d}")
    os.makedirs(dir_path, exist_ok=True)
    data_path = os.path.join(dir_path, "data.quantdev")
    index_path = os.path.join(dir_path, "index.quantdev")
    return dir_path, data_path, index_path

# ---------------------------------------------------------------------------
# 3a. Index & Integrity Helpers
# ---------------------------------------------------------------------------

def _checksum64(data: bytes) -> int:
    """64-bit checksum using blake2b (stdlib)."""
    h = hashlib.blake2b(digest_size=8)
    h.update(data)
    return int.from_bytes(h.digest(), "little")


def _detect_index_layout(index_path: str) -> int:
    """Returns 0 for legacy/missing, 1 for v1 headered."""
    try:
        size = os.path.getsize(index_path)
        if size < INDEX_HDR_SIZE:
            return 0
        with open(index_path, "rb") as f:
            head = f.read(INDEX_HDR_SIZE)
        if len(head) != INDEX_HDR_SIZE:
            return 0
        magic, version, _ = INDEX_HDR_STRUCT.unpack(head)
        if magic == INDEX_MAGIC and version == INDEX_VERSION:
            return 1
        return 0
    except OSError:
        return 0


def _init_index_v1_if_needed(index_path: str) -> None:
    """Create v1 index header if file missing or empty."""
    try:
        if not os.path.exists(index_path) or os.path.getsize(index_path) == 0:
            with open(index_path, "wb") as f:
                f.write(INDEX_HDR_STRUCT.pack(INDEX_MAGIC, INDEX_VERSION, 0))
    except OSError:
        pass


def _increment_index_v1_count(index_path: str, delta: int) -> None:
    """Update record_count in v1 header by +delta."""
    try:
        with open(index_path, "r+b") as f:
            head = f.read(INDEX_HDR_SIZE)
            if len(head) != INDEX_HDR_SIZE:
                return
            magic, version, count = INDEX_HDR_STRUCT.unpack(head)
            if magic == INDEX_MAGIC and version == INDEX_VERSION:
                count += delta
                f.seek(0)
                f.write(INDEX_HDR_STRUCT.pack(magic, version, count))
                f.flush()
                os.fsync(f.fileno())
    except OSError:
        pass


def _is_day_indexed(year: int, month: int, day: int) -> bool:
    """Check if index contains a valid entry for (year, month, day)."""
    _, data_path, index_path = _ensure_paths(year, month)
    if not os.path.exists(index_path) or not os.path.exists(data_path):
        return False

    try:
        data_size = os.path.getsize(data_path)
    except OSError:
        return False

    layout = _detect_index_layout(index_path)
    row_struct = INDEX_ROW_STRUCT_V1 if layout == 1 else INDEX_ROW_STRUCT_V0
    row_size = INDEX_ROW_SIZE_V1 if layout == 1 else INDEX_ROW_SIZE_V0
    start_offset = INDEX_HDR_SIZE if layout == 1 else 0

    try:
        with open(index_path, "rb") as f:
            f.seek(start_offset)
            while True:
                raw = f.read(row_size)
                if len(raw) < row_size:
                    break
                
                if layout == 1:
                    d, off, length, _ = row_struct.unpack(raw)
                else:
                    d, off, length = row_struct.unpack(raw)
                
                if d == day:
                    if (off + length) <= data_size:
                        return True
                    return False
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
            host, context=_ssl_context, timeout=float(CONFIG["TIMEOUT"])
        )
        conns[host] = conn
    return conn


def _close_connection(host: str) -> None:
    conns = getattr(_thread_local, "conns", None)
    if conns:
        conn = conns.pop(host, None)
        if conn:
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
        
        try:
            conn = _get_connection(host)
            conn.request(method, path, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            resp.close()

            if resp.status == 200:
                return data
            if resp.status == 404:
                return None

        except (http.client.HTTPException, OSError, ssl.SSLError):
            _close_connection(host)

        if attempt < retries:
            time.sleep(backoff * (2 ** (attempt - 1)))

    return None

# ---------------------------------------------------------------------------
# 5. Zip -> AGG3
# ---------------------------------------------------------------------------

def _process_zip_to_agg3(day: dt.date, zip_bytes: bytes) -> bytes | None:
    buf_size = AGG_ROW_SIZE * int(CONFIG["ROWS_PER_CHUNK"])
    
    # Thread-local buffer reuse to minimize allocs
    buf = getattr(_thread_local, "buf", None)
    if buf is None or len(buf) != buf_size:
        buf = bytearray(buf_size)
        _thread_local.buf = buf
    
    view = memoryview(buf)
    pack_into = AGG_ROW_STRUCT.pack_into
    chunks: list[bytes] = []
    
    c_min = 2**63 - 1
    c_max = -2**63
    total_count = 0

    try:
        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return None
            
            with zf.open(csv_names[0], "r") as f_in:
                wrapper = TextIOWrapper(f_in, encoding="utf-8", newline="")
                reader = csv_reader(wrapper)
                
                headers = next(reader, None)
                if not headers:
                    return None
                
                # Normalize headers
                h_map = {h.replace("_", "").lower().strip(): i for i, h in enumerate(headers)}
                
                idx_id = h_map.get("aggtradeid", h_map.get("id", 0))
                idx_px = h_map.get("price", 1)
                idx_qt = h_map.get("quantity", 2)
                idx_fi = h_map.get("firsttradeid", 3)
                idx_li = h_map.get("lasttradeid", 4)
                idx_ts = h_map.get("transacttime", 5)
                idx_bm = h_map.get("isbuyermaker", 6)
                
                offset = 0
                
                for row in reader:
                    if not row:
                        continue
                    try:
                        ts = int(row[idx_ts])
                        if ts < c_min: c_min = ts
                        if ts > c_max: c_max = ts
                        
                        fi = int(row[idx_fi])
                        li = int(row[idx_li])
                        cnt = li - fi + 1
                        if cnt <= 0: continue
                        if cnt > 0xFFFFFFFF: cnt = 0xFFFFFFFF
                        
                        # Use round to correct floating point drift before int cast
                        px = int(round(float(row[idx_px]) * PX_SCALE))
                        qt = int(round(float(row[idx_qt]) * QT_SCALE))
                        
                        is_maker = row[idx_bm].lower() in ("true", "1", "t")
                        flags = FLAG_IS_BUYER_MAKER if is_maker else 0
                        
                        pack_into(view, offset, int(row[idx_id]), px, qt, fi, cnt, flags, ts)
                        offset += AGG_ROW_SIZE
                        total_count += 1
                        
                        if offset >= buf_size:
                            chunks.append(view[:offset].tobytes())
                            offset = 0
                            
                    except (ValueError, IndexError):
                        continue
                
                if offset > 0:
                    chunks.append(view[:offset].tobytes())
                    
    except Exception:
        return None
    
    if total_count == 0:
        return b""
        
    hdr = AGG_HDR_STRUCT.pack(
        AGG_HDR_MAGIC, 1, day.day, int(CONFIG["ZSTD_LEVEL"]), total_count, c_min, c_max
    )
    return hdr + b"".join(chunks)

# ---------------------------------------------------------------------------
# 6. Per-day worker
# ---------------------------------------------------------------------------

def _process_day(day: dt.date) -> str:
    if _stop_event.is_set():
        return "stop"
    
    y, m, d = day.year, day.month, day.day
    
    # 1. Check index (fast path)
    if _is_day_indexed(y, m, d):
        return "skip"

    # 2. Download
    try:
        year_str = f"{y}"
        month_str = f"{m:02d}"
        day_str = f"{d:02d}"
        sym = str(CONFIG["SYMBOL"])
        ds = str(CONFIG["DATASET"])
        pfx = str(CONFIG["S3_PREFIX"])
        path = f"/{pfx}/daily/{ds}/{sym}/{sym}-{ds}-{year_str}-{month_str}-{day_str}.zip"
        
        zip_data = _http_request(str(CONFIG["HOST_DATA"]), "GET", path)
    except Exception:
        return "error"
        
    if _stop_event.is_set():
        return "stop"
    if zip_data is None:
        return "missing"
    
    # 3. Process ZIP -> AGG3
    agg_blob = _process_zip_to_agg3(day, zip_data)
    if agg_blob is None:
        return "error"
    if len(agg_blob) == 0:
        return "missing" # Empty CSV

    # 4. Checksum & Compress
    c_sum = _checksum64(agg_blob)
    try:
        c_blob = zstd.compress(agg_blob, level=int(CONFIG["ZSTD_LEVEL"]))
    except Exception:
        return "error"
        
    blob_len = len(c_blob)
    
    # 5. Write (Critical Section)
    lock = _get_month_lock(y, m)
    _, data_path, index_path = _ensure_paths(y, m)
    
    try:
        with lock:
            if _is_day_indexed(y, m, d):
                return "skip"

            if not os.path.exists(data_path):
                with open(data_path, "wb"): pass
            
            layout = _detect_index_layout(index_path)
            if layout == 0:
                _init_index_v1_if_needed(index_path)
                layout = _detect_index_layout(index_path)

            # Append Data
            with open(data_path, "ab") as f_data:
                start_offset = f_data.tell()
                f_data.write(c_blob)
                f_data.flush()
                os.fsync(f_data.fileno())

            # Append Index
            if layout == 1:
                row = INDEX_ROW_STRUCT_V1.pack(d, start_offset, blob_len, c_sum)
            else:
                row = INDEX_ROW_STRUCT_V0.pack(d, start_offset, blob_len)
                
            with open(index_path, "ab") as f_idx:
                f_idx.write(row)
                f_idx.flush()
                os.fsync(f_idx.fileno())
                
            if layout == 1:
                _increment_index_v1_count(index_path, 1)
        
        return "ok"
    except OSError:
        return "error"

def _worker_chunk(days: list[dt.date]) -> dict[str, int]:
    stats: dict[str, int] = {"ok": 0, "skip": 0, "missing": 0, "error": 0, "stop": 0}
    for day in days:
        if _stop_event.is_set():
            stats["stop"] += 1
            break
        res = _process_day(day)
        stats[res] = stats.get(res, 0) + 1
    return stats

# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def _split_into_chunks(seq: list[dt.date], n: int) -> list[list[dt.date]]:
    k, m = divmod(len(seq), n)
    return [seq[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def main() -> None:
    signal.signal(signal.SIGINT, lambda s, f: _stop_event.set())
    
    print(f"--- data.py (3.3 / Free-Threaded) | Symbol: {CONFIG['SYMBOL']} ---")
    
    s_date = CONFIG["FALLBACK_DATE"]
    assert isinstance(s_date, dt.date)
    e_date = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
    
    if e_date < s_date:
        print("Nothing to do.")
        return

    days = [s_date + dt.timedelta(days=i) for i in range((e_date - s_date).days + 1)]
    chunks = _split_into_chunks(days, CPU_THREADS)
    
    print(f"[job] {len(days)} days -> {CPU_THREADS} threads.")
    
    t0 = time.perf_counter()
    stats_total: dict[str, int] = {}
    
    with ThreadPoolExecutor(max_workers=CPU_THREADS) as ex:
        futs = [ex.submit(_worker_chunk, c) for c in chunks]
        
        done_cnt = 0
        for f in as_completed(futs):
            for k, v in f.result().items():
                stats_total[k] = stats_total.get(k, 0) + v
            done_cnt += 1
            print(f"[status] Chunks: {done_cnt}/{CPU_THREADS} | Stats: {stats_total}", end="\r")
            
    print(f"\n[done] {time.perf_counter() - t0:.2f}s | Final: {stats_total}")

if __name__ == "__main__":
    main()