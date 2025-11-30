"""
data.py (Universal Edition)

High-Frequency Data Engine for Python 3.14t (Free-Threaded)
Target: Any Multi-Core CPU (Intel/AMD/ARM) running Python 3.14t
"""

import sys
import os
import gc
import time
import struct
import datetime as dt
import http.client
import ssl
import signal
import zipfile
import re
import threading
from io import BytesIO, TextIOWrapper
from csv import reader as csv_reader
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# 0. Runtime Environment Check
# ==============================================================================

if sys._is_gil_enabled():
    print("[FATAL] This script requires Python 3.14t with GIL DISABLED.", file=sys.stderr)
    sys.exit(1)

try:
    import compression.zstd as zstd
except ImportError:
    print("[FATAL] Native 'compression.zstd' not found.", file=sys.stderr)
    sys.exit(1)

# Auto-detect cores (e.g., 24 on Ryzen 7900X)
SYSTEM_THREADS = os.cpu_count() or 4

def _print_system_info():
    print(f"[system] Detected {SYSTEM_THREADS} Logical Cores.")
    print(f"[system] Runtime: {sys.version.split()[0]} (Free-Threaded)")

# ==============================================================================
# 1. Universal Configuration  <-- LOOK HERE TO EDIT
# ==============================================================================

CONFIG: dict[str, object] = {
    # ----------------------------------------------------------------------
    # [USER EDITABLE] Target Asset
    # Change "BTCUSDT" to "ETHUSDT", "SOLUSDT", etc.
    # ----------------------------------------------------------------------
    "SYMBOL": "ETHUSDT",  # <<< CHANGE THIS for different coins

    "BASE_DIR": "data",

    # ----------------------------------------------------------------------
    # [USER EDITABLE] Data Source / Market Type
    # Options:
    #   - USD-M Futures: "data/futures/um"  (Default)
    #   - COIN-M Futures: "data/futures/cm"
    #   - Spot Market:    "data/spot"
    # ----------------------------------------------------------------------
    "S3_PREFIX": "data/futures/um", # <<< CHANGE THIS for Spot/Futures
    
    "HOST_DATA": "data.binance.vision",
    "DATASET":   "aggTrades",

    # ----------------------------------------------------------------------
    # [USER EDITABLE] Start Date
    # If downloading a new coin (e.g., PEPE), set this to 2023-01-01
    # to avoid scanning years of non-existent data.
    # ----------------------------------------------------------------------
    "FALLBACK_DATE": dt.date(2020, 1, 1), # <<< CHANGE THIS if coin is new

    # Dynamic Tuning (Auto-scales to your CPU)
    "WORKERS": SYSTEM_THREADS,
    "ROWS_PER_CHUNK": 50_000,

    # Network Resilience
    "TIMEOUT": 15,
    "RETRIES": 5,
    "BACKOFF": 0.5,
    "VERIFY_SSL": True,
    "USER_AGENT": f"QuantEngine/3.14t (Auto-Scale: {SYSTEM_THREADS} cores)",
}

UTC = dt.timezone.utc
_thread_local = threading.local()
_stop_event = threading.Event()

_ssl_context = ssl.create_default_context()
if not CONFIG["VERIFY_SSL"]:
    _ssl_context.check_hostname = False
    _ssl_context.verify_mode = ssl.CERT_NONE

# ==============================================================================
# 2. Binary Schema (Fixed Standard)
# ==============================================================================

PX_SCALE = 100_000_000
QT_SCALE = 100_000_000

AGG_ROW_STRUCT = struct.Struct("<QQQQHHqB3x")
AGG_ROW_SIZE = AGG_ROW_STRUCT.size
AGG_HDR_STRUCT = struct.Struct("<4sBBHQqq16x")
AGG_HDR_MAGIC = b"AGG2"
INDEX_ROW_STRUCT = struct.Struct("<HQQ")
INDEX_ROW_SIZE = INDEX_ROW_STRUCT.size
FLAG_IS_BUYER_MAKER = 1 << 0

# ==============================================================================
# 3. Thread-Safe Locking & Storage
# ==============================================================================

_month_locks: dict[tuple[int, int], threading.Lock] = {}
_guard = threading.Lock()

def _get_month_lock(year: int, month: int) -> threading.Lock:
    key = (year, month)
    with _guard:
        return _month_locks.setdefault(key, threading.Lock())

def _ensure_paths(year: int, month: int) -> tuple[str, str, str]:
    base_dir = str(CONFIG["BASE_DIR"])
    symbol = str(CONFIG["SYMBOL"])
    dir_path = os.path.join(base_dir, symbol, f"{year:04d}", f"{month:02d}")
    os.makedirs(dir_path, exist_ok=True)
    return (
        dir_path,
        os.path.join(dir_path, "data.quantdev"),
        os.path.join(dir_path, "index.quantdev"),
    )

def _is_day_indexed(year: int, month: int, day: int) -> bool:
    _, data_path, index_path = _ensure_paths(year, month)
    
    if not os.path.exists(index_path) or not os.path.exists(data_path):
        return False

    try:
        data_size = os.path.getsize(data_path)
        with open(index_path, "rb") as f:
            while True:
                chunk = f.read(INDEX_ROW_SIZE)
                if not chunk or len(chunk) < INDEX_ROW_SIZE:
                    break
                d, offset, length = INDEX_ROW_STRUCT.unpack(chunk)
                if d == day:
                    if (offset + length) <= data_size:
                        return True
        return False
    except OSError:
        return False

# ==============================================================================
# 4. Universal Networking (Persistent)
# ==============================================================================

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
    conns = getattr(_thread_local, "conns", {})
    if host in conns:
        try:
            conns[host].close()
        except Exception:
            pass
        del conns[host]

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
            
            if resp.status == 200:
                return resp.read()
            elif resp.status == 404:
                resp.read() # Drain
                return None
            
            resp.read()
        except (http.client.HTTPException, OSError, ssl.SSLError):
            _close_connection(host)
        
        if attempt < retries:
            time.sleep(backoff * (2 ** (attempt - 1)))
            
    return None

# ==============================================================================
# 5. Genesis Detection
# ==============================================================================

def _fetch_genesis_date(symbol: str) -> dt.date:
    host = str(CONFIG["HOST_DATA"])
    dataset = str(CONFIG["DATASET"])
    prefix = f"{CONFIG['S3_PREFIX']}/daily/{dataset}/{symbol}/"
    path = f"/?prefix={prefix}&delimiter=/"

    print(f"[init] Probing index: {host}{path}")
    html_bytes = _http_request(host, "GET", path)

    fallback = CONFIG["FALLBACK_DATE"]
    if not isinstance(fallback, dt.date): fallback = dt.date(2020, 1, 1)

    if not html_bytes:
        print(f"[warn] Index unreachable. Defaulting to {fallback}")
        return fallback

    text = html_bytes.decode("utf-8", "replace")
    
    # Matches SYMBOL-DATASET-YYYY-MM-DD
    pattern = rf"{re.escape(symbol)}-{re.escape(dataset)}-(\d{{4}}-\d{{2}}-\d{{2}})"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    if not matches:
        print(f"[warn] No files found in index. Defaulting to {fallback}")
        return fallback

    dates = []
    for date_str in matches:
        try:
            dates.append(dt.datetime.strptime(date_str, "%Y-%m-%d").date())
        except ValueError:
            continue

    if not dates:
        return fallback

    return min(dates)

# ==============================================================================
# 6. Core Logic
# ==============================================================================

def _download_day_zip(day: dt.date) -> bytes | None:
    year, month, day_str = day.year, f"{day.month:02d}", f"{day.day:02d}"
    sym = str(CONFIG["SYMBOL"])
    dataset = str(CONFIG["DATASET"])
    prefix = str(CONFIG["S3_PREFIX"])
    
    path = (
        f"/{prefix}/daily/{dataset}/{sym}/"
        f"{sym}-{dataset}-{year}-{month}-{day_str}.zip"
    )
    return _http_request(str(CONFIG["HOST_DATA"]), "GET", path)

def _process_zip_to_agg2(day: dt.date, zip_bytes: bytes) -> bytes | None:
    buf_size = AGG_ROW_SIZE * int(CONFIG["ROWS_PER_CHUNK"])
    
    buf = getattr(_thread_local, "buf", None)
    if buf is None or len(buf) != buf_size:
        buf = bytearray(buf_size)
        _thread_local.buf = buf

    view = memoryview(buf)
    pack_into = AGG_ROW_STRUCT.pack_into
    chunks = []
    c_min, c_max, total_count = 2**63 - 1, -2**63, 0

    try:
        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names: return None
            
            with zf.open(csv_names[0], "r") as f_in:
                wrapper = TextIOWrapper(f_in, encoding="utf-8", newline="")
                reader = csv_reader(wrapper)
                first_row = next(reader, None)
                if not first_row: return None
                
                headers = [h.replace("_", "").lower().strip() for h in first_row]
                idx = {"id": 0, "px": 1, "qt": 2, "fi": 3, "li": 4, "ts": 5, "bm": 6}
                
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

                def _process(rows):
                    nonlocal offset, total_count, c_min, c_max
                    for row in rows:
                        if not row: continue
                        try:
                            ts = int(row[idx["ts"]])
                            if ts < c_min: c_min = ts
                            if ts > c_max: c_max = ts
                            
                            fi, li = int(row[idx["fi"]]), int(row[idx["li"]])
                            cnt = li - fi + 1
                            if cnt < 0: continue
                            if cnt > 65535: cnt = 65535

                            px = int(float(row[idx["px"]]) * PX_SCALE)
                            qt = int(float(row[idx["qt"]]) * QT_SCALE)
                            is_maker = row[idx["bm"]].lower() in ("true", "1", "t")

                            pack_into(view, offset, 
                                      int(row[idx["id"]]), px, qt, fi, cnt, 
                                      FLAG_IS_BUYER_MAKER if is_maker else 0, 
                                      ts, 
                                      0 if is_maker else 1)
                            
                            offset += AGG_ROW_SIZE
                            total_count += 1
                            if offset >= buf_size:
                                chunks.append(view[:offset].tobytes())
                                offset = 0
                        except (ValueError, IndexError): continue

                if "aggtradeid" not in headers and "price" not in headers:
                    _process([first_row])
                _process(reader)
                
                if offset > 0: chunks.append(view[:offset].tobytes())

    except Exception as e:
        print(f"[parse-err] {day}: {e}", file=sys.stderr)
        return None

    if total_count == 0: return b""
    hdr = AGG_HDR_STRUCT.pack(AGG_HDR_MAGIC, 1, day.day, 0, total_count, c_min, c_max)
    return hdr + b"".join(chunks)

def _worker_task(day: dt.date) -> str:
    if _stop_event.is_set(): return "stop"
    
    y, m, d = day.year, day.month, day.day
    if _is_day_indexed(y, m, d): return "skip"
    
    zip_data = _download_day_zip(day)
    if _stop_event.is_set(): return "stop"
    if zip_data is None: return "missing"
    
    agg_blob = _process_zip_to_agg2(day, zip_data)
    if not agg_blob: return "error" if agg_blob is None else "missing"
    
    try: c_blob = zstd.compress(agg_blob, level=3)
    except Exception: return "error"
    
    blob_len = len(c_blob)
    lock = _get_month_lock(y, m)
    _, d_path, i_path = _ensure_paths(y, m)
    
    try:
        with lock:
            if _is_day_indexed(y, m, d): return "skip"
            
            if not os.path.exists(d_path):
                with open(d_path, "ab"): pass
            
            start_offset = os.path.getsize(d_path)
            
            with open(i_path, "ab") as f_idx:
                f_idx.write(INDEX_ROW_STRUCT.pack(d, start_offset, blob_len))
                f_idx.flush()
                os.fsync(f_idx.fileno())
            
            with open(d_path, "ab") as f_data:
                f_data.write(c_blob)
                f_data.flush()
                os.fsync(f_data.fileno())
                
        return "ok"
    except OSError as e:
        print(f"[io-err] {day}: {e}", file=sys.stderr)
        return "error"

# ==============================================================================
# 7. Main Entry
# ==============================================================================

def main() -> None:
    signal.signal(signal.SIGINT, lambda s, f: _stop_event.set())
    
    _print_system_info()
    print(f"[job] Symbol: {CONFIG['SYMBOL']}")
    print(f"[job] Source: {CONFIG['HOST_DATA']}")

    start_date = _fetch_genesis_date(str(CONFIG["SYMBOL"]))
    print(f"[job] Start Date: {start_date}")

    end_date = dt.datetime.now(UTC).date() - dt.timedelta(days=1)
    if end_date < start_date:
        print("[info] Up to date (End date < Start date).")
        return

    days = [start_date + dt.timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    print(f"[job] Queue: {len(days)} days")

    stats = {"ok": 0, "skip": 0, "missing": 0, "error": 0, "stop": 0}
    gc.disable()
    start_time = time.perf_counter()
    
    try:
        with ThreadPoolExecutor(max_workers=int(CONFIG["WORKERS"])) as pool:
            future_to_day = {pool.submit(_worker_task, d): d for d in days}
            completed = 0
            
            for future in as_completed(future_to_day):
                day = future_to_day[future]
                try: 
                    res = future.result()
                except Exception as e:
                    print(f"\n[err] {day}: {e}", file=sys.stderr)
                    res = "error"
                
                stats[res] = stats.get(res, 0) + 1
                completed += 1
                
                char = {"ok": "D", "skip": ".", "missing": "_", "error": "!", "stop": "S"}.get(res, "?")
                sys.stdout.write(char)
                if completed % 100 == 0: sys.stdout.write(f" | {completed}\n")
                sys.stdout.flush()
                
                if completed % 1000 == 0: gc.collect()

    finally:
        gc.enable()
        elapsed = time.perf_counter() - start_time
        print(f"\n\n[Done] Time: {elapsed:.2f}s")
        print(f"[Stats] {stats}")

if __name__ == "__main__":
    main()