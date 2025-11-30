"""
build_bars.py
Feature Engineering Engine.
Reads Raw Data (using backtest.scan_dataset) -> Writes .bars files.
"""
import sys
import math
import time
from concurrent.futures import ThreadPoolExecutor

if sys._is_gil_enabled():
    raise RuntimeError("Run with -X gil=0")

import config
import backtest # Reusing discovery logic
try:
    import compression.zstd as zstd
except ImportError:
    print("Missing compression.zstd")
    sys.exit(1)

def process_day(job):
    src_path, off, ln, date_str = job
    dst_path = src_path.replace("data.quantdev", "data.bars")
    
    try:
        with open(src_path, "rb") as f:
            f.seek(off)
            blob = f.read(ln)
        raw_bytes = zstd.decompress(blob)
    except Exception: return "err_read"

    # Iterators
    iter_rows = config.AGG_ROW_STRUCT.iter_unpack
    raw_data = raw_bytes[config.AGG_HDR_SIZE:]
    
    # State
    bar_buffer = bytearray()
    
    b_open = 0.0
    b_high = -1.0
    b_low = 1e9
    b_vol = 0.0
    b_buy = 0.0
    b_sell = 0.0
    b_ts_start = 0.0
    
    path_len = 0.0
    last_px = 0.0
    
    # Constants
    TARGET = config.DELTA_THRESHOLD * config.QT_SCALE
    INV_PX = 1.0 / config.PX_SCALE
    INV_QT = 1.0 / config.QT_SCALE
    
    count = 0
    
    for r in iter_rows(raw_data):
        # r: id, px, qt, fi, cnt, flags, ts, pad
        px, qt, ts = r[1], r[2], r[6]
        
        if b_open == 0.0:
            b_open = px
            b_high = px
            b_low = px
            b_ts_start = ts
            last_px = px
            path_len = 0.0

        if px > b_high: b_high = px
        if px < b_low: b_low = px
        
        path_len += abs(px - last_px)
        last_px = px
        
        b_vol += qt
        # flags&1 == 1 -> Maker Buyer -> Taker Seller
        if (r[5] & 1): b_sell += qt
        else: b_buy += qt
        
        # Check Delta Trigger
        net_delta = b_buy - b_sell
        
        if abs(net_delta) >= TARGET:
            # Close Bar
            b_close = px
            
            o, h, l, c = b_open * INV_PX, b_high * INV_PX, b_low * INV_PX, b_close * INV_PX
            v, d = b_vol * INV_QT, net_delta * INV_QT
            
            # Metrics
            dist = abs(b_close - b_open)
            eff = (dist / path_len) if path_len > 0 else 0.0
            imp = ((c - o) / d) if abs(d) > 1e-9 else 0.0
            
            bar_buffer.extend(config.BAR_STRUCT.pack(
                float(b_ts_start), float(ts), o, h, l, c, v, d, eff, imp
            ))
            count += 1
            
            # Reset
            b_open = 0.0
            b_vol = 0.0
            b_buy = 0.0
            b_sell = 0.0
            
    if count > 0:
        try:
            with open(dst_path, "wb") as f:
                f.write(bar_buffer)
            return "ok"
        except: return "err_write"
        
    return "empty"

def run():
    print(f"--- Building Fractal Bars (Threshold: {config.DELTA_THRESHOLD} BTC) ---")
    years = backtest.discover_years(config.SYMBOL)
    jobs = backtest.scan_dataset(config.SYMBOL, years=years)
    
    print(f"[Builder] Processing {len(jobs)} days...")
    t0 = time.perf_counter()
    stats = {"ok": 0, "empty": 0, "err": 0}
    
    with ThreadPoolExecutor(max_workers=config.WORKERS) as pool:
        futures = [pool.submit(process_day, j) for j in jobs]
        for f in futures:
            res = f.result()
            if res == "ok": stats["ok"] += 1
            elif res == "empty": stats["empty"] += 1
            else: stats["err"] += 1
            
    print(f"[Done] {time.perf_counter()-t0:.2f}s | Stats: {stats}")

if __name__ == "__main__":
    run()