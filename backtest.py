"""
backtest.py
- Scans File System for Data.
- Runs 'algo.py' on pre-built .bars files.
"""
import sys
import os
import time
import struct
from concurrent.futures import ThreadPoolExecutor

if sys._is_gil_enabled():
    raise RuntimeError("Run with -X gil=0")

import config
import algo
import metrics

# (raw_path, offset, length, date_str)
JobType = tuple[str, int, int, str]

# --- DISCOVERY LOGIC (Since data.py is immutable/ETL-only) ---

def discover_years(symbol: str) -> list[int]:
    """Scans disk for available years."""
    base = os.path.join(config.BASE_DIR, symbol)
    if not os.path.exists(base):
        return []
    years = []
    for d in os.listdir(base):
        if d.isdigit() and os.path.isdir(os.path.join(base, d)):
            years.append(int(d))
    years.sort()
    return years

def scan_dataset(symbol: str, years: list[int] | None = None, months: list[tuple[int, int]] | None = None) -> list[JobType]:
    """
    Parses 'index.quantdev' files to find raw data chunks.
    Used by build_bars.py (to find raw) and backtest.py (to find days).
    """
    jobs = []
    
    # Determine targets
    targets = []
    if months:
        targets = sorted(months)
    elif years:
        for y in sorted(years):
            for m in range(1, 13):
                targets.append((y, m))
    
    IDX_HDR = config.INDEX_HDR_STRUCT
    IDX_ROW = config.INDEX_ROW_STRUCT
    IDX_ROW_SZ = config.INDEX_ROW_SIZE
    IDX_MAGIC = config.INDEX_MAGIC
    IDX_VER = config.INDEX_VERSION

    for (y, m) in targets:
        dir_p = os.path.join(config.BASE_DIR, symbol, f"{y:04d}", f"{m:02d}")
        idx_p = os.path.join(dir_p, "index.quantdev")
        dat_p = os.path.join(dir_p, "data.quantdev")

        if not os.path.exists(idx_p) or not os.path.exists(dat_p):
            continue

        try:
            with open(idx_p, "rb") as f:
                # 1. Read Header
                raw_hdr = f.read(config.INDEX_HDR_SIZE)
                if len(raw_hdr) < config.INDEX_HDR_SIZE: continue
                magic, ver, _ = IDX_HDR.unpack(raw_hdr)
                if magic != IDX_MAGIC or ver != IDX_VER: continue 

                # 2. Read Rows
                while True:
                    chunk = f.read(IDX_ROW_SZ)
                    if len(chunk) < IDX_ROW_SZ: break
                    day, off, ln, _ = IDX_ROW.unpack(chunk)
                    
                    jobs.append((dat_p, off, ln, f"{y:04d}-{m:02d}-{day:02d}"))

        except Exception:
            continue

    jobs.sort(key=lambda x: x[3])
    return jobs

# --- SIMULATION LOGIC ---

def worker_bars(shard_idx, days_to_process):
    """
    Reads .bars files (derived from raw_path) and runs Algo.
    """
    trades = []
    BAR_ITER = config.BAR_STRUCT.iter_unpack
    
    position = 0
    entry_px = 0.0
    entry_ts = 0.0
    
    # Metadata for metrics
    entry_eff = 0.0
    entry_imp = 0.0

    for (raw_path, _, _, date_str) in days_to_process:
        # Map raw data path to bars path
        bar_path = raw_path.replace("data.quantdev", "data.bars")
        
        if not os.path.exists(bar_path):
            continue
            
        try:
            with open(bar_path, "rb") as f:
                blob = f.read()
            
            # Hot Loop: ~5-10ns per bar overhead in 3.14t
            for bar in BAR_ITER(blob):
                # Unpack: ts_start, ts_end, o, h, l, c, vol, delta, eff, impact
                ts_end = bar[1]
                c = bar[5]
                
                # Exit Logic
                if position != 0:
                    roi_bps = ((c - entry_px) / entry_px) * 10000.0 * position
                    
                    # Simple Exit Logic (Placeholder for full Risk Manager)
                    if roi_bps > 40 or roi_bps < -15:
                        pnl = roi_bps - config.COST_BASIS_BPS
                        trades.append({
                            "entry_ts": entry_ts,
                            "exit_ts": ts_end,
                            "net_pnl_bps": pnl,
                            "side": position,
                            "k1": entry_eff, # Using Efficiency as K1 proxy
                            "k3": entry_imp  # Using Impact as K3 proxy
                        })
                        position = 0
                
                # Entry Logic
                sig = algo.decide(bar)
                if position == 0 and sig != 0:
                    position = sig
                    entry_px = c
                    entry_ts = ts_end
                    entry_eff = bar[8]
                    entry_imp = bar[9]

        except Exception: continue
        
    return trades

def run():
    print(f"--- Backtesting {config.SYMBOL} on Fractal Bars ---")
    years = discover_years(config.SYMBOL)
    if not years:
        print("[Error] No data found. Run build_bars.py first.")
        return

    all_jobs = scan_dataset(config.SYMBOL, years=years)
    print(f"[Scan] Found {len(all_jobs)} days.")
    
    w = config.WORKERS
    n = len(all_jobs)
    k, m = divmod(n, w)
    shards = [all_jobs[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(w)]
    
    t0 = time.perf_counter()
    trades = []
    
    with ThreadPoolExecutor(max_workers=w) as ex:
        futs = [ex.submit(worker_bars, i, s) for i, s in enumerate(shards)]
        for f in futs:
            res = f.result()
            if res: trades.extend(res)
            
    print(f"[Exec] {time.perf_counter()-t0:.2f}s | Trades: {len(trades)}")
    
    if trades:
        sc = metrics.generate_scorecard(trades)
        metrics.print_scorecard(sc)
    else:
        print("No trades generated.")

if __name__ == "__main__":
    run()