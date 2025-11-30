"""
quick.py
"""
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor

if sys._is_gil_enabled():
    raise RuntimeError("Run with -X gil=0")

import config
import backtest
import metrics

def get_recent_months(symbol, n=6):
    base = os.path.join(config.BASE_DIR, symbol)
    if not os.path.exists(base): return []
    
    years = sorted([int(x) for x in os.listdir(base) if x.isdigit()], reverse=True)
    results = []
    for y in years:
        y_path = os.path.join(base, str(y))
        months = sorted([int(x) for x in os.listdir(y_path) if x.isdigit()], reverse=True)
        for m in months:
            # Check for data
            if os.path.exists(os.path.join(y_path, f"{m:02d}", "index.quantdev")):
                results.append((y, m))
                if len(results) >= n: return results
    return results

def run():
    print(f"--- Quick Check: {config.SYMBOL} ---")
    months = get_recent_months(config.SYMBOL)
    if not months:
        print("[Error] No data.")
        return
        
    months.sort()
    print(f"[Scan] Range: {months[0]} -> {months[-1]}")
    
    # Use backtest scanner
    jobs = backtest.scan_dataset(config.SYMBOL, months=months)
    
    # Shard
    w = config.WORKERS
    n = len(jobs)
    k, m = divmod(n, w)
    shards = [jobs[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(w)]
    
    t0 = time.perf_counter()
    trades = []
    
    with ThreadPoolExecutor(max_workers=w) as ex:
        # Use CORRECT worker function
        futs = [ex.submit(backtest.worker_bars, i, s) for i, s in enumerate(shards)]
        for f in futs:
            res = f.result()
            if res: trades.extend(res)
            
    print(f"[Done] {time.perf_counter()-t0:.2f}s | Trades: {len(trades)}")
    
    if trades:
        sc = metrics.generate_scorecard(trades)
        metrics.print_scorecard(sc)
    else:
        print("No trades generated.")

if __name__ == "__main__":
    run()