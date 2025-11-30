"""
quick.py
Rapid Verification - Auto-Latest (Data-Safe, Ordered).

Automatically finds the newest 6 months of data and runs a 24-core simulation,
reusing the same sharding + worker logic as backtest.py.
"""

import sys, threading
from concurrent.futures import ThreadPoolExecutor

# Mandatory 3.14t boilerplate
if sys._is_gil_enabled():
    print("!!! WARNING: GIL is ENABLED. Run with: python -X gil=0 quick.py")

CPU_THREADS = 24

import time
import os
from typing import List, Tuple

import config
import backtest
import metrics


def get_latest_months(symbol: str, n: int = 6) -> List[Tuple[int, int]]:
    """
    Scans the data directory to find the most recent N (year, month) folders
    that actually have an index.quantdev present.
    """
    base = os.path.join(config.BASE_DIR, symbol)
    if not os.path.exists(base):
        return []

    try:
        years = [d for d in os.listdir(base) if d.isdigit()]
        years.sort(key=int, reverse=True)  # 2025, 2024, ...
    except Exception:
        return []

    results: List[Tuple[int, int]] = []
    for y_str in years:
        y_int = int(y_str)
        y_path = os.path.join(base, y_str)
        try:
            months = [d for d in os.listdir(y_path) if d.isdigit()]
            months.sort(key=int, reverse=True)  # 12, 11, ...
        except Exception:
            continue

        for m_str in months:
            m_int = int(m_str)
            base_path = os.path.join(y_path, m_str)
            idx_path = os.path.join(base_path, "index.quantdev")
            dat_path = os.path.join(base_path, "data.quantdev")
            if os.path.exists(idx_path) and os.path.exists(dat_path):
                results.append((y_int, m_int))
                if len(results) >= n:
                    return results

    return results


def run_latest_6_months():
    symbol = config.SYMBOL

    # 1. DISCOVERY
    target_months = get_latest_months(symbol, n=6)
    if not target_months:
        print(f"[Fail] No data folders found for {symbol}")
        return

    # Sorted chronological for reporting
    target_months.sort()
    start_str = f"{target_months[0][0]:04d}-{target_months[0][1]:02d}"
    end_str = f"{target_months[-1][0]:04d}-{target_months[-1][1]:02d}"

    print(f"--- QUICK CHECK: {symbol} [Last 6 Months] ---")
    print(f"Range: {start_str} to {end_str}")
    print(f"Cores: {config.WORKERS} (Ryzen 7900X)")

    # 2. JOB SCANNING (DATA-SAFE: month-internal sort by Day)
    print(f"[1/4] Scanning indices for last 6 months...")

    jobs: List[Tuple[str, int, int, str]] = []
    for (year, month) in target_months:
        base_path = os.path.join(config.BASE_DIR, symbol, f"{year:04d}", f"{month:02d}")
        index_path = os.path.join(base_path, "index.quantdev")
        data_path = os.path.join(base_path, "data.quantdev")

        if not os.path.exists(index_path) or not os.path.exists(data_path):
            continue

        try:
            with open(index_path, "rb") as f:
                idx_bytes = f.read()

            month_jobs = []
            ptr = 0
            limit = len(idx_bytes)
            while ptr < limit:
                day, off, ln = config.INDEX_ROW_STRUCT.unpack_from(idx_bytes, ptr)
                ptr += config.INDEX_ROW_SIZE
                month_jobs.append((day, off, ln))

            # CRITICAL: sort by Day (not by write order)
            month_jobs.sort(key=lambda x: x[0])

            for day, off, ln in month_jobs:
                jobs.append(
                    (
                        data_path,
                        off,
                        ln,
                        f"{year:04d}-{month:02d}-{day:02d}",
                    )
                )
        except Exception as e:
            print(f"[WARN] Failed to read index {index_path}: {e}")
            continue

    if not jobs:
        print("[Fail] Indices found but no jobs. Run data.py.")
        return

    # Enforce chronological order across all months by date string
    jobs.sort(key=lambda j: j[3])

    # 3. SHARDING
    workers = config.WORKERS
    if workers != CPU_THREADS:
        print(f"[WARN] config.WORKERS={workers} but CPU_THREADS={CPU_THREADS}")

    shards: List[List[Tuple[str, int, int, str]]] = [[] for _ in range(workers)]
    for i, job in enumerate(jobs):
        shards[i % workers].append(job)

    print(f"[2/4] Sharded {len(jobs)} chunks across {workers} workers.")

    # 4. PARALLEL EXECUTION (reuse backtest.worker_process_shard)
    print(f"[3/4] Executing Turbo Engine (vectorized)...")
    t0 = time.perf_counter()
    all_trades = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(backtest.worker_process_shard, i, shards[i], symbol)
            for i in range(workers)
            if shards[i]
        ]
        for fut in futures:
            res = fut.result()
            if res:
                all_trades.extend(res)

    elapsed = time.perf_counter() - t0

    # 5. METRICS
    print(f"\n[4/4] Done in {elapsed:.4f}s")
    if elapsed > 0:
        # Rough row estimate: if you want exact, you would need to read AGG_HDR count per chunk.
        rows_est = len(jobs) * 50_000.0
        print(f"Throughput (rough): {rows_est / elapsed:,.0f} rows/sec")

    print("\n--- Strategy Performance (Recent 6 Months) ---")
    print(f"Period: {start_str} -> {end_str}")

    if all_trades:
        rep = metrics.full_report(all_trades)
        core = rep["core"]
        print(f"Trades:   {int(core['total_trades'])}")
        print(f"Net PnL:  {core['net_pnl_bps']:.2f} bps")
        print(f"Sharpe:   {core['trade_sharpe']:.2f}")
        print(f"Win Rate: {core['win_rate_pct']:.2f}%")

        print("\n[Kernel Contribution]")
        for k, v in rep["by_kernel"].items():
            print(f"  {k:<12}: {v['net_pnl_bps']:8.2f} bps")
    else:
        print("[WARN] No trades generated. (Check thresholds in algo.py.)")


if __name__ == "__main__":
    run_latest_6_months()
