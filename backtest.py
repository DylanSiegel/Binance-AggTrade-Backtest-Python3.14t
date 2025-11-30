"""
backtest.py (Contiguous Sharding, Data-Safe, Auto-Discover Years)
Vectorized-style backtest on AGG2 blobs.

CRITICAL FIX:
- Sharding is now CONTIGUOUS (each worker sees a continuous time block).
- Prevents artificial time gaps that destroyed performance under round-robin.
- Years are auto-discovered from the data directory.
"""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import math

# Mandatory 3.14t boilerplate
if sys._is_gil_enabled():
    raise RuntimeError("GIL must be disabled. Run with: python -X gil=0 backtest.py")

CPU_THREADS = 24

import os
import time
from typing import List, Tuple

import config
import algo
import metrics

try:
    import compression.zstd as zstd
except ImportError:
    print("[FATAL] compression.zstd not found.")
    sys.exit(1)

# (data_path, offset, length, yyyy_mm_dd)
JobType = Tuple[str, int, int, str]


def discover_years(symbol: str) -> List[int]:
    """
    Auto-discover all years under BASE_DIR/symbol that actually contain
    at least one month with (index.quantdev, data.quantdev).
    """
    base = os.path.join(config.BASE_DIR, symbol)
    if not os.path.exists(base):
        return []

    years: List[int] = []
    for name in os.listdir(base):
        if not name.isdigit():
            continue
        try:
            y = int(name)
        except ValueError:
            continue

        year_path = os.path.join(base, name)
        if not os.path.isdir(year_path):
            continue

        has_data = False
        for m_name in os.listdir(year_path):
            if not m_name.isdigit():
                continue
            month_path = os.path.join(year_path, m_name)
            if not os.path.isdir(month_path):
                continue

            idx_path = os.path.join(month_path, "index.quantdev")
            dat_path = os.path.join(month_path, "data.quantdev")
            if os.path.exists(idx_path) and os.path.exists(dat_path):
                has_data = True
                break

        if has_data:
            years.append(y)

    years.sort()
    return years


def scan_dataset(symbol: str, years: List[int]) -> List[JobType]:
    """
    Build a list of all (data_path, offset, length, date_str) jobs.
    Sorted strictly chronologically: Year -> Month -> Day.
    """
    all_jobs: List[JobType] = []
    print(f"[Scan] Indexing years={years} for {symbol}...")

    for year in years:
        for month in range(1, 13):
            base = os.path.join(config.BASE_DIR, symbol, f"{year:04d}", f"{month:02d}")
            idx_p = os.path.join(base, "index.quantdev")
            dat_p = os.path.join(base, "data.quantdev")

            if not os.path.exists(idx_p) or not os.path.exists(dat_p):
                continue

            try:
                with open(idx_p, "rb") as f:
                    idx_bytes = f.read()

                month_jobs = []
                ptr = 0
                limit = len(idx_bytes)
                while ptr < limit:
                    # Index row: <HQQ> = Day, Offset, Length
                    day, off, ln = config.INDEX_ROW_STRUCT.unpack_from(idx_bytes, ptr)
                    ptr += config.INDEX_ROW_SIZE
                    month_jobs.append((day, off, ln))

                # Intra-month order
                month_jobs.sort(key=lambda x: x[0])

                for day, off, ln in month_jobs:
                    date_str = f"{year:04d}-{month:02d}-{day:02d}"
                    all_jobs.append((dat_p, off, ln, date_str))
            except Exception as e:
                print(f"[WARN] Failed to read index {idx_p}: {e}")
                continue

    # Global chronological order
    all_jobs.sort(key=lambda j: j[3])
    print(f"[Scan] Found {len(all_jobs)} chunks.")
    return all_jobs


def worker_process_shard(shard_id: int, jobs: List[JobType], symbol: str):
    """
    Process a CONTIGUOUS list of days with one AlphaEngine.
    State flows naturally from Day N to Day N+1 within this shard.
    """
    engine = algo.AlphaEngine(symbol)
    trades = []

    position = 0
    entry_px = 0.0
    entry_ts = 0

    last_px = 0.0
    last_ts = 0

    ROW = config.AGG_ROW_STRUCT
    HDR = config.AGG_HDR_SIZE

    for (path, off, ln, _) in jobs:
        try:
            with open(path, "rb") as f:
                f.seek(off)
                c_blob = f.read(ln)

            raw = zstd.decompress(c_blob)
        except Exception as e:
            print(f"[WARN] Shard {shard_id}: blob read failed at {path}@{off}: {e}")
            continue

        # Vectorized update
        signals = engine.update_batch(ROW.iter_unpack(raw[HDR:]))

        for ts, px, sig in signals:
            last_px = px
            last_ts = ts

            # EXIT LOGIC
            if position != 0:
                roi = (px - entry_px) / entry_px if position > 0 else (entry_px - px) / entry_px

                # Take Profit (+40bps) / Stop Loss (-15bps)
                if roi > 0.0040 or roi < -0.0015:
                    pnl = (roi * 10000.0) - config.COST_BASIS_BPS
                    trades.append(
                        {
                            "entry_ts": entry_ts,
                            "exit_ts": ts,
                            "net_pnl_bps": pnl,
                            "side": position,
                            "kernel": "turbo",
                        }
                    )
                    position = 0

            # ENTRY LOGIC (Only if flat)
            if position == 0 and sig != 0:
                position = sig
                entry_px = px
                entry_ts = ts

    # Force close at end of shard to capture final PnL
    if position != 0 and last_px > 0.0:
        roi = (last_px - entry_px) / entry_px if position > 0 else (entry_px - last_px) / entry_px
        pnl = (roi * 10000.0) - config.COST_BASIS_BPS
        trades.append(
            {
                "entry_ts": entry_ts,
                "exit_ts": last_ts,
                "net_pnl_bps": pnl,
                "side": position,
                "kernel": "turbo",
            }
        )

    return trades


def run_backtest(years: List[int] | None = None):
    symbol = config.SYMBOL

    # Auto-discover full history if not provided
    if years is None:
        years = discover_years(symbol)
        if not years:
            print(f"[Backtest] No year folders found under {config.BASE_DIR}/{symbol}.")
            return

    t0 = time.perf_counter()

    jobs = scan_dataset(symbol, years)
    if not jobs:
        print("[Backtest] No data jobs found.")
        return

    workers = config.WORKERS
    if workers <= 0:
        workers = 1

    # --- CONTIGUOUS SHARDING ---
    # Split jobs into continuous blocks instead of round-robin.
    chunk_size = math.ceil(len(jobs) / workers)
    shards: List[List[JobType]] = []
    for i in range(0, len(jobs), chunk_size):
        shards.append(jobs[i : i + chunk_size])

    actual_workers = len(shards)
    print(f"[Backtest] Processing {len(jobs)} chunks on {actual_workers} threads (Contiguous Block Mode)...")

    trades = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(worker_process_shard, i, shards[i], symbol)
            for i in range(actual_workers)
        ]
        for fut in futures:
            res = fut.result()
            if res:
                trades.extend(res)

    dt = time.perf_counter() - t0
    print(f"[Backtest] Done in {dt:.2f}s")

    if not trades:
        print("[Backtest] No trades generated.")
        return

    report = metrics.full_report(trades)
    core = report["core"]

    print("\n--- CORE METRICS ---")
    print(f"Years:       {years[0]} -> {years[-1]}")
    print(f"Trades:      {core['total_trades']:.0f}")
    print(f"Net PnL:     {core['net_pnl_bps']:.2f} bps")
    print(f"Avg Trade:   {core['avg_trade_bps']:.2f} bps")
    print(f"Win Rate:    {core['win_rate_pct']:.2f}%")
    print(f"ProfitFact:  {core['profit_factor']:.2f}")
    print(f"TradeSharpe: {core['trade_sharpe']:.2f}")

    print("\n--- BY KERNEL ---")
    for k, v in report["by_kernel"].items():
        print(
            f"{k:<12} -> trades={v['total_trades']:.0f}, "
            f"net={v['net_pnl_bps']:.2f} bps, sharpe={v['trade_sharpe']:.2f}"
        )


if __name__ == "__main__":
    # Full auto-discovered history by default
    run_backtest()
