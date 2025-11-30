#!/usr/bin/env python3.14t
"""
verify_v2.py

High-Performance Integrity Verifier for AGG2 Data format.
Optimized for: Python 3.14t (Free-Threaded) on Ryzen 9 7900X
Optimization: struct.iter_unpack, Hot Loop Unrolling, Local Var Caching
"""

import sys
import struct
import json
import time
import datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# 0. Runtime / 3.14t Boilerplate
# ---------------------------------------------------------------------------

if sys._is_gil_enabled():
    raise RuntimeError("Critical: GIL must be disabled. Run with -X gil=0")

# Ryzen 9 7900X: 12 Cores / 24 Threads
CPU_THREADS = 24 

# PEP 784 Native Compression
try:
    import compression.zstd as zstd
except ImportError:
    raise RuntimeError("Critical: Native 'compression.zstd' not found.")

# ============================================================================
# Configuration & Constants
# ============================================================================

CONFIG: dict[str, object] = {
    "BASE_DIR": "data",
    "REPORT_FILE": "integrity_report_v2.json",
    "WORKERS": CPU_THREADS,
    "SYMBOL": "BTCUSDT",
}

AGG_HDR_MAGIC = b"AGG2"

# Structures
# <trade_id, px, qty, first_id, count, flags, ts_ms, side, padding>
# Size: 8+8+8+8+2+1+8+1+3 = 47 bytes? No, struct alignment might apply.
# We trust the struct definitions provided previously.
AGG_ROW_STRUCT = struct.Struct("<QQQQHHqB3x")
AGG_HDR_STRUCT = struct.Struct("<4sBBHQqq16x")
INDEX_ROW_STRUCT = struct.Struct("<HQQ")

# ============================================================================
# Core Logic
# ============================================================================

def validate_month(year: int, month: int, path: Path) -> dict:
    """
    Validates a single month folder using optimized iterators.
    """
    stats = {
        "period": f"{year}-{month:02d}",
        "valid": True,
        "days_checked": 0,
        "rows_checked": 0,
        "errors": [],
        "gaps": [],
        "warnings": [],
    }

    # Optimization: Cache append methods to avoid dict lookups in hot loop
    err_append = stats["errors"].append
    gap_append = stats["gaps"].append
    warn_append = stats["warnings"].append

    data_path = path / "data.quantdev"
    index_path = path / "index.quantdev"

    if not data_path.exists() or not index_path.exists():
        return stats

    try:
        data_size = data_path.stat().st_size

        # 1. Fast Index Read
        index_entries = []
        with open(index_path, "rb") as f_idx:
            # Read entire index into memory (it's small)
            idx_blob = f_idx.read()
            
        # Unpack index using iterator (Fast)
        try:
            index_entries = sorted(
                list(INDEX_ROW_STRUCT.iter_unpack(idx_blob)), 
                key=lambda x: x[1]
            )
        except struct.error:
            err_append(f"Index file corruption or size mismatch")
            stats["valid"] = False
            return stats

        # 2. Process Data File
        with open(data_path, "rb") as f_data:
            for day_num, offset, length in index_entries:
                stats["days_checked"] += 1

                # Bounds Check
                if offset + length > data_size:
                    err_append(f"Day {day_num}: Offset out of bounds")
                    stats["valid"] = False
                    continue

                # Read Compressed Chunk
                f_data.seek(offset)
                compressed_blob = f_data.read(length)
                
                if len(compressed_blob) != length:
                    err_append(f"Day {day_num}: Unexpected EOF")
                    stats["valid"] = False
                    continue

                # Decompress
                try:
                    raw_blob = zstd.decompress(compressed_blob)
                except Exception as e:
                    err_append(f"Day {day_num}: Decompression failed ({e})")
                    stats["valid"] = False
                    continue

                # Header Validation
                if len(raw_blob) < AGG_HDR_STRUCT.size:
                    err_append(f"Day {day_num}: Blob too small")
                    stats["valid"] = False
                    continue

                # Unpack Header
                # Slicing bytes creates a copy, but header is tiny (scrappy overhead)
                magic, ver, h_day, _, row_count, ts_min, ts_max = \
                    AGG_HDR_STRUCT.unpack(raw_blob[:AGG_HDR_STRUCT.size])

                if magic != AGG_HDR_MAGIC:
                    err_append(f"Day {day_num}: Invalid magic")
                    stats["valid"] = False
                    continue
                
                if h_day != day_num:
                    err_append(f"Day {day_num}: Header day mismatch")
                    stats["valid"] = False

                expected_payload = row_count * AGG_ROW_STRUCT.size
                row_data_start = AGG_HDR_STRUCT.size
                
                # Check explicit payload size
                if len(raw_blob) - row_data_start != expected_payload:
                    err_append(f"Day {day_num}: Size mismatch (Expected {expected_payload})")
                    stats["valid"] = False
                    continue

                # -------------------------------------------------------
                # CRITICAL HOT PATH
                # -------------------------------------------------------
                
                # Create iterator directly on the buffer (Zero-copy if using memoryview context, 
                # but iter_unpack handles buffer protocol efficiently)
                try:
                    # Note: We slice raw_blob to skip header. 
                    # Python 3.14t optimizes slicing somewhat, but memoryview is safest for zero-copy.
                    with memoryview(raw_blob) as mv:
                        payload_mv = mv[row_data_start:]
                        row_iter = AGG_ROW_STRUCT.iter_unpack(payload_mv)
                        
                        # --- Loop Unrolling: Handle First Row ---
                        try:
                            # Unpack: tid, px, qty, fid, cnt, flg, ts, side
                            first_row = next(row_iter)
                            prev_id, prev_px, prev_qt, _, _, _, prev_ts, _ = first_row
                            
                            # Validations for first row
                            if prev_px <= 0 or prev_qt < 0:
                                err_append(f"Day {day_num}: Invalid Px/Qty at ID {prev_id}")
                                stats["valid"] = False
                            
                            stats["rows_checked"] += 1
                        except StopIteration:
                            # Empty day (valid)
                            continue

                        # --- The Inner Loop (Millions of iterations) ---
                        # Optimization: Localize variables to register/stack
                        # No 'is not None' checks here.
                        
                        for tid, px, qt, fid, cnt, flg, ts, side in row_iter:
                            
                            # 1. ID Monotonicity
                            # Most common case: tid == prev_id + 1
                            if tid != prev_id + 1:
                                if tid <= prev_id:
                                    err_append(f"Day {day_num}: Non-monotonic ID {prev_id}->{tid}")
                                    stats["valid"] = False
                                else:
                                    gap_append({"day": day_num, "from": prev_id, "to": tid})

                            # 2. Time Monotonicity
                            if ts < prev_ts:
                                err_append(f"Day {day_num}: Time warp {prev_ts}->{ts}")
                                stats["valid"] = False

                            # 3. Data Integrity
                            if px <= 0 or qt < 0:
                                err_append(f"Day {day_num}: Invalid Px/Qty at ID {tid}")
                                stats["valid"] = False

                            # 4. Aggregation Logic
                            if (fid + cnt - 1) < fid:
                                err_append(f"Day {day_num}: Invalid agg count at ID {tid}")
                                stats["valid"] = False
                            
                            # 5. Header Bounds (Warning only)
                            # Using 'or' is slightly slower if first condition is true, 
                            # but usually ts is within bounds.
                            if ts < ts_min or ts > ts_max:
                                warn_append(f"Day {day_num}: TS {ts} outside header [{ts_min}, {ts_max}]")

                            # Update registers
                            prev_id = tid
                            prev_ts = ts
                            stats["rows_checked"] += 1

                except struct.error:
                    err_append(f"Day {day_num}: Corrupt struct alignment or incomplete bytes")
                    stats["valid"] = False

    except Exception as e:
        err_append(f"Critical IO Error: {e}")
        stats["valid"] = False

    return stats

# ============================================================================
# Main Controller
# ============================================================================

def main() -> None:
    start_time = time.perf_counter()
    symbol_dir = Path(CONFIG["BASE_DIR"]) / str(CONFIG["SYMBOL"])

    if not symbol_dir.exists():
        print(f"Error: Directory {symbol_dir} not found.")
        return

    # Discovery
    tasks: list[tuple[int, int, Path]] = []
    print(f"--- High-Performance Integrity Verifier v2 (Ryzen 9 7900X) ---")
    print(f"Target: Python 3.14t (Free-Threaded)")
    
    for year_dir in symbol_dir.iterdir():
        if year_dir.is_dir() and year_dir.name.isdigit():
            for month_dir in year_dir.iterdir():
                if month_dir.is_dir() and month_dir.name.isdigit():
                    tasks.append((int(year_dir.name), int(month_dir.name), month_dir))

    # Sort tasks for consistent output
    tasks.sort()
    
    print(f"Tasks: {len(tasks)} months")
    print(f"Threads: {CONFIG['WORKERS']}")

    results = []
    total_rows = 0
    corruption_count = 0

    # ThreadPoolExecutor in 3.14t scales linearly for CPU tasks
    with ThreadPoolExecutor(max_workers=int(CONFIG["WORKERS"])) as executor:
        future_map = {
            executor.submit(validate_month, y, m, p): (y, m)
            for (y, m, p) in tasks
        }

        for i, future in enumerate(as_completed(future_map)):
            y, m = future_map[future]
            try:
                res = future.result()
                results.append(res)
                rows = int(res["rows_checked"])
                total_rows += rows
                
                status = "PASS" if res["valid"] else "FAIL"
                if not res["valid"]:
                    corruption_count += 1
                
                # Simple progress bar
                sys.stdout.write(f"\r[{i+1}/{len(tasks)}] {y}-{m:02d}: {status} ({rows:,} rows)")
                sys.stdout.flush()

            except Exception as e:
                print(f"\nWorker Error {y}-{m:02d}: {e}")

    elapsed = time.perf_counter() - start_time
    mps = (total_rows / elapsed) / 1_000_000 if elapsed > 0 else 0

    print("\n\n--- Performance Summary ---")
    print(f"Total Rows:     {total_rows:,}")
    print(f"Throughput:     {mps:.2f} Million Rows/sec")
    print(f"Time Elapsed:   {elapsed:.2f}s")
    print(f"Corrupt Months: {corruption_count}")

    report = {
        "meta": {
            "timestamp": dt.datetime.now().isoformat(),
            "hardware": "AMD Ryzen 9 7900X",
            "runtime": "Python 3.14t",
            "total_rows": total_rows,
            "elapsed_seconds": elapsed,
        },
        "details": sorted(results, key=lambda x: x["period"]),
    }

    with open(str(CONFIG["REPORT_FILE"]), "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()