import sys
import time
import array
from concurrent.futures import ThreadPoolExecutor

if sys._is_gil_enabled():
    raise RuntimeError("Performance Critical: Must run on Python 3.14t (Free-Threaded)")

CPU_THREADS = 24
DATA_SIZE = 50_000_000  # 50 Million ticks

# Setup Data
prices = array.array('d', [100.0 + (i % 100) * 0.05 for i in range(DATA_SIZE)])
signals = array.array('b', [(i % 3) - 1 for i in range(DATA_SIZE)]) # -1, 0, 1 (Sell, Hold, Buy)

print(f"Stateful Benchmark: {DATA_SIZE:,} ticks on {CPU_THREADS} threads")

def backtest_worker(start, end, initial_cash):
    """
    Complex Stateful Logic:
    - Maintains 'cash' and 'position' variables.
    - Loops cannot be vectorized easily by Polars without 'scan'.
    - 3.14t handles this naturally in parallel.
    """
    local_prices = prices
    local_signals = signals
    
    cash = initial_cash
    position = 0.0
    
    # The Loop that kills Pandas/Polars performance
    for i in range(start, end):
        price = local_prices[i]
        sig = local_signals[i]
        
        if sig == 1 and cash > price: # Buy
            position += 1
            cash -= price
        elif sig == -1 and position > 0: # Sell
            position -= 1
            cash += price
            
    return cash + (position * local_prices[end-1])

print("-" * 60)
print("Running Stateful Iterative Backtest (The 'Loop' Killer)...")

start_time = time.perf_counter()

chunk_size = DATA_SIZE // CPU_THREADS
futures = []

with ThreadPoolExecutor(max_workers=CPU_THREADS) as executor:
    # Distribute chunks
    # Note: Real backtests need state passing between chunks, 
    # but for raw throughput testing, we treat them as independent strategy shards.
    for i in range(CPU_THREADS):
        s = i * chunk_size
        e = s + chunk_size if i < CPU_THREADS - 1 else DATA_SIZE
        futures.append(executor.submit(backtest_worker, s, e, 100000.0))

final_values = [f.result() for f in futures]

duration = time.perf_counter() - start_time
tps = DATA_SIZE / duration

print(f"Time: {duration:.4f}s")
print(f"TPS:  {tps:,.0f} Events/Sec")
print("-" * 60)
print("INTERPRETATION:")
print("This loop contains if/elif logic and state updates per row.")
print("To do this in Polars requires 'expr.map_elements' (slow) or Rust.")
print("Python 3.14t does it naturally at C-speed.")