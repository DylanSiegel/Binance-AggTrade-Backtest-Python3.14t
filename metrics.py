"""
metrics.py
Robust Statistics Engine.
"""
import math
import statistics

def core_trade_metrics(trades):
    # Default safe return
    ret = {
        "total_trades": 0,
        "net_pnl_bps": 0.0,
        "avg_trade_bps": 0.0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "trade_sharpe": 0.0
    }
    
    if not trades:
        return ret
    
    pnls = [float(t["net_pnl_bps"]) for t in trades]
    wins = [p for p in pnls if p > 0.0]
    losses = [p for p in pnls if p <= 0.0]

    gross = sum(pnls)
    count = len(pnls)
    
    # Avoid div by zero
    mean = gross / count if count > 0 else 0.0
    var = statistics.pvariance(pnls) if count > 1 else 0.0
    std = math.sqrt(var)

    win_rate = (len(wins) / count) * 100.0 if count > 0 else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 0.0
    trade_sharpe = (mean / std) * math.sqrt(count) if std > 0.0 else 0.0

    return {
        "total_trades": float(count),
        "net_pnl_bps": float(gross),
        "avg_trade_bps": float(mean),
        "win_rate_pct": float(win_rate),
        "profit_factor": float(profit_factor),
        "trade_sharpe": float(trade_sharpe),
    }

def full_report(trades):
    core = core_trade_metrics(trades)
    
    by_kernel = {}
    for t in trades:
        k = t.get("kernel", "unknown")
        if k not in by_kernel: by_kernel[k] = []
        by_kernel[k].append(t)
    
    k_stats = {k: core_trade_metrics(v) for k, v in by_kernel.items()}
    
    return {"core": core, "by_kernel": k_stats, "risk": {}}