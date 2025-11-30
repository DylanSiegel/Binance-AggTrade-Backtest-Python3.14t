"""
metrics.py
The Master Scorecard Engine.
Calculates "Max Data" trade statistics using Python Standard Library.
"""
import math
import statistics
import collections
from itertools import groupby

def _safe_mean(data):
    return statistics.fmean(data) if data else 0.0

def _safe_std(data, mu=None):
    if len(data) < 2: return 0.0
    return statistics.stdev(data, xbar=mu)

def _percentile(data, p):
    if not data: return 0.0
    data.sort()
    k = (len(data) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c: return data[int(k)]
    d0 = data[int(f)]
    d1 = data[int(c)]
    return d0 + (d1 - d0) * (k - f)

def _downside_deviation(data, target=0.0):
    if not data: return 0.0
    downside_sq = sum((min(x - target, 0.0)) ** 2 for x in data)
    return math.sqrt(downside_sq / len(data))

def _calc_drawdowns(pnls):
    """
    Returns (MaxDD Bps, MaxDD Duration in Trades, Equity Curve)
    """
    if not pnls: return 0.0, 0, []
    
    equity = [0.0]
    curr = 0.0
    for p in pnls:
        curr += p
        equity.append(curr)
        
    peak = -1e9
    max_dd = 0.0
    max_dur = 0
    curr_dur = 0
    
    for val in equity:
        if val > peak:
            peak = val
            curr_dur = 0
        else:
            dd = peak - val
            curr_dur += 1
            if dd > max_dd: max_dd = dd
            if curr_dur > max_dur: max_dur = curr_dur
            
    return max_dd, max_dur, equity

def _calc_streaks(pnls):
    """
    Returns (Max Win Streak, Max Loss Streak)
    """
    if not pnls: return 0, 0
    
    # 1 = Win, -1 = Loss
    signs = [1 if p > 0 else -1 for p in pnls if p != 0]
    if not signs: return 0, 0

    max_win = 0
    max_loss = 0
    
    for k, g in groupby(signs):
        length = len(list(g))
        if k == 1:
            max_win = max(max_win, length)
        else:
            max_loss = max(max_loss, length)
            
    return max_win, max_loss

def generate_scorecard(trades: list[dict]):
    """
    Ingests a list of trade dictionaries.
    Returns the Master Scorecard (Dictionary).
    """
    # Initialize Master Structure (Subset of the prompt's massive list)
    sc = {
        "strategy_returns": {},
        "trade_level": {},
        "regime_conditional": {},
        "robustness_overfitting": {},
        "advanced_structure": {}
    }
    
    if not trades:
        return sc

    # --- Pre-Process Data ---
    pnls = [t['net_pnl_bps'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    # Time Analysis
    # Assumes t['entry_ts'] and t['exit_ts'] are timestamps in ms
    holds_sec = [(t['exit_ts'] - t['entry_ts']) / 1000.0 for t in trades]
    
    # --- 1. Strategy Returns ---
    n = len(pnls)
    mean_bps = _safe_mean(pnls)
    std_bps = _safe_std(pnls, mean_bps)
    gross_pnl = sum(pnls)
    
    max_dd_bps, max_dd_dur, equity_curve = _calc_drawdowns(pnls)
    
    # Annualized approximations (Assuming crypto 24/7, ~Trades/Day estimate needed)
    # We estimate trades per day from data span
    t_start = trades[0]['entry_ts']
    t_end = trades[-1]['exit_ts']
    days = max((t_end - t_start) / 86_400_000.0, 1.0)
    trades_per_day = n / days
    
    # Annualized Vol (Bps) -> StdDev * Sqrt(Trades/Year)
    ann_vol_bps = std_bps * math.sqrt(trades_per_day * 365)
    # Annualized Return (Bps) -> Mean * Trades/Year
    ann_ret_bps = mean_bps * trades_per_day * 365
    
    # Ratios
    sharpe = (ann_ret_bps / ann_vol_bps) if ann_vol_bps > 0 else 0.0
    
    # Sortino (Downside Dev)
    down_dev = _downside_deviation(pnls)
    ann_down_dev = down_dev * math.sqrt(trades_per_day * 365)
    sortino = (ann_ret_bps / ann_down_dev) if ann_down_dev > 0 else 0.0
    
    calmar = (ann_ret_bps / max_dd_bps) if max_dd_bps > 0 else 0.0
    
    sc["strategy_returns"] = {
        "total_trades": n,
        "net_pnl_bps": gross_pnl,
        "gross_mean_return_bps_per_trade": mean_bps,
        "return_vol_bps_per_trade": std_bps,
        "sharpe_annualized": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown_bps": max_dd_bps,
        "max_drawdown_duration_trades": max_dd_dur,
        "recovery_factor": (gross_pnl / max_dd_bps) if max_dd_bps > 0 else 0.0,
        "pnl_skewness": (statistics.mean([(x - mean_bps)**3 for x in pnls]) / (std_bps**3)) if std_bps > 0 else 0.0,
        "pnl_percentile_1": _percentile(pnls, 0.01),
        "pnl_percentile_5": _percentile(pnls, 0.05),
        "pnl_percentile_95": _percentile(pnls, 0.95),
        "pnl_percentile_99": _percentile(pnls, 0.99),
    }

    # --- 2. Trade Level ---
    n_wins = len(wins)
    n_loss = len(losses)
    win_rate = (n_wins / n) * 100.0
    avg_win = _safe_mean(wins)
    avg_loss = _safe_mean(losses)
    
    profit_factor = (sum(wins) / abs(sum(losses))) if sum(losses) != 0 else 0.0
    sqn = (math.sqrt(n) * (mean_bps / std_bps)) if std_bps > 0 else 0.0
    
    streak_win, streak_loss = _calc_streaks(pnls)
    
    sc["trade_level"] = {
        "win_rate_pct": win_rate,
        "avg_win_bps": avg_win,
        "avg_loss_bps": avg_loss,
        "profit_factor": profit_factor,
        "risk_reward_ratio": (avg_win / abs(avg_loss)) if avg_loss != 0 else 0.0,
        "sqn": sqn,
        "trades_per_day": trades_per_day,
        "avg_holding_sec": _safe_mean(holds_sec),
        "max_consecutive_wins": streak_win,
        "max_consecutive_losses": streak_loss,
    }

    # --- 3. Regime Conditional (Kernel Analysis) ---
    # We group by 'k3' (Volatility/Toxicity) and 'k1' (Momentum) if available
    
    # Discretize K3 (Vol) into Low/Mid/High
    # K3 roughly: < -1 (Low Vol), -1 to 1 (Mid), > 1 (High Vol/Tox)
    k3_buckets = {"low_vol": [], "mid_vol": [], "high_vol": []}
    
    for t in trades:
        k3 = t.get('k3', 0.0)
        p = t['net_pnl_bps']
        if k3 < -0.5: k3_buckets["low_vol"].append(p)
        elif k3 > 0.5: k3_buckets["high_vol"].append(p)
        else: k3_buckets["mid_vol"].append(p)
        
    sc["regime_conditional"] = {
        "avg_pnl_low_vol": _safe_mean(k3_buckets["low_vol"]),
        "avg_pnl_mid_vol": _safe_mean(k3_buckets["mid_vol"]),
        "avg_pnl_high_vol": _safe_mean(k3_buckets["high_vol"]),
        "count_low_vol": len(k3_buckets["low_vol"]),
        "count_high_vol": len(k3_buckets["high_vol"]),
    }

    # --- 4. Robustness (Split Half Test) ---
    mid = n // 2
    first_half = pnls[:mid]
    second_half = pnls[mid:]
    
    m1 = _safe_mean(first_half)
    s1 = _safe_std(first_half)
    sh1 = m1 / s1 if s1 > 0 else 0
    
    m2 = _safe_mean(second_half)
    s2 = _safe_std(second_half)
    sh2 = m2 / s2 if s2 > 0 else 0
    
    sc["robustness_overfitting"] = {
        "sharpe_is_first_half": sh1 * math.sqrt(trades_per_day*365),
        "sharpe_oos_second_half": sh2 * math.sqrt(trades_per_day*365),
        "consistency_ratio": min(sh1, sh2) / max(sh1, sh2) if max(sh1, sh2) != 0 else 0.0
    }

    return sc