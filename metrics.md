## A. Conceptual Master List (Improved & Expanded)

### 0. Data Quality & Microstructure Sanity

These are preconditions before trusting any other metric.

1. Sample coverage

   * `n_bars`, `n_days`, `n_trades`, `n_symbols`
   * % days with data gaps.

2. Missing / invalid data

   * `%_missing_signal`
   * `%_missing_prices`
   * `%_missing_volume`
   * `%_nonmonotonic_timestamps`
   * `%_bad_quotes` (crossed/locked NBBO, negative spread)

3. Outliers & clipping

   * `%_signal_outliers` (|z| > threshold)
   * `%_return_outliers`
   * `%_volume_spikes` beyond quantile threshold.

4. Microstructure sanity checks

   * `%_negative_spreads`
   * `%_crossed_markets`
   * `avg_spread_bps`, `median_spread_bps`
   * `avg_depth_shares` at best bid/ask
   * `realized_vol_1m` / `realized_vol_5m` (environment indicator)

---

### 1. Signal Distribution & Stationarity (“Hygiene”)

1. Basic distribution

   * `signal_mean`
   * `signal_std`
   * `signal_median`
   * `signal_pct_1`, `signal_pct_5`, `signal_pct_95`, `signal_pct_99`
   * `signal_iqr` (interquartile range)

2. Normality / shape

   * `signal_skew`
   * `signal_kurtosis`
   * `signal_jarque_bera_pvalue`

3. Stationarity / memory

   * `acf_lag1_signal`
   * `acf_lags_signal` (vector)
   * `pacf_lag1_signal`
   * `hurst_exponent_signal`
   * `adf_pvalue_signal`
   * `kpss_pvalue_signal`
   * `variance_ratio_signal`

4. Entropy / complexity

   * `shannon_entropy_signal`
   * `approximate_entropy_signal`
   * `sample_entropy_signal`
   * `%_zero_nearzero_signal`

5. Fractal / chaotic

   * `fractal_dimension_signal`
   * `higuchi_fractal_dimension_signal`
   * `lyapunov_exponent_signal`
   * `lempel_ziv_complexity_signal`

---

### 2. Predictive Power vs Returns (Cross-Sectional & Time-Series)

Use whichever is appropriate to your setup (cross-sectional IC or time-series regression).

1. Linear & rank correlation

   * `ic_pearson` (corr(S, R))
   * `ic_spearman` (rank IC)
   * `ic_kendall_tau`
   * `ic_decimated` (non-overlapping windows)
   * `ic_time_weighted` (decaying weights)
   * `ic_term_structure` (IC vs horizon)
   * `ic_volatility` (std of IC over time)
   * `ic_information_ratio` (mean(IC) / std(IC))

2. Binned / monotonicity

   * `decile_spread_top_bottom_bps`
   * `quintile_spread_q5_q1_bps`
   * `quantile_monotonicity_score` (Spearman between bin index and avg return)
   * `hit_rate_top_quantile` (% of top bucket with positive R)
   * `hit_rate_top_vs_bottom_diff`

3. Regression-based

   * `ols_beta_signal_to_returns`
   * `ols_alpha_bps`
   * `ols_tstat_signal`
   * `hac_tstat_newey_west`
   * `pedersen_qz_tstat`
   * `predictive_r2_is`
   * `predictive_r2_oos`
   * `fama_macbeth_tstat`
   * `fama_macbeth_r2`

4. Classification-style (if directional/threshold decisions)

   * `auc_roc`
   * `auc_pr` (precision-recall)
   * `brier_score`
   * `confusion_matrix` for sign prediction
   * `f1_score` (if you classify + vs − returns).

5. Nonlinear dependence

   * `mutual_information_signal_return`
   * `maximal_information_coefficient`
   * `distance_correlation_signal_return`
   * `transfer_entropy_signal_to_return`

---

### 3. Horizon & Decay / Term Structure

1. IC term structure

   * `ic_1m`, `ic_5m`, `ic_15m`, `ic_30m`, `ic_60m`, `ic_240m`, `ic_1d`
     (or whatever horizons you use)
2. Half-lives & ratios

   * `ic_half_life_bars`
   * `ic_decay_ratio_60m_15m`
   * `ic_decay_ratio_1d_15m`
3. Lead-lag & causality

   * `cross_corr_peak_lag_bars`
   * `granger_causality_pvalue_signal_to_returns`

---

### 4. Turnover, Capacity & Cost Layer

1. Signal & position persistence

   * `signal_autocorr_lag1`
   * `position_autocorr_lag1`
   * `avg_holding_period_bars`
   * `avg_holding_period_days`

2. Turnover & volume

   * `portfolio_turnover_per_bar`
   * `portfolio_turnover_daily`
   * `turnover_pct_of_adv`
   * `avg_trade_size_shares`
   * `avg_trade_size_notional`
   * `participation_rate` (traded volume / market volume)

3. Capacity & cost

   * `break_even_cost_bps`
   * `expected_transaction_cost_bps` (model-based)
   * `expected_tc_drag_bps` (turnover × cost)
   * `turnover_adjusted_ic` (IC × sqrt(1 − autocorr) or similar)
   * `turnover_elasticity_of_performance` (ΔPnL / Δturnover)
   * `capacity_curve_slope` (performance vs participation)

---

### 5. Orthogonality, Factor Decomposition & Novelty

1. Simple correlations vs known risk drivers

   * `corr_with_market_return`
   * `corr_with_momentum_factor`
   * `corr_with_value_factor`
   * `corr_with_size_factor`
   * `corr_with_volatility`
   * `corr_with_volume`
   * `corr_with_vix_or_proxy`

2. Factor regressions

   * `factor_model_r2` (S vs known factor set)
   * `residual_ic_after_factors` (IC of residual)
   * `incremental_ir_in_book` (ΔIR when added to existing strategy stack)
   * `partial_corr_with_returns_controlling_factors`
   * `vif_signal` (variance inflation factor; redundancy indication)

---

### 6. Regime & Conditional Performance

1. Volatility regimes

   * `ic_low_vol_regime`
   * `ic_mid_vol_regime`
   * `ic_high_vol_regime`

2. Trend / mean-reversion regimes

   * `ic_trending_days`
   * `ic_mean_reverting_days`

3. Liquidity regimes

   * `ic_high_liquidity`
   * `ic_mid_liquidity`
   * `ic_low_liquidity`

4. Market state / macro events

   * `ic_bull_days`
   * `ic_bear_days`
   * `ic_large_gap_days`
   * `ic_macro_event_days` (FOMC, NFP, earnings)
   * `ic_quiet_days`

5. Time-of-day / intraday

   * `ic_open_30m`
   * `ic_midday`
   * `ic_close_30m`
   * `ic_by_time_bucket` (dict)

6. Conditional interactions

   * `corr_signal_times_vol_with_returns`
   * `ic_conditioned_on_spread_narrow`
   * `ic_conditioned_on_spread_wide`

---

### 7. Strategy-Level Returns (“Money Layer”)

1. Mean returns

   * `gross_mean_return_bps_per_bar`
   * `net_mean_return_bps_per_bar`
   * `gross_mean_return_bps_daily`
   * `net_mean_return_bps_daily`

2. Volatility & Sharpe-like

   * `return_vol_bps_per_bar`
   * `return_vol_annualized`
   * `sharpe_annualized`
   * `sortino_ratio`
   * `calmar_ratio`
   * `information_ratio_vs_benchmark`

3. PnL distribution shape

   * `pnl_skewness`
   * `pnl_kurtosis`
   * `pnl_percentile_1`, `pnl_percentile_5`, `pnl_percentile_95`, `pnl_percentile_99`
   * `tail_ratio` (avg gain in upper tail / avg loss in lower tail)

4. Drawdowns

   * `max_drawdown_pct`
   * `max_drawdown_bps`
   * `max_drawdown_duration_bars`
   * `avg_drawdown_pct`
   * `ulcer_index`
   * `recovery_factor` (total return / max DD)

5. Risk metrics (VaR/ES)

   * `var_95_daily`
   * `es_95_daily`
   * `var_99_daily`
   * `es_99_daily`
   * `downside_volatility`

---

### 8. Trade-Level Metrics

1. Win / loss breakdown

   * `trade_win_rate_pct`
   * `avg_win_bps`
   * `avg_loss_bps`
   * `risk_reward_ratio` (avg win / |avg loss|)
   * `profit_factor`

2. Expectancy & frequency

   * `expectancy_bps_per_trade`
   * `trades_per_day`
   * `trades_per_symbol_per_day`
   * `sqn` (System Quality Number: expectancy / std × √#trades)

3. Holding time

   * `median_trade_holding_time_bars`
   * `pct_trades_closed_same_bar`
   * `%_overnight_trades` (if applicable)

4. Pathological trade patterns

   * `%_round_trips_flat_pnl`
   * `%_stop_out_trades`
   * `%_gap_open_trades`

---

### 9. Robustness & Overfitting

1. IS vs OOS

   * `ic_is`
   * `ic_oos`
   * `oos_to_is_ic_ratio`
   * `sharpe_is`
   * `sharpe_oos`
   * `oos_to_is_sharpe_ratio`

2. Walk-forward & subperiod stability

   * `walk_forward_ic_stability_score`
   * `walk_forward_pnl_stability_score`
   * `yearly_positive_pct`
   * `quarterly_positive_pct`
   * `subsample_sharpe_dispersion` (std of Sharpe across years/quarters)

3. Resampling / randomness

   * `mc_shuffle_ic_pvalue` (randomized label test)
   * `bootstrap_ic_ci` (e.g., 95% CI)
   * `bootstrap_sharpe_ci`
   * `decimated_grid_agreement_score` (downsampled data agreement)

4. Parameter robustness

   * `param_neighborhood_stability_score` (±10% param change)
   * `num_significant_params` vs `num_tried_params`
   * `deflated_sharpe`
   * `psc_probability_success_chance`
   * `minimum_track_record_length`

---

### 10. Execution Quality & Market Impact

1. Benchmark-based execution

   * `slippage_vs_vwap_bps`
   * `slippage_vs_twap_bps`
   * `implementation_shortfall_bps`

2. Spread capture & adverse selection

   * `avg_effective_spread_bps`
   * `avg_realized_spread_bps`
   * `short_term_adverse_selection_bps`
     (post-trade price move against you)

3. Impact & liquidity footprint

   * `kyle_lambda`
   * `price_impact_per_1m_notional_bps`
   * `average_queue_position_fraction` (if you track it)
   * `fill_rate_pct`
   * `%_partial_fills`
   * `%_canceled_before_fill`

---

### 11. Advanced Time-Series / Structural Metrics

1. Structure & cycles

   * `hilbert_dominant_cycle_period`
   * `spectral_density_peak_freq`
   * `autocorr_decay_time_constant`

2. Complexity & chaos (already partly in hygiene)

   * `lempel_ziv_complexity_signal`
   * `lyapunov_exponent_signal`

---

### 12. Portfolio / Book-Level Contribution (if multiple alphas)

1. Risk & diversification

   * `correlation_to_book_pnl`
   * `beta_to_book`
   * `beta_to_benchmark`
   * `marginal_var`
   * `marginal_es`
   * `diversification_ratio` (portfolio vol / sum component vols)
   * `herfindahl_of_signal_weights`
   * `cluster_exposure_concentration` (e.g., by sector, style, regime)

2. Incremental performance

   * `incremental_sharpe_contribution`
   * `incremental_return_contribution`
   * `incremental_drawdown_contribution` (DD impact)

---

## B. Expanded `master_scorecard` Skeleton

Below is a unified dictionary you can use as the “ultimate” metric namespace. You can delete what you do not compute; keys are there so you do not forget checks.

```python
master_scorecard = {
    "data_quality_microstructure": {
        "n_bars": None,
        "n_days": None,
        "n_trades": None,
        "n_symbols": None,
        "pct_missing_signal": None,
        "pct_missing_prices": None,
        "pct_missing_volume": None,
        "pct_nonmonotonic_timestamps": None,
        "pct_bad_quotes": None,
        "pct_negative_spreads": None,
        "pct_crossed_markets": None,
        "pct_signal_outliers": None,
        "pct_return_outliers": None,
        "pct_volume_spikes": None,
        "avg_spread_bps": None,
        "median_spread_bps": None,
        "avg_depth_shares": None,
        "realized_vol_1m": None,
        "realized_vol_5m": None,
    },

    "signal_distribution_stationarity": {
        "signal_mean": None,
        "signal_std": None,
        "signal_median": None,
        "signal_pct_1": None,
        "signal_pct_5": None,
        "signal_pct_95": None,
        "signal_pct_99": None,
        "signal_iqr": None,
        "signal_skew": None,
        "signal_kurtosis": None,
        "signal_jarque_bera_pvalue": None,
        "acf_lag1_signal": None,
        "acf_lags_signal": None,          # list / array
        "pacf_lag1_signal": None,
        "hurst_exponent_signal": None,
        "adf_pvalue_signal": None,
        "kpss_pvalue_signal": None,
        "variance_ratio_signal": None,
        "shannon_entropy_signal": None,
        "approximate_entropy_signal": None,
        "sample_entropy_signal": None,
        "pct_zero_nearzero_signal": None,
        "fractal_dimension_signal": None,
        "higuchi_fractal_dimension_signal": None,
        "lyapunov_exponent_signal": None,
        "lempel_ziv_complexity_signal": None,
    },

    "signal_predictive_power": {
        "ic_pearson": None,
        "ic_spearman": None,
        "ic_kendall_tau": None,
        "ic_decimated": None,
        "ic_time_weighted": None,
        "ic_term_structure": None,        # dict{horizon: IC}
        "ic_volatility": None,
        "ic_information_ratio": None,
        "decile_spread_top_bottom_bps": None,
        "quintile_spread_q5_q1_bps": None,
        "quantile_monotonicity_score": None,
        "hit_rate_top_quantile": None,
        "hit_rate_top_vs_bottom_diff": None,
        "ols_beta_signal_to_returns": None,
        "ols_alpha_bps": None,
        "ols_tstat_signal": None,
        "hac_tstat_newey_west": None,
        "pedersen_qz_tstat": None,
        "predictive_r2_is": None,
        "predictive_r2_oos": None,
        "fama_macbeth_tstat": None,
        "fama_macbeth_r2": None,
        "auc_roc": None,
        "auc_pr": None,
        "brier_score": None,
        "confusion_matrix": None,          # nested structure
        "f1_score": None,
        "mutual_information_signal_return": None,
        "maximal_information_coefficient": None,
        "distance_correlation_signal_return": None,
        "transfer_entropy_signal_to_return": None,
    },

    "signal_term_structure": {
        "ic_1m": None,
        "ic_5m": None,
        "ic_15m": None,
        "ic_30m": None,
        "ic_60m": None,
        "ic_240m": None,
        "ic_1d": None,
        "ic_half_life_bars": None,
        "ic_decay_ratio_60m_15m": None,
        "ic_decay_ratio_1d_15m": None,
        "cross_corr_peak_lag_bars": None,
        "granger_causality_pvalue": None,
    },

    "turnover_capacity_costs": {
        "signal_autocorr_lag1": None,
        "position_autocorr_lag1": None,
        "avg_holding_period_bars": None,
        "avg_holding_period_days": None,
        "portfolio_turnover_per_bar": None,
        "portfolio_turnover_daily": None,
        "turnover_pct_of_adv": None,
        "avg_trade_size_shares": None,
        "avg_trade_size_notional": None,
        "participation_rate": None,
        "break_even_cost_bps": None,
        "expected_transaction_cost_bps": None,
        "expected_tc_drag_bps": None,
        "turnover_adjusted_ic": None,
        "turnover_elasticity_of_performance": None,
        "capacity_curve_slope": None,
    },

    "orthogonality_novelty": {
        "corr_with_market_return": None,
        "corr_with_momentum_factor": None,
        "corr_with_value_factor": None,
        "corr_with_size_factor": None,
        "corr_with_volatility": None,
        "corr_with_volume": None,
        "corr_with_vix": None,
        "factor_model_r2": None,
        "residual_ic_after_factors": None,
        "incremental_ir_in_book": None,
        "partial_corr_with_returns_controlling_factors": None,
        "vif_signal": None,
    },

    "regime_conditional": {
        "ic_low_vol_regime": None,
        "ic_mid_vol_regime": None,
        "ic_high_vol_regime": None,
        "ic_trending_days": None,
        "ic_mean_reverting_days": None,
        "ic_high_liquidity": None,
        "ic_mid_liquidity": None,
        "ic_low_liquidity": None,
        "ic_bull_days": None,
        "ic_bear_days": None,
        "ic_large_gap_days": None,
        "ic_macro_event_days": None,
        "ic_quiet_days": None,
        "ic_open_30m": None,
        "ic_midday": None,
        "ic_close_30m": None,
        "ic_by_time_bucket": None,         # dict{bucket: IC}
        "corr_signal_times_vol_with_returns": None,
        "ic_conditioned_on_spread_narrow": None,
        "ic_conditioned_on_spread_wide": None,
    },

    "strategy_returns": {
        "gross_mean_return_bps_per_bar": None,
        "net_mean_return_bps_per_bar": None,
        "gross_mean_return_bps_daily": None,
        "net_mean_return_bps_daily": None,
        "return_vol_bps_per_bar": None,
        "return_vol_annualized": None,
        "sharpe_annualized": None,
        "sortino_ratio": None,
        "calmar_ratio": None,
        "information_ratio_vs_benchmark": None,
        "pnl_skewness": None,
        "pnl_kurtosis": None,
        "pnl_percentile_1": None,
        "pnl_percentile_5": None,
        "pnl_percentile_95": None,
        "pnl_percentile_99": None,
        "tail_ratio": None,
        "max_drawdown_pct": None,
        "max_drawdown_bps": None,
        "max_drawdown_duration_bars": None,
        "avg_drawdown_pct": None,
        "ulcer_index": None,
        "recovery_factor": None,
        "var_95_daily": None,
        "es_95_daily": None,
        "var_99_daily": None,
        "es_99_daily": None,
        "downside_volatility": None,
    },

    "trade_level": {
        "trade_win_rate_pct": None,
        "avg_win_bps": None,
        "avg_loss_bps": None,
        "risk_reward_ratio": None,
        "profit_factor": None,
        "expectancy_bps_per_trade": None,
        "trades_per_day": None,
        "trades_per_symbol_per_day": None,
        "sqn": None,
        "median_trade_holding_time_bars": None,
        "pct_trades_closed_same_bar": None,
        "pct_overnight_trades": None,
        "pct_round_trips_flat_pnl": None,
        "pct_stop_out_trades": None,
        "pct_gap_open_trades": None,
    },

    "robustness_overfitting": {
        "ic_is": None,
        "ic_oos": None,
        "oos_to_is_ic_ratio": None,
        "sharpe_is": None,
        "sharpe_oos": None,
        "oos_to_is_sharpe_ratio": None,
        "walk_forward_ic_stability_score": None,
        "walk_forward_pnl_stability_score": None,
        "yearly_positive_pct": None,
        "quarterly_positive_pct": None,
        "subsample_sharpe_dispersion": None,
        "mc_shuffle_ic_pvalue": None,
        "bootstrap_ic_ci": None,
        "bootstrap_sharpe_ci": None,
        "decimated_grid_agreement_score": None,
        "param_neighborhood_stability_score": None,
        "num_significant_params": None,
        "num_tried_params": None,
        "deflated_sharpe": None,
        "psc_probability_success_chance": None,
        "minimum_track_record_length": None,
    },

    "execution_impact": {
        "slippage_vs_vwap_bps": None,
        "slippage_vs_twap_bps": None,
        "implementation_shortfall_bps": None,
        "avg_effective_spread_bps": None,
        "avg_realized_spread_bps": None,
        "short_term_adverse_selection_bps": None,
        "kyle_lambda": None,
        "price_impact_per_1m_notional_bps": None,
        "average_queue_position_fraction": None,
        "fill_rate_pct": None,
        "pct_partial_fills": None,
        "pct_canceled_before_fill": None,
    },

    "advanced_structure": {
        "hilbert_dominant_cycle_period": None,
        "spectral_density_peak_freq": None,
        "autocorr_decay_time_constant": None,
        "lempel_ziv_complexity_signal": None,
        "lyapunov_exponent_signal": None,
    },

    "portfolio_book_contribution": {
        "correlation_to_book_pnl": None,
        "beta_to_book": None,
        "beta_to_benchmark": None,
        "marginal_var": None,
        "marginal_es": None,
        "diversification_ratio": None,
        "herfindahl_of_signal_weights": None,
        "cluster_exposure_concentration": None,
        "incremental_sharpe_contribution": None,
        "incremental_return_contribution": None,
        "incremental_drawdown_contribution": None,
    },
}
```
```python
master_scorecard = {
    "data_quality_microstructure": {
        "n_bars": None,
        "n_days": None,
        "n_trades": None,
        "n_symbols": None,
        "pct_days_with_data_gaps": None,

        "pct_missing_signal": None,
        "pct_missing_prices": None,
        "pct_missing_volume": None,
        "pct_nonmonotonic_timestamps": None,
        "pct_bad_quotes": None,              # requires quotes/NBBO

        "pct_signal_outliers": None,
        "pct_return_outliers": None,
        "pct_volume_spikes": None,

        "pct_negative_spreads": None,        # requires quotes
        "pct_crossed_markets": None,         # requires quotes
        "avg_spread_bps": None,              # requires quotes
        "median_spread_bps": None,           # requires quotes
        "avg_depth_shares": None,            # requires order book

        "realized_vol_1m": None,
        "realized_vol_5m": None,
    },

    "signal_distribution_stationarity": {
        "signal_mean": None,
        "signal_std": None,
        "signal_median": None,
        "signal_pct_1": None,
        "signal_pct_5": None,
        "signal_pct_95": None,
        "signal_pct_99": None,
        "signal_iqr": None,

        "signal_skew": None,
        "signal_kurtosis": None,
        "signal_jarque_bera_pvalue": None,

        "acf_lag1_signal": None,
        "acf_lags_signal": None,                  # list / array
        "pacf_lag1_signal": None,
        "hurst_exponent_signal": None,
        "adf_pvalue_signal": None,
        "kpss_pvalue_signal": None,
        "variance_ratio_signal": None,

        "shannon_entropy_signal": None,
        "approximate_entropy_signal": None,
        "sample_entropy_signal": None,
        "pct_zero_nearzero_signal": None,

        "fractal_dimension_signal": None,
        "higuchi_fractal_dimension_signal": None,
        "lyapunov_exponent_signal": None,
        "lempel_ziv_complexity_signal": None,

        "signal_sign_flip_rate": None,
        "signal_state_persistence": None,         # avg run length of sign/quantile
        "signal_chow_break_pvalue": None,         # structural break test
    },

    "signal_predictive_power": {
        # Linear & rank correlation
        "ic_pearson": None,
        "ic_spearman": None,
        "ic_kendall_tau": None,
        "ic_decimated": None,
        "ic_time_weighted": None,
        "ic_term_structure": None,                # dict{horizon: IC}
        "ic_volatility": None,
        "ic_information_ratio": None,

        # Binned / monotonicity
        "decile_spread_top_bottom_bps": None,
        "quintile_spread_q5_q1_bps": None,
        "quantile_monotonicity_score": None,
        "hit_rate_top_quantile": None,
        "hit_rate_top_vs_bottom_diff": None,

        # Regression-based
        "ols_beta_signal_to_returns": None,
        "ols_alpha_bps": None,
        "ols_tstat_signal": None,
        "hac_tstat_newey_west": None,
        "pedersen_qz_tstat": None,
        "predictive_r2_is": None,
        "predictive_r2_oos": None,
        "fama_macbeth_tstat": None,
        "fama_macbeth_r2": None,

        # Classification-style
        "auc_roc": None,
        "auc_pr": None,
        "brier_score": None,
        "confusion_matrix": None,                # nested structure
        "f1_score": None,

        # Nonlinear dependence
        "mutual_information_signal_return": None,
        "maximal_information_coefficient": None,
        "distance_correlation_signal_return": None,
        "transfer_entropy_signal_to_return": None,

        # IC distribution / stability
        "ic_skewness": None,
        "ic_kurtosis": None,
        "pct_positive_ic": None,
        "max_ic_drawdown": None,
        "ic_ljung_box_pvalue": None,
    },

    "signal_term_structure": {
        "ic_1m": None,
        "ic_5m": None,
        "ic_15m": None,
        "ic_30m": None,
        "ic_60m": None,
        "ic_240m": None,
        "ic_1d": None,

        "ic_half_life_bars": None,
        "ic_decay_ratio_60m_15m": None,
        "ic_decay_ratio_1d_15m": None,

        "cross_corr_peak_lag_bars": None,
        "granger_causality_pvalue": None,
    },

    "turnover_capacity_costs": {
        "signal_autocorr_lag1": None,
        "position_autocorr_lag1": None,
        "avg_holding_period_bars": None,
        "avg_holding_period_days": None,

        "portfolio_turnover_per_bar": None,
        "portfolio_turnover_daily": None,
        "turnover_pct_of_adv": None,
        "avg_trade_size_shares": None,
        "avg_trade_size_notional": None,
        "participation_rate": None,

        "break_even_cost_bps": None,
        "expected_transaction_cost_bps": None,
        "expected_tc_drag_bps": None,

        "turnover_adjusted_ic": None,
        "turnover_elasticity_of_performance": None,
        "capacity_curve_slope": None,

        # Nonlinearity / crowding
        "slippage_vs_participation_curve": None,  # mapping: adv% -> slippage
        "impact_concavity_cost": None,           # deviation from linear cost
        "capacity_crowding_indicator": None,     # performance drop vs ADV%
    },

    "orthogonality_novelty": {
        "corr_with_market_return": None,
        "corr_with_momentum_factor": None,       # requires factors
        "corr_with_value_factor": None,
        "corr_with_size_factor": None,
        "corr_with_volatility": None,
        "corr_with_volume": None,
        "corr_with_vix": None,                   # or proxy

        "factor_model_r2": None,
        "residual_ic_after_factors": None,
        "incremental_ir_in_book": None,
        "partial_corr_with_returns_controlling_factors": None,
        "vif_signal": None,
    },

    # New: alignment vs microstructure primitives (AGG2-native)
    "microstructure_alignment": {
        "corr_with_signed_volume": None,         # corr(signal, v_i)
        "corr_with_trade_sign": None,            # corr(signal, ε_i)
        "corr_with_duration": None,              # corr(signal, d_i)
        "corr_with_child_count": None,           # corr(signal, c_eff)
        "corr_with_order_flow_imbalance": None,  # short-window OFI / OFIB
        "corr_with_vpin": None,
        "corr_with_trade_intensity": None,       # 1/d or Hawkes λ
        "corr_with_meta_order_pressure": None,   # smoothed Q(t) / runs
    },

    "regime_conditional": {
        "ic_low_vol_regime": None,
        "ic_mid_vol_regime": None,
        "ic_high_vol_regime": None,

        "ic_trending_days": None,
        "ic_mean_reverting_days": None,

        "ic_high_liquidity": None,
        "ic_mid_liquidity": None,
        "ic_low_liquidity": None,

        "ic_bull_days": None,
        "ic_bear_days": None,
        "ic_large_gap_days": None,
        "ic_macro_event_days": None,             # needs event calendar
        "ic_quiet_days": None,

        "ic_open_30m": None,
        "ic_midday": None,
        "ic_close_30m": None,
        "ic_by_time_bucket": None,               # dict{bucket: IC}

        "corr_signal_times_vol_with_returns": None,
        "ic_conditioned_on_spread_narrow": None, # needs spread
        "ic_conditioned_on_spread_wide": None,
    },

    "strategy_returns": {
        "gross_mean_return_bps_per_bar": None,
        "net_mean_return_bps_per_bar": None,
        "gross_mean_return_bps_daily": None,
        "net_mean_return_bps_daily": None,

        "return_vol_bps_per_bar": None,
        "return_vol_annualized": None,
        "sharpe_annualized": None,
        "sortino_ratio": None,
        "calmar_ratio": None,
        "information_ratio_vs_benchmark": None,

        "pnl_skewness": None,
        "pnl_kurtosis": None,
        "pnl_percentile_1": None,
        "pnl_percentile_5": None,
        "pnl_percentile_95": None,
        "pnl_percentile_99": None,
        "tail_ratio": None,

        "max_drawdown_pct": None,
        "max_drawdown_bps": None,
        "max_drawdown_duration_bars": None,
        "avg_drawdown_pct": None,
        "ulcer_index": None,
        "recovery_factor": None,

        "var_95_daily": None,
        "es_95_daily": None,
        "var_99_daily": None,
        "es_99_daily": None,
        "downside_volatility": None,
    },

    "trade_level": {
        "trade_win_rate_pct": None,
        "avg_win_bps": None,
        "avg_loss_bps": None,
        "risk_reward_ratio": None,
        "profit_factor": None,

        "expectancy_bps_per_trade": None,
        "trades_per_day": None,
        "trades_per_symbol_per_day": None,
        "sqn": None,                              # System Quality Number

        "median_trade_holding_time_bars": None,
        "pct_trades_closed_same_bar": None,
        "pct_overnight_trades": None,

        "pct_round_trips_flat_pnl": None,
        "pct_stop_out_trades": None,
        "pct_gap_open_trades": None,
    },

    "robustness_overfitting": {
        "ic_is": None,
        "ic_oos": None,
        "oos_to_is_ic_ratio": None,
        "sharpe_is": None,
        "sharpe_oos": None,
        "oos_to_is_sharpe_ratio": None,

        "walk_forward_ic_stability_score": None,
        "walk_forward_pnl_stability_score": None,
        "yearly_positive_pct": None,
        "quarterly_positive_pct": None,
        "subsample_sharpe_dispersion": None,

        "mc_shuffle_ic_pvalue": None,
        "bootstrap_ic_ci": None,
        "bootstrap_sharpe_ci": None,
        "decimated_grid_agreement_score": None,

        "param_neighborhood_stability_score": None,
        "num_significant_params": None,
        "num_tried_params": None,
        "deflated_sharpe": None,
        "psc_probability_success_chance": None,
        "minimum_track_record_length": None,
    },

    "execution_impact": {
        "slippage_vs_vwap_bps": None,
        "slippage_vs_twap_bps": None,
        "implementation_shortfall_bps": None,

        "avg_effective_spread_bps": None,        # needs quotes
        "avg_realized_spread_bps": None,         # needs quotes
        "short_term_adverse_selection_bps": None,

        "kyle_lambda": None,
        "price_impact_per_1m_notional_bps": None,

        "average_queue_position_fraction": None, # needs order info
        "fill_rate_pct": None,
        "pct_partial_fills": None,
        "pct_canceled_before_fill": None,
    },

    "advanced_structure": {
        "hilbert_dominant_cycle_period": None,
        "spectral_density_peak_freq": None,
        "autocorr_decay_time_constant": None,

        "lempel_ziv_complexity_signal": None,
        "lyapunov_exponent_signal": None,
    },

    "portfolio_book_contribution": {
        "correlation_to_book_pnl": None,
        "beta_to_book": None,
        "beta_to_benchmark": None,

        "marginal_var": None,
        "marginal_es": None,
        "diversification_ratio": None,
        "herfindahl_of_signal_weights": None,
        "cluster_exposure_concentration": None,

        "incremental_sharpe_contribution": None,
        "incremental_return_contribution": None,
        "incremental_drawdown_contribution": None,

        # PCA overlap / redundancy vs alpha book
        "pca_loading_on_first_component": None,
        "pca_explained_variance_share": None,
    },

    # Optional explicit tail / ruin metrics
    "risk_tail_ruin": {
        "cdar_95_daily": None,                    # Conditional drawdown at risk
        "cdar_99_daily": None,
        "probability_of_ruin": None,              # under a bankroll model
    },
}
```
