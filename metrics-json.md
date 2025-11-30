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
