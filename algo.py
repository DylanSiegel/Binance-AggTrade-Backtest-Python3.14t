"""
algo.py — Adaptive Microstructure Alpha Engine (v4.1)

Designed for:
- Python 3.14t (free-threaded, -X gil=0)
- AGG2 backtest stack (BTCUSDT etc.)

Key properties:
- Pure stdlib, no NumPy / pandas / multiprocessing.
- Single-pass tick processing with volume bars.
- Adaptive EWMA correlation-based feature weighting.
- Continuous alpha score A_t with quantile-based threshold.
- Uses:
    * Flow–impact residual
    * Flow reversal (short vs long horizon)
    * Iceberg / exhaustion proxy
    * Short-horizon toxicity regime
    * Microbar sign persistence (trendiness of micro flow)

External interface:
- Row: AGG2 row tuple.
- AlphaEngine(symbol).update_batch(rows) -> list[(ts_ms, px, side)]
"""

import math
import statistics
from collections import deque
from collections.abc import Iterable

# AGG2 Row: id, px, qt, fi, cnt, flags, ts, side
Row = tuple[int, int, int, int, int, int, int, int]
RowIter = Iterable[Row]


class EWMAStat:
    """
    Exponentially Weighted Mean/Variance tracker.
    """

    __slots__ = ("alpha", "initialized", "mean", "var")

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.initialized = False
        self.mean = 0.0
        self.var = 0.0

    def update(self, x: float) -> None:
        a = self.alpha
        if not self.initialized:
            self.mean = x
            self.var = 0.0
            self.initialized = True
            return
        m = self.mean
        delta = x - m
        m_new = m + a * delta
        # Approximate EWMA variance update
        self.var = (1.0 - a) * (self.var + a * delta * delta)
        self.mean = m_new

    def stats(self) -> tuple[float, float]:
        if not self.initialized:
            return (0.0, 0.0)
        v = self.var
        if v < 0.0:
            v = 0.0
        return (self.mean, math.sqrt(v))


class EWMACorr:
    """
    Exponentially Weighted Correlation tracker between x and y.
    """

    __slots__ = (
        "alpha",
        "initialized",
        "Ex",
        "Ey",
        "Exx",
        "Eyy",
        "Exy",
        "count",
    )

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.initialized = False
        self.Ex = 0.0
        self.Ey = 0.0
        self.Exx = 0.0
        self.Eyy = 0.0
        self.Exy = 0.0
        self.count = 0

    def update(self, x: float, y: float) -> None:
        a = self.alpha
        if not self.initialized:
            self.Ex = x
            self.Ey = y
            self.Exx = x * x
            self.Eyy = y * y
            self.Exy = x * y
            self.initialized = True
            self.count = 1
            return

        self.Ex = (1.0 - a) * self.Ex + a * x
        self.Ey = (1.0 - a) * self.Ey + a * y
        self.Exx = (1.0 - a) * self.Exx + a * (x * x)
        self.Eyy = (1.0 - a) * self.Eyy + a * (y * y)
        self.Exy = (1.0 - a) * self.Exy + a * (x * y)
        self.count += 1

    def corr(self) -> float:
        if not self.initialized or self.count < 50:
            return 0.0
        Ex = self.Ex
        Ey = self.Ey
        Exx = self.Exx
        Eyy = self.Eyy
        Exy = self.Exy

        var_x = Exx - Ex * Ex
        var_y = Eyy - Ey * Ey
        if var_x <= 1e-12 or var_y <= 1e-12:
            return 0.0
        num = Exy - Ex * Ey
        den = math.sqrt(var_x * var_y)
        if den <= 0.0:
            return 0.0
        return num / den


class AlphaEngine:
    __slots__ = (
        # Tick tracking
        "last_ts_ms",
        "last_lpx",
        "prev_bar_lpx",
        "cum_q",

        # Flow windows
        "flow_win_short",   # 5m
        "flow_sum_short",
        "flow_vol_short",
        "flow_win_long",    # 30m
        "flow_sum_long",
        "flow_vol_long",
        "flow_long_abs_ewma",
        "flow_short_abs_ewma",
        "meta_flow_sign",

        # Main volume bar (execution bar)
        "bar_target",
        "bar_vol",
        "bar_buy_vol",
        "bar_sell_vol",
        "bar_csum",
        "bar_trades",
        "bar_open_px",
        "bar_high_px",
        "bar_low_px",
        "bar_close_px",
        "bar_timestamp",
        "avg_bar_ms",

        # Micro volume bar (for micro-flow sign persistence)
        "mvb_target",
        "mvb_vol",
        "mvb_buy_vol",
        "mvb_sell_vol",
        "mvb_open_px",
        "mvb_timestamp",
        "mvb_sign_hist",

        # Residual / impact model
        "window",           # (ts, lpx, cum_q) 3h window
        "anchor_ts",
        "anchor_x",
        "anchor_Q",
        "intercept",
        "resid_hist",
        "resid_mad",
        "stats_counter",

        # Realized volatility on main bars
        "ret_hist",
        "sigma",

        # Iceberg / exhaustion
        "ice_hist",
        "ice_ewma",
        "ice_z_prev",
        "ice_z_now",

        # Toxicity
        "tox_ewma",

        # Adaptive feature stats
        "feature_corrs",
        "feature_weights",
        "prev_features",
        "have_prev_features",
        "feature_update_skip",
        "feature_update_ctr",
        "alpha_scale",
        "max_weight",
        "min_weight_for_trading",
        "corr_alpha",

        # Alpha distribution (for threshold)
        "abs_alpha_hist",
        "alpha_quantile",
        "min_alpha_hist",

        # Misc
        "symbol",
    )

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

        # Tick tracking
        self.last_ts_ms = 0
        self.last_lpx = 0.0
        self.prev_bar_lpx = 0.0
        self.cum_q = 0.0

        # Flow windows
        self.flow_win_short = deque()  # (ts, signed_flow, vol)
        self.flow_sum_short = 0.0
        self.flow_vol_short = 0.0

        self.flow_win_long = deque()
        self.flow_sum_long = 0.0
        self.flow_vol_long = 0.0

        self.flow_long_abs_ewma = EWMAStat(alpha=0.01)
        self.flow_short_abs_ewma = EWMAStat(alpha=0.01)
        self.meta_flow_sign = 0.0

        # Main bar: moderate volume for more observations
        self.bar_target = 250.0
        self.bar_vol = 0.0
        self.bar_buy_vol = 0.0
        self.bar_sell_vol = 0.0
        self.bar_csum = 0.0
        self.bar_trades = 0
        self.bar_open_px = 0.0
        self.bar_high_px = 0.0
        self.bar_low_px = 0.0
        self.bar_close_px = 0.0
        self.bar_timestamp = 0
        self.avg_bar_ms = 60_000.0  # initial guess ~1 minute

        # Micro bar
        self.mvb_target = 50.0
        self.mvb_vol = 0.0
        self.mvb_buy_vol = 0.0
        self.mvb_sell_vol = 0.0
        self.mvb_open_px = 0.0
        self.mvb_timestamp = 0
        self.mvb_sign_hist = deque(maxlen=64)

        # Residual / impact model
        self.window = deque()  # (ts, lpx, cum_q)
        self.anchor_ts = 0
        self.anchor_x = 0.0
        self.anchor_Q = 0.0
        self.intercept = 0.0

        self.resid_hist = deque(maxlen=512)
        self.resid_mad = 0.001
        self.stats_counter = 0

        # Volatility on main bars
        self.ret_hist = deque(maxlen=256)
        self.sigma = 0.0005  # ~5 bps initial

        # Iceberg / exhaustion
        self.ice_hist = deque(maxlen=64)
        self.ice_ewma = EWMAStat(alpha=0.02)
        self.ice_z_prev = 0.0
        self.ice_z_now = 0.0

        # Toxicity
        self.tox_ewma = EWMAStat(alpha=0.02)

        # Features and correlations
        self.corr_alpha = 0.01
        num_features = 5  # residual, flow, iceberg, toxicity, micro-sign-persistence
        self.feature_corrs = [EWMACorr(self.corr_alpha) for _ in range(num_features)]
        self.feature_weights = [0.0 for _ in range(num_features)]
        self.prev_features = None
        self.have_prev_features = False

        self.feature_update_skip = 5
        self.feature_update_ctr = 0
        self.alpha_scale = 1.0
        self.max_weight = 1.0
        self.min_weight_for_trading = 0.05

        # Alpha distribution for threshold
        self.abs_alpha_hist = deque(maxlen=1024)
        self.alpha_quantile = 0.97   # roughly top 3% absolute alpha
        self.min_alpha_hist = 200    # bars before using quantile

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_flow_windows(self, ts: int, signed_flow: float, total_vol: float) -> None:
        # 5 minute window
        FIVE_MIN = 300_000
        self.flow_win_short.append((ts, signed_flow, total_vol))
        self.flow_sum_short += signed_flow
        self.flow_vol_short += total_vol

        while self.flow_win_short and ts - self.flow_win_short[0][0] > FIVE_MIN:
            _, f, v = self.flow_win_short.popleft()
            self.flow_sum_short -= f
            self.flow_vol_short -= v

        # 30 minute window
        THIRTY_MIN = 1_800_000
        self.flow_win_long.append((ts, signed_flow, total_vol))
        self.flow_sum_long += signed_flow
        self.flow_vol_long += total_vol

        while self.flow_win_long and ts - self.flow_win_long[0][0] > THIRTY_MIN:
            _, f, v = self.flow_win_long.popleft()
            self.flow_sum_long -= f
            self.flow_vol_long -= v

    def _update_residual_model(self, ts: int, lpx: float, signed_flow: float) -> float:
        """
        Simplified directional impact model: anchor in ~3h window.
        """
        self.cum_q += signed_flow
        self.window.append((ts, lpx, self.cum_q))

        THREE_HOURS = 10_800_000
        while self.window and ts - self.window[0][0] > THREE_HOURS:
            self.window.popleft()

        if self.window:
            self.anchor_ts, self.anchor_x, self.anchor_Q = self.window[0]
        else:
            self.anchor_ts, self.anchor_x, self.anchor_Q = ts, lpx, self.cum_q

        dq_raw = self.cum_q - self.anchor_Q
        dq_abs = abs(dq_raw) + 1.0
        impact_sign = 1.0 if dq_raw >= 0.0 else -1.0

        theo = self.anchor_x + self.intercept + (0.4 * impact_sign * math.log(dq_abs))
        resid = lpx - theo
        self.resid_hist.append(resid)

        # update robust stats occasionally
        self.stats_counter += 1
        if self.stats_counter >= 10 and len(self.resid_hist) >= 50:
            self.stats_counter = 0
            vals = list(self.resid_hist)
            try:
                med = statistics.median(vals)
                dev = [abs(v - med) for v in vals]
                mad = statistics.median(dev) or 0.0001
                self.resid_mad = mad

                # update intercept using sparse sampling
                if len(self.window) >= 40:
                    bases: list[float] = []
                    x0 = self.anchor_x
                    q0 = self.anchor_Q
                    for i in range(0, len(self.window), 4):
                        _, xi, Qi = self.window[i]
                        dQi = Qi - q0
                        s_i = 1.0 if dQi >= 0.0 else -1.0
                        bases.append(
                            xi
                            - x0
                            - (0.4 * s_i * math.log(abs(dQi) + 1.0))
                        )
                    if bases:
                        self.intercept = statistics.median(bases)
            except Exception:
                pass

        return resid

    def _update_sigma_and_returns(self, lpx: float) -> float:
        """
        Update realized volatility using main-bar log returns.
        Returns the most recent return.
        """
        if self.prev_bar_lpx == 0.0:
            self.prev_bar_lpx = lpx
            return 0.0
        r = lpx - self.prev_bar_lpx
        self.prev_bar_lpx = lpx

        self.ret_hist.append(r)
        if len(self.ret_hist) >= 20:
            m2 = sum(x * x for x in self.ret_hist) / len(self.ret_hist)
            if m2 > 0.0:
                self.sigma = math.sqrt(m2)
        return r

    def _compute_iceberg_score(
        self,
        price_change: float,
        qty: float,
        count: int,
    ) -> float:
        """
        Iceberg score: high count, high volume, low impact.
        Also updates EWMA z-scores for exhaustion logic.
        """
        sig = self.sigma if self.sigma > 1e-8 else 1e-8
        if qty <= 0.0:
            score = 0.0
        else:
            impact_rel = abs(price_change) / (qty * sig)
            if impact_rel > 1.0:
                impact_rel = 1.0
            score = float(count) * (1.0 - impact_rel)

        self.ice_hist.append(score)
        self.ice_ewma.update(score)
        mu_ice, sd_ice = self.ice_ewma.stats()
        prev_z = self.ice_z_now
        if sd_ice > 1e-8:
            z_now = (score - mu_ice) / sd_ice
        else:
            z_now = 0.0
        self.ice_z_prev = prev_z
        self.ice_z_now = z_now
        return score

    def _micro_sign_persistence(self) -> float:
        """
        Approximate micro-flow sign autocorrelation using micro-bar signs.
        Returns value in [-1, 1].
        """
        hist = self.mvb_sign_hist
        n = len(hist)
        if n < 4:
            return 0.0
        num = 0.0
        den = 0.0
        prev = hist[0]
        for i in range(1, n):
            s = hist[i]
            num += s * prev
            den += 1.0
            prev = s
        if den == 0.0:
            return 0.0
        # Already in [-1,1]
        return num / den

    def _feature_vector(
        self,
        ts: int,
        resid: float,
        bar_signed_flow: float,
        total_vol: float,
        ice_score: float,
    ) -> list[float]:
        """
        Build feature vector for this main bar.
        Features in [-1, 1].
        """

        # 1) Residual feature: large positive resid is bearish, negative is bullish
        if self.resid_mad > 0.0:
            z_resid = resid / self.resid_mad
        else:
            z_resid = 0.0
        z_resid = max(-10.0, min(10.0, z_resid))
        f_resid = -math.tanh(0.5 * z_resid)

        # 2) Flow reversal feature: long vs short horizon imbalance
        eps = 1e-8
        long_norm = 0.0
        short_norm = 0.0

        if self.flow_vol_long > eps:
            long_norm = self.flow_sum_long / self.flow_vol_long
        if self.flow_vol_short > eps:
            short_norm = self.flow_sum_short / self.flow_vol_short

        # Update EWMA of absolute flow norms to get adaptive scale
        if self.flow_vol_long > eps:
            self.flow_long_abs_ewma.update(abs(long_norm))
        if self.flow_vol_short > eps:
            self.flow_short_abs_ewma.update(abs(short_norm))

        mu_long, _ = self.flow_long_abs_ewma.stats()
        mu_short, _ = self.flow_short_abs_ewma.stats()

        # meta-flow sign: direction of longer-term flow
        if long_norm > 0.0:
            self.meta_flow_sign = 1.0
        elif long_norm < 0.0:
            self.meta_flow_sign = -1.0
        else:
            self.meta_flow_sign = 0.0

        # Gate: only consider if both norms are meaningfully non-zero
        min_scale_long = max(mu_long * 0.5, 1e-5)
        min_scale_short = max(mu_short * 0.5, 1e-5)

        if (
            abs(long_norm) < min_scale_long
            or abs(short_norm) < min_scale_short
        ):
            f_flow = 0.0
        else:
            # Reversal: long and short with opposite signs
            flow_reversal_raw = -(long_norm * short_norm)
            f_flow = math.tanh(2.0 * flow_reversal_raw)

        # 3) Iceberg exhaustion feature:
        # If we were recently in high z_ice and now z_ice dropped,
        # fade prior meta-flow direction.
        z_prev = self.ice_z_prev
        z_now = self.ice_z_now

        f_ice = 0.0
        if abs(self.meta_flow_sign) > 0.0:
            # exhaustion if prior was elevated and now normalized/lower
            if z_prev > 1.5 and z_now < 0.5:
                # Fade previous meta-flow
                f_ice = -self.meta_flow_sign
            elif z_prev < -1.5 and z_now > -0.5:
                # Opposite meta-flow case (persistent selling exhausted)
                f_ice = -self.meta_flow_sign

        # 4) Toxicity / regime feature:
        tox_raw = 0.0
        if self.flow_vol_short > eps:
            tox_raw = abs(self.flow_sum_short) / self.flow_vol_short

        self.tox_ewma.update(tox_raw)
        mu_tox, sd_tox = self.tox_ewma.stats()
        if sd_tox > 1e-8:
            z_tox = (tox_raw - mu_tox) / sd_tox
        else:
            z_tox = 0.0

        # High toxicity is bad for MR; moderate is okay.
        f_tox = -math.tanh(0.5 * z_tox)

        # 5) Microbar sign persistence: positive = trending micro-flow
        micro_pers = self._micro_sign_persistence()
        # Let correlation decide whether trendiness is good or bad
        f_mvb = micro_pers

        return [f_resid, f_flow, f_ice, f_tox, f_mvb]

    def _update_feature_weights(self, bar_return: float) -> None:
        """
        Update EWMA correlations and map them to weights.
        Uses s_{t-1} and r_t.
        """
        if not self.have_prev_features or self.prev_features is None:
            return

        feats_prev = self.prev_features
        for i, f_prev in enumerate(feats_prev):
            self.feature_corrs[i].update(f_prev, bar_return)

        self.feature_update_ctr += 1
        if self.feature_update_ctr < self.feature_update_skip:
            return
        self.feature_update_ctr = 0

        # Re-compute weights from correlations
        for i, stat in enumerate(self.feature_corrs):
            c = stat.corr()
            w = c * self.alpha_scale
            if w > self.max_weight:
                w = self.max_weight
            elif w < -self.max_weight:
                w = -self.max_weight
            self.feature_weights[i] = w

    def _alpha_and_signal(
        self,
        ts: int,
        features: list[float],
    ) -> int:
        """
        Compute alpha score A_t and decide signal based on adaptive quantile
        of |alpha|. Returns -1, 0, +1.
        """
        # Linear combo
        A = 0.0
        for w, f in zip(self.feature_weights, features):
            A += w * f

        abs_A = abs(A)
        hist = self.abs_alpha_hist

        # Can we trade yet?
        max_w = 0.0
        for w in self.feature_weights:
            if abs(w) > max_w:
                max_w = abs(w)

        can_trade = (
            len(hist) >= self.min_alpha_hist
            and max_w >= self.min_weight_for_trading
        )

        sig = 0

        if can_trade:
            # Threshold from previous alpha history (exclude current A)
            arr = sorted(hist)
            n = len(arr)
            if n > 0:
                idx = int(round((n - 1) * self.alpha_quantile))
                if idx < 0:
                    idx = 0
                if idx >= n:
                    idx = n - 1
                thr = arr[idx]
                if thr < 0.0:
                    thr = 0.0

                if abs_A >= thr and A != 0.0:
                    sig = 1 if A > 0.0 else -1

        # Update history and previous features for next bar
        hist.append(abs_A)
        self.prev_features = features
        self.have_prev_features = True

        return sig

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def update_batch(self, rows: RowIter):
        """
        High-performance vectorized update.

        Returns list of (ts_ms, px, side) trade signals.
        """

        PX_DIV = 100_000_000.0
        QT_DIV = 100_000_000.0

        _log = math.log
        _abs = abs

        signals: list[tuple[int, float, int]] = []

        # Local cache for main bar
        b_vol = self.bar_vol
        b_buy = self.bar_buy_vol
        b_sell = self.bar_sell_vol
        b_csum = self.bar_csum
        b_trades = self.bar_trades
        b_open = self.bar_open_px
        b_high = self.bar_high_px
        b_low = self.bar_low_px
        b_target = self.bar_target

        # Micro bar
        mvb_vol = self.mvb_vol
        mvb_buy = self.mvb_buy_vol
        mvb_sell = self.mvb_sell_vol
        mvb_open = self.mvb_open_px
        mvb_target = self.mvb_target

        last_px = 0.0
        last_ts = self.last_ts_ms

        for r in rows:
            px_raw = r[1]
            if px_raw == 0:
                continue

            px = px_raw / PX_DIV
            ts = r[6]
            qt = r[2] / QT_DIV

            # side: 1.0 = aggressive buy, -1.0 = aggressive sell
            side = 1.0 if r[7] == 1 else -1.0
            c = r[4]

            if self.last_lpx == 0.0:
                self.last_lpx = _log(px)
                b_open = px
                b_high = px
                b_low = px
                mvb_open = px
                last_px = px
                last_ts = ts
                continue

            vol = _abs(qt)
            signed_flow = side * vol

            # ---------------- MAIN BAR ACCUMULATION ----------------
            b_vol += vol
            if side > 0.0:
                b_buy += vol
            else:
                b_sell += vol
            b_csum += c
            b_trades += 1

            if px > b_high:
                b_high = px
            if px < b_low:
                b_low = px

            # ---------------- MICRO BAR ACCUMULATION ----------------
            mvb_vol += vol
            if side > 0.0:
                mvb_buy += vol
            else:
                mvb_sell += vol

            # close micro bar if target reached
            if mvb_vol >= mvb_target:
                mvb_close = px
                mvb_ts = ts
                if mvb_open > 0.0 and mvb_close > 0.0:
                    lr = _log(mvb_close) - _log(mvb_open)
                    mvb_sign = 0.0
                    if lr > 0.0:
                        mvb_sign = 1.0
                    elif lr < 0.0:
                        mvb_sign = -1.0
                    if mvb_sign != 0.0:
                        self.mvb_sign_hist.append(mvb_sign)
                # reset micro bar
                mvb_vol = 0.0
                mvb_buy = 0.0
                mvb_sell = 0.0
                mvb_open = px
                self.mvb_timestamp = mvb_ts

            # close main bar if target reached
            last_px = px
            last_ts = ts

            if b_vol >= b_target:
                self.bar_vol = b_vol
                self.bar_buy_vol = b_buy
                self.bar_sell_vol = b_sell
                self.bar_csum = b_csum
                self.bar_trades = b_trades
                self.bar_open_px = b_open
                self.bar_high_px = b_high
                self.bar_low_px = b_low
                self.bar_close_px = last_px
                self.bar_timestamp = ts

                # update bar duration estimate
                if self.last_ts_ms > 0 and ts > self.last_ts_ms:
                    dt_ms = ts - self.last_ts_ms
                    self.avg_bar_ms = 0.9 * self.avg_bar_ms + 0.1 * dt_ms

                lpx = _log(last_px)
                self.last_lpx = lpx

                # FLOW windows based on bar signed flow
                bar_signed_flow = self.bar_buy_vol - self.bar_sell_vol
                self._update_flow_windows(ts, bar_signed_flow, self.bar_vol)

                # Residual / impact model
                resid = self._update_residual_model(ts, lpx, bar_signed_flow)

                # Volatility update
                bar_ret = self._update_sigma_and_returns(lpx)

                # Iceberg score (updates z_ice as well)
                price_change = self.bar_close_px - self.bar_open_px
                ice_score = self._compute_iceberg_score(
                    price_change,
                    self.bar_vol,
                    self.bar_trades,
                )

                # Update feature weights using previous features and this bar's return
                self._update_feature_weights(bar_ret)

                # Build current feature vector and alpha
                feats = self._feature_vector(
                    ts,
                    resid,
                    bar_signed_flow,
                    self.bar_vol,
                    ice_score,
                )
                sig = self._alpha_and_signal(ts, feats)

                if sig != 0:
                    signals.append((ts, last_px, sig))

                # reset main bar accumulators
                b_vol = 0.0
                b_buy = 0.0
                b_sell = 0.0
                b_csum = 0.0
                b_trades = 0
                b_open = last_px
                b_high = last_px
                b_low = last_px

        # persist bar state
        self.bar_vol = b_vol
        self.bar_buy_vol = b_buy
        self.bar_sell_vol = b_sell
        self.bar_csum = b_csum
        self.bar_trades = b_trades
        self.bar_open_px = b_open
        self.bar_high_px = b_high
        self.bar_low_px = b_low

        self.mvb_vol = mvb_vol
        self.mvb_buy_vol = mvb_buy
        self.mvb_sell_vol = mvb_sell
        self.mvb_open_px = mvb_open

        self.last_ts_ms = last_ts
        if last_px > 0.0:
            self.last_lpx = _log(last_px)

        return signals
