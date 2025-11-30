"""
algo.py — Adaptive Microstructure Alpha Engine (v4.0)

Designed for:
- Python 3.14t (free-threaded, -X gil=0)
- Your AGG2 backtest stack (BTCUSDT futures data, etc.)

Key properties:
- Pure stdlib, no NumPy / pandas / multiprocessing.
- Single-pass tick processing with volume bars.
- Adaptive feature normalization and online correlation-based weighting.
- Continuous alpha score A_t with dynamic thresholds (no fixed bps levels).

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


class OnlineCorr:
    """Online correlation tracker between x and y using running sums."""

    __slots__ = ("n", "sum_x", "sum_y", "sum_xx", "sum_yy", "sum_xy")

    def __init__(self) -> None:
        self.n = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_xx = 0.0
        self.sum_yy = 0.0
        self.sum_xy = 0.0

    def update(self, x: float, y: float) -> None:
        self.n += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_xx += x * x
        self.sum_yy += y * y
        self.sum_xy += x * y

    def corr(self) -> float:
        n = self.n
        if n < 10:
            return 0.0
        num = n * self.sum_xy - self.sum_x * self.sum_y
        den_x = n * self.sum_xx - self.sum_x * self.sum_x
        den_y = n * self.sum_yy - self.sum_y * self.sum_y
        if den_x <= 0.0 or den_y <= 0.0:
            return 0.0
        return num / math.sqrt(den_x * den_y)


class OnlineMeanVar:
    """Welford-style mean/std estimator."""

    __slots__ = ("n", "mean", "M2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def stats(self) -> tuple[float, float]:
        if self.n < 2:
            return (self.mean, 0.0)
        var = self.M2 / (self.n - 1)
        if var < 0.0:
            var = 0.0
        return (self.mean, math.sqrt(var))


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

        # Micro volume bar (flow / sign autocorr)
        "mvb_target",
        "mvb_vol",
        "mvb_buy_vol",
        "mvb_sell_vol",
        "mvb_open_px",
        "mvb_close_px",
        "mvb_timestamp",
        "mvb_sign_hist",    # recent micro-bar sign of return

        # Residual / impact model (simplified directional model)
        "window",           # (ts, lpx, cum_q) 3h window
        "anchor_ts",
        "anchor_x",
        "anchor_Q",
        "intercept",
        "resid_hist",
        "resid_mad",
        "stats_counter",

        # Realized volatility on main bars
        "ret_hist",         # log returns of main bars
        "sigma",

        # Iceberg / exhaustion
        "ice_hist",         # iceberg score history
        "last_ice_event_ts",

        # Adaptive feature stats
        "feature_stats",    # list[OnlineCorr]
        "feature_weights",  # list[float]
        "prev_features",    # list[float] | None
        "have_prev_features",
        "abs_alpha_stats",  # OnlineMeanVar
        "last_alpha",
        "last_alpha_ts",

        # Misc
        "feature_update_skip",
        "feature_update_ctr",
        "alpha_scale",
        "max_weight",
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

        # Main bar: modest size for more observations
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
        self.avg_bar_ms = 60_000.0  # initial guess: ~1 minute

        # Micro volume bar
        self.mvb_target = 50.0
        self.mvb_vol = 0.0
        self.mvb_buy_vol = 0.0
        self.mvb_sell_vol = 0.0
        self.mvb_open_px = 0.0
        self.mvb_close_px = 0.0
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
        self.last_ice_event_ts = 0

        # Adaptive feature stats: 4 core features
        self.feature_stats = [OnlineCorr() for _ in range(4)]
        self.feature_weights = [0.0 for _ in range(4)]
        self.prev_features: list[float] | None = None
        self.have_prev_features = False

        self.abs_alpha_stats = OnlineMeanVar()
        self.last_alpha = 0.0
        self.last_alpha_ts = 0

        self.feature_update_skip = 10  # update weights every N bars
        self.feature_update_ctr = 0
        self.alpha_scale = 1.0
        self.max_weight = 1.0

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
        Update simplified impact model: anchor at oldest point in ~3h window,
        with cumulative signed flow, and compute residual.
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

        # Impact factor 0.4 (Kyle lambda proxy). Intercept adapts over time.
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
                    bases = []
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

    def _compute_iceberg_score(self, price_change: float, qty: float, count: int) -> float:
        """
        Iceberg score: high count, high volume, low impact.
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
        return score

    def _feature_vector(
        self,
        ts: int,
        resid: float,
        signed_flow: float,
        total_vol: float,
        ice_score: float,
    ) -> list[float]:
        """
        Build a 4-dimensional feature vector for this main bar.
        Each feature is squashed into [-1, 1].
        """

        # 1) Residual feature: large positive resid is bearish, negative is bullish
        if self.resid_mad > 0.0:
            z_resid = resid / self.resid_mad
        else:
            z_resid = 0.0
        z_resid = max(-10.0, min(10.0, z_resid))
        f_resid = -math.tanh(0.5 * z_resid)  # positive when resid is negative (cheap vs impact)

        # 2) Flow reversal feature
        eps = 1e-8
        long_norm = 0.0
        if self.flow_vol_long > eps:
            long_norm = self.flow_sum_long / self.flow_vol_long
        short_norm = 0.0
        if self.flow_vol_short > eps:
            short_norm = self.flow_sum_short / self.flow_vol_short

        # large discrepancy between long-term imbalance and short-term move is "reversal"
        flow_reversal_raw = -(long_norm * short_norm)
        f_flow = math.tanh(2.0 * flow_reversal_raw)

        # 3) Iceberg exhaustion feature:
        if len(self.ice_hist) >= 10:
            mu_ice = statistics.fmean(self.ice_hist)
            sd_ice = statistics.pstdev(self.ice_hist, mu_ice)
            if sd_ice > 1e-8:
                z_ice = (ice_score - mu_ice) / sd_ice
            else:
                z_ice = 0.0
        else:
            z_ice = 0.0

        if len(self.ice_hist) >= 2:
            prev = list(self.ice_hist)[-2]
            delta_ice = ice_score - prev
        else:
            delta_ice = 0.0

        # negative delta after generally high scores => exhaustion (mean-reversion friendly)
        f_ice = 0.0
        if len(self.ice_hist) >= 10:
            f_ice = -math.tanh(0.5 * delta_ice)

        # 4) Toxicity / volatility regime feature:
        tox_raw = 0.0
        if self.flow_vol_short > eps:
            tox_raw = abs(self.flow_sum_short) / self.flow_vol_short

        sig = self.sigma if self.sigma > 1e-8 else 1e-8
        tox_scaled = tox_raw / sig
        # moderate / declining toxicity is favourable to mean reversion
        f_tox = -math.tanh(0.1 * (tox_scaled - 1.0))

        return [f_resid, f_flow, f_ice, f_tox]

    def _update_feature_weights(self, bar_return: float) -> None:
        """
        Online correlation-based weight update:

        We use s_{t-1} and r_t to update correlations. Then we map
        correlations to weights in [-max_weight, max_weight].
        """
        if not self.have_prev_features or self.prev_features is None:
            return

        # Update correlations for each feature
        for i, f_prev in enumerate(self.prev_features):
            self.feature_stats[i].update(f_prev, bar_return)

        self.feature_update_ctr += 1
        if self.feature_update_ctr < self.feature_update_skip:
            return
        self.feature_update_ctr = 0

        # Re-compute weights from correlations
        for i, stat in enumerate(self.feature_stats):
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
        Compute alpha score A_t and decide signal sign based on adaptive threshold.
        Returns -1, 0, +1.
        """
        # linear combo
        A = 0.0
        for w, f in zip(self.feature_weights, features):
            A += w * f
        self.last_alpha = A
        self.last_alpha_ts = ts

        # update absolute alpha statistics
        abs_A = abs(A)
        self.abs_alpha_stats.update(abs_A)
        mean_abs, std_abs = self.abs_alpha_stats.stats()

        # need some history for stable threshold
        if self.abs_alpha_stats.n < 100:
            self.prev_features = features
            self.have_prev_features = True
            return 0

        # adaptive threshold: mean(|A|) + k * std(|A|)
        k_thr = 1.5
        thr = mean_abs + k_thr * std_abs
        if thr <= 0.0:
            thr = mean_abs if mean_abs > 0.0 else 0.0

        sig = 0
        if abs_A >= thr and A != 0.0:
            sig = 1 if A > 0.0 else -1

        self.prev_features = features
        self.have_prev_features = True
        return sig

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def update_batch(self, rows: RowIter):
        """
        High-Performance Vectorized Update.

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

                # Iceberg score
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
