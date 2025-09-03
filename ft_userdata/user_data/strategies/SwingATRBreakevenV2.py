# --- SwingATRBreakevenV2.py
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import talib.abstract as ta

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade


class SwingATRBreakevenV2(IStrategy):
    """
    Swing strategy with:
      - Volatility-adaptive initial SL from ATR at entry (default 1.5 * ATR)
      - Breakeven once profit >= +1.5%
      - ATR trailing only once profit >= +3.0%
      - Trend filter via EMAs + RSI pullback recovery

    Long-only. Exits are primarily controlled by custom_stoploss().
    """

    INTERFACE_VERSION: int = 3

    # --------- Core settings
    timeframe = "1h"
    can_short = False
    process_only_new_candles = True

    # Let custom stoploss manage exits. Keep ROI out of the way.
    minimal_roi = {"0": 0.99}
    stoploss = -0.10
    use_custom_stoploss = True
    trailing_stop = False

    # We’ll emit exit signals (secondary), but SL is primary.
    use_exit_signal = True
    exit_profit_only = False
    ignore_buying_expired_candle_after = 2

    # --------- Hyperopt-able knobs (sensible defaults)
    ema_fast_len = IntParameter(18, 30, default=20, space="buy")
    ema_slow_len = IntParameter(45, 80, default=50, space="buy")
    ema_trend_len = IntParameter(180, 250, default=200, space="buy")
    rsi_len = IntParameter(10, 20, default=14, space="buy")
    rsi_pullback_recover = IntParameter(42, 55, default=48, space="buy")  # RSI must cross back above this

    atr_len = IntParameter(10, 21, default=14, space="buy")
    atr_init_mult = DecimalParameter(1.0, 2.5, decimals=2, default=1.50, space="buy")
    atr_trail_mult = DecimalParameter(0.7, 2.0, decimals=2, default=1.00, space="sell")

    breakeven_trigger = DecimalParameter(0.01, 0.03, decimals=3, default=0.015, space="sell")  # +1.5%
    trail_trigger = DecimalParameter(0.02, 0.05, decimals=3, default=0.030, space="sell")      # +3.0%

    # Keep ATR/SL upper bounds sane
    max_init_sl = DecimalParameter(0.01, 0.05, decimals=3, default=0.035, space="sell")  # cap initial SL at 3.5%
    max_trail_sl = DecimalParameter(0.01, 0.08, decimals=3, default=0.05, space="sell")  # cap trail at 5%

    # Will store per-pair informative data (ATR at timestamps)
    custom_info: Dict[str, pd.DataFrame] = {}

    def informative_pairs(self):
        return []

    # ------------- Indicators
    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if df.empty:
            return df

        # EMAs & trend slope
        fast = int(self.ema_fast_len.value)
        slow = int(self.ema_slow_len.value)
        trend = int(self.ema_trend_len.value)

        df["ema_fast"] = ta.EMA(df, timeperiod=fast)
        df["ema_slow"] = ta.EMA(df, timeperiod=slow)
        df["ema_trend"] = ta.EMA(df, timeperiod=trend)
        # slope proxy: EMA now vs EMA N bars ago
        slope_lookback = 3
        df["ema_trend_slope"] = df["ema_trend"] - df["ema_trend"].shift(slope_lookback)

        # Momentum
        rsi_len = int(self.rsi_len.value)
        df["rsi"] = ta.RSI(df, timeperiod=rsi_len)

        # ATR for volatility
        atr_len = int(self.atr_len.value)
        df["atr"] = ta.ATR(df, timeperiod=atr_len)

        # Volume guard
        df["vol_mean_20"] = ta.SMA(df["volume"], timeperiod=20)

        # Keep what we need for custom_stoploss
        # NOTE: 'date' column exists in Freqtrade dataframes
        self.custom_info[metadata["pair"]] = df[["date", "close", "atr"]].copy().set_index("date")

        return df

    # ------------- Entry logic (conservative swing long)
    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df["enter_long"] = 0
        df["enter_tag"] = ""

        conds: List = [
            (df["ema_slow"] > df["ema_trend"]),        # structure: EMA50 above EMA200
            (df["ema_trend_slope"] > 0),               # EMA200 rising
            (df["close"] > df["ema_slow"]),            # price above EMA50 (stay with trend)
            (df["rsi"].shift(1) < int(self.rsi_pullback_recover.value)),  # was below threshold (pullback)
            (df["rsi"] > int(self.rsi_pullback_recover.value)),           # and recovered above it
            (df["vol_mean_20"] > 0),
        ]
        if conds:
            df.loc[np.all(conds, axis=0), ["enter_long", "enter_tag"]] = (1, "trend_pullback_recover")

        return df

    # ------------- Exit signals (secondary to custom SL)
    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df["exit_long"] = 0
        df["exit_tag"] = ""

        # Optional signal exit: momentum overheated + close < EMA fast (weakness after pop)
        exit_conds: List = [
            (df["rsi"] > 70),
            (df["close"] < df["ema_fast"]),
            (df["vol_mean_20"] > 0),
        ]
        if exit_conds:
            df.loc[np.all(exit_conds, axis=0), ["exit_long", "exit_tag"]] = (1, "rsi_hot_weak_close")

        return df

    # ------------- Custom Stoploss (ATR initial, breakeven, ATR trailing)
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Return a negative ratio relative to current_rate (e.g., -0.02 = 2% below current price).
        Logic:
          1) Initial SL at entry = atr_init_mult * ATR(entry_time) below entry
          2) If profit >= breakeven_trigger -> move SL to (entry + fees)  (≈ breakeven)
          3) If profit >= trail_trigger -> trail by atr_trail_mult * ATR(current_time)
          4) Never loosen below the initial SL; never below self.stoploss
          5) Cap the SL distance by max_init_sl / max_trail_sl
        """
        # Safety
        if pair not in self.custom_info:
            return self.stoploss

        info_df = self.custom_info[pair]
        if info_df.empty:
            return self.stoploss

        # Get ATR at (or before) trade open time
        try:
            open_idx = info_df.index.get_loc(trade.open_date_utc, method="ffill")
            atr_at_open = float(info_df.iloc[open_idx]["atr"])
        except Exception:
            atr_at_open = float(info_df["atr"].iloc[-1])

        # Current ATR
        try:
            curr_idx = info_df.index.get_loc(current_time, method="ffill")
            atr_now = float(info_df.iloc[curr_idx]["atr"])
        except Exception:
            atr_now = float(info_df["atr"].iloc[-1])

        # --- 1) Initial ATR-based stop as fraction of entry
        init_mult = float(self.atr_init_mult.value)
        max_init = float(self.max_init_sl.value)     # cap initial SL (e.g., 3.5%)
        init_sl_frac = atr_at_open / max(trade.open_rate, 1e-9) * init_mult
        init_sl_frac = min(init_sl_frac, max_init)
        sl_candidates = [ -init_sl_frac ]            # negative

        # --- 2) Breakeven once profit crosses threshold
        be_trig = float(self.breakeven_trigger.value)
        if current_profit is not None and current_profit >= be_trig:
            # Set SL to ~entry incl. fees (no net loss)
            be_price = trade.open_rate * (1 + trade.fee_open + trade.fee_close)
            be_ratio = (be_price / max(current_rate, 1e-9)) - 1.0   # negative small magnitude
            sl_candidates.append( float(be_ratio) )

        # --- 3) ATR trailing once profit >= trail_trigger
        trail_trig = float(self.trail_trigger.value)
        if current_profit is not None and current_profit >= trail_trig:
            trail_mult = float(self.atr_trail_mult.value)
            max_trail = float(self.max_trail_sl.value)
            trail_frac = atr_now / max(current_rate, 1e-9) * trail_mult
            trail_frac = min(max(0.004, trail_frac), max_trail)     # 0.4%..cap
            sl_candidates.append( -trail_frac )

        # --- Final SL = tightest (closest to 0), but respect hard floor
        sl = max(sl_candidates)              # e.g., max(-0.035, -0.015, -0.01) = -0.01 (tightest)
        sl = max(sl, float(self.stoploss))   # do not exceed absolute floor

        return float(sl)

    # Optional plotting for `freqtrade plot-dataframe`
    plot_config = {
        "main_plot": {
            "ema_fast": {"color": "blue"},
            "ema_slow": {"color": "orange"},
            "ema_trend": {"color": "green"},
        },
        "subplots": {
            "RSI": {"rsi": {"color": "purple"}},
            "ATR": {"atr": {"color": "black"}},
        },
    }