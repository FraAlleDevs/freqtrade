# --- FixedRiskRewardLoss.py
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class FixedRiskRewardLossImproved(IStrategy):
    """
    Trend-following entries (pullback buys) + ATR-based initial SL + fixed R:R trailing via custom_stoploss.
    Live-realistic (no stacking). Works on 5m/15m.
    """

    INTERFACE_VERSION: int = 3

    timeframe = "5m"
    startup_candle_count = 120

    # We let custom_stoploss do the heavy lifting.
    # Keep a conservative hard floor as last resort.
    stoploss = -0.18
    use_custom_stoploss = True

    # Remove time-based forced exits for now (let signals + R:R manage)
    minimal_roi = {"0": 0.99}  # essentially disabled

    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_buying_expired_candle_after = 5

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # --- R:R settings ---
    rr_ratio: float = 2.0          # TP at 2R (more achievable on 5m)
    break_even_R: float = 0.8      # move SL to breakeven around 0.8R
    atr_mult: float = 2.2          # initial SL distance = ATR * 2.2

    rsi_len: int = 14
    rsi_overbought: int = 72       # exit filter

    custom_info: Dict[str, pd.DataFrame] = {}

    def informative_pairs(self):
        return []

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["ema20"] = ta.EMA(df, timeperiod=20)
        df["ema50"] = ta.EMA(df, timeperiod=50)
        df["ema100"] = ta.EMA(df, timeperiod=100)
        df["rsi"] = ta.RSI(df, timeperiod=self.rsi_len)

        ha = qtpylib.heikinashi(df)
        df["ha_open"] = ha["open"]
        df["ha_close"] = ha["close"]

        # Bollinger for pullback context
        ub, mb, lb = ta.BBANDS(df["close"], timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = ub, mb, lb

        df["atr"] = ta.ATR(df, timeperiod=14)
        df["stoploss_rate"] = df["close"] - (df["atr"] * self.atr_mult)

        # store per-pair refs for custom SL
        self.custom_info[metadata["pair"]] = df[["date", "stoploss_rate"]].copy().set_index("date")
        return df

    def _strong_trend(self, df: DataFrame) -> pd.Series:
        return (df["ema20"] > df["ema50"]) & (df["ema50"] > df["ema100"])

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["enter_long"] = 0
        df["enter_tag"] = ""

        strong = self._strong_trend(df)
        ha_green = df["ha_close"] > df["ha_open"]

        # Pullback types:
        pullback_ema20 = (df["low"] <= df["ema20"]) & (df["close"] > df["ema20"])
        pullback_bblow = (df["close"] <= df["bb_lower"])

        mask = (strong & ha_green & (pullback_ema20 | pullback_bblow) & (df["volume"] > 0))
        df.loc[mask, ["enter_long", "enter_tag"]] = (1, "pullback_entry")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["exit_long"] = 0
        df["exit_tag"] = ""

        # Exit if fast trend weakens OR overbought cool-off
        ema_cross_down = qtpylib.crossed_below(df["ema20"], df["ema50"])
        rsi_hot = df["rsi"] > self.rsi_overbought
        price_below_ema20 = df["close"] < df["ema20"]

        exit_mask = (ema_cross_down | rsi_hot | price_below_ema20) & (df["volume"] > 0)
        df.loc[exit_mask, ["exit_long", "exit_tag"]] = (1, "trend_weaken_or_rsi_hot")
        return df

    # ------- custom stoploss with R:R trailing -------
    def _lookup_open_sl_abs(self, pair: str, open_time: datetime) -> Optional[float]:
        hist = self.custom_info.get(pair)
        if hist is None or hist.empty:
            return None
        row = hist.loc[:open_time].tail(1)
        if row.empty:
            return None
        return float(row["stoploss_rate"].iloc[0])

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        init_sl_abs = self._lookup_open_sl_abs(pair, trade.open_date_utc)
        if init_sl_abs is None or current_rate <= 0:
            return self.stoploss

        risk_abs = max(0.0, trade.open_rate - init_sl_abs)
        if risk_abs <= 0:
            return self.stoploss

        tp_abs = trade.open_rate + self.rr_ratio * risk_abs
        be_abs = trade.open_rate * (1 + trade.fee_open + trade.fee_close)
        be_trigger_abs = trade.open_rate + self.break_even_R * risk_abs

        # default = initial SL projected to current price
        result = (init_sl_abs / current_rate) - 1.0

        # move to breakeven after ~0.8R
        if current_rate >= be_trigger_abs:
            result = max(result, (be_abs / current_rate) - 1.0)

        # once >= TP line, trail at TP line
        if current_rate >= tp_abs:
            result = max(result, (tp_abs / current_rate) - 1.0)

        return max(result, self.stoploss)