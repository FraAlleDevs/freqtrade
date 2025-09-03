# --- ReliableBaselineV1.py
from typing import List
import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
import talib.abstract as ta


def _crossed_above(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) <= b.shift(1)) & (a > b)


def _crossed_below(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) >= b.shift(1)) & (a < b)


class ReliableBaselineV1(IStrategy):
    """
    A reliable baseline that *will* generate trades on most crypto pairs/timeframes.
    3 entry mechanisms (trend + momentum + mean-reversion) and pragmatic exits.
    Ready for Hyperopt (parameters exposed).
    """

    INTERFACE_VERSION: int = 3

    # -------- Core setup --------
    timeframe = "5m"                  # You can override with -t 15m on CLI
    startup_candle_count = 200        # enough for BB20 and longer EMAs
    can_short = False

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_buying_expired_candle_after = 5

    # -------- Risk / ROI / Trailing --------
    stoploss = -0.06                  # 6% hard floor
    minimal_roi = {                   # time-based ROI (minutes since entry)
        "0": 0.015,                   # immediately accept +1.5%
        "360": 0.005,                 # after 6h accept +0.5%
        "1440": -1                    # force exit after 1 day (5m TF)
    }
    trailing_stop = True
    trailing_stop_positive = 0.01     # trail once +1%
    trailing_stop_positive_offset = 0.02
    trailing_only_offset = True

    # -------- Hyperopt-able params --------
    ema_fast_len = IntParameter(9, 21, default=12, space="buy")
    ema_slow_len = IntParameter(22, 60, default=26, space="buy")

    rsi_len = IntParameter(10, 20, default=14, space="buy")
    rsi_cross_level = IntParameter(25, 35, default=30, space="buy")          # for RSI cross-up
    rsi_ema_confirm = IntParameter(45, 60, default=50, space="buy")          # for EMA cross-up confirmation
    rsi_sell = IntParameter(62, 80, default=70, space="sell")

    bb_stds = DecimalParameter(1.8, 2.2, default=2.0, decimals=1, space="buy")
    bb_rsi_max_for_entry = IntParameter(30, 45, default=40, space="buy")     # allow BB dip only if RSI low

    # -------- Indicators --------
    def informative_pairs(self):
        return []

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        if df.empty:
            return df

        # Ensure numeric (extra safety on some datasets)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # EMAs
        fast = int(self.ema_fast_len.value)
        slow = int(self.ema_slow_len.value)
        df["ema_fast"] = ta.EMA(df, timeperiod=fast)
        df["ema_slow"] = ta.EMA(df, timeperiod=slow)

        # RSI
        rlen = int(self.rsi_len.value)
        df["rsi"] = ta.RSI(df, timeperiod=rlen)

        # Bollinger Bands
        stds = float(self.bb_stds.value)
        upper, middle, lower = ta.BBANDS(
            df["close"], timeperiod=20, nbdevup=stds, nbdevdn=stds, matype=0
        )
        df["bb_upperband"] = upper
        df["bb_middleband"] = middle
        df["bb_lowerband"] = lower

        return df

    # -------- Entries --------
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["enter_long"] = 0
        df["enter_tag"] = ""

        # 1) Trend entry: EMA fast crosses above EMA slow + RSI confirmation
        ema_cross_up = _crossed_above(df["ema_fast"], df["ema_slow"])
        trend_ok = ema_cross_up & (df["rsi"] > int(self.rsi_ema_confirm.value))

        # 2) Momentum entry: RSI crosses up through oversold level
        rsi_cross_up = (df["rsi"].shift(1) < int(self.rsi_cross_level.value)) & (df["rsi"] >= int(self.rsi_cross_level.value))

        # 3) Mean-reversion dip: touch lower BB with depressed RSI
        bb_dip = (df["close"] <= df["bb_lowerband"]) & (df["rsi"] <= int(self.bb_rsi_max_for_entry.value))

        entry_mask = (trend_ok | rsi_cross_up | bb_dip) & (df["volume"] > 0)

        df.loc[entry_mask, ["enter_long", "enter_tag"]] = (1, "trend_or_rsi_or_bb")
        return df

    # -------- Exits --------
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["exit_long"] = 0
        df["exit_tag"] = ""

        # Exit on trend weakening or overbought or price losing fast EMA
        ema_cross_down = _crossed_below(df["ema_fast"], df["ema_slow"])
        rsi_hot = df["rsi"] >= int(self.rsi_sell.value)
        lose_fast = df["close"] < df["ema_fast"]

        exit_mask = (ema_cross_down | rsi_hot | lose_fast) & (df["volume"] > 0)

        df.loc[exit_mask, ["exit_long", "exit_tag"]] = (1, "trend_weaken_or_hot_or_lose_fast")
        return df

    # Nice plotting for `freqtrade plot-dataframe`
    plot_config = {
        "main_plot": {
            "ema_fast": {"color": "orange"},
            "ema_slow": {"color": "blue"},
            "bb_upperband": {"color": "grey"},
            "bb_middleband": {"color": "lightblue"},
            "bb_lowerband": {"color": "grey"},
        },
        "subplots": {
            "RSI": {"rsi": {"color": "purple"}}
        },
    }