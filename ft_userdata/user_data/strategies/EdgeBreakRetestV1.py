# --- EdgeBreakRetestV1.py (fixed assignment) ---
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade

import talib.abstract as ta


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _crossed_above(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) <= b.shift(1)) & (a > b)


def _crossed_below(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) >= b.shift(1)) & (a < b)


class EdgeBreakRetestV1(IStrategy):
    INTERFACE_VERSION: int = 3
    timeframe = "5m"
    startup_candle_count = 200
    can_short = False

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

    minimal_roi = {"0": 0.99}      # disable ROI TP; we use R:R + signals
    trailing_stop = False
    use_custom_stoploss = True
    stoploss = -0.20               # hard floor only

    # Hyperoptables
    adx_trend_th = IntParameter(16, 26, default=20, space="buy")
    rsi_len = IntParameter(10, 18, default=14, space="buy")
    mr_rsi_max = IntParameter(32, 42, default=36, space="buy")
    bb_stds = DecimalParameter(1.8, 2.4, default=2.0, decimals=1, space="buy")
    ema_pullback_window = IntParameter(1, 3, default=1, space="buy")
    atr_mult = DecimalParameter(1.4, 2.4, default=1.8, decimals=2, space="stoploss")
    rr_ratio = DecimalParameter(1.6, 3.0, default=2.2, decimals=2, space="roi")

    custom_info: Dict[str, pd.DataFrame] = {}

    def informative_pairs(self):
        return []

    # -------- Indicators --------
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        if df.empty:
            return df

        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = _to_num(df[col])

        df["ema20"] = ta.EMA(df, timeperiod=20)
        df["ema50"] = ta.EMA(df, timeperiod=50)
        df["ema100"] = ta.EMA(df, timeperiod=100)
        df["rsi"] = ta.RSI(df, timeperiod=int(self.rsi_len.value))
        df["adx14"] = ta.ADX(df, timeperiod=14)

        stds = float(self.bb_stds.value)
        ub, mb, lb = ta.BBANDS(df["close"], timeperiod=20, nbdevup=stds, nbdevdn=stds, matype=0)
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = ub, mb, lb

        df["atr"] = ta.ATR(df, timeperiod=14)
        df["init_sl_price"] = df["close"] - (df["atr"] * float(self.atr_mult.value))

        # numeric safety
        for col in ["ema20","ema50","ema100","rsi","adx14","bb_upper","bb_mid","bb_lower","atr","init_sl_price"]:
            df[col] = _to_num(df[col])

        # store per-pair ref
        self.custom_info[metadata["pair"]] = (
            df[["date", "init_sl_price"]].copy().set_index("date")
        )
        return df

    # -------- Entries --------
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["enter_long"] = 0
        df["enter_tag"] = ""

        adx_th = int(self.adx_trend_th.value)
        trend_regime = df["adx14"] >= adx_th
        range_regime = df["adx14"] < adx_th

        ema_stack = (df["ema20"] > df["ema50"]) & (df["ema50"] > df["ema100"])
        pull_allow = 1 + (0.001 * int(self.ema_pullback_window.value))
        pullback_tag = df["low"] <= df["ema20"] * pull_allow
        reclaim = df["close"] > df["ema20"]
        trend_entry = trend_regime & ema_stack & pullback_tag & reclaim

        mr_entry = range_regime & (df["close"] <= df["bb_lower"]) & (df["rsi"] <= int(self.mr_rsi_max.value))

        # assign separately (avoid tuple/ndim issue)
        df.loc[trend_entry & (df["volume"] > 0), "enter_long"] = 1
        df.loc[trend_entry & (df["volume"] > 0), "enter_tag"] = "trend_pullback"

        df.loc[mr_entry & (df["volume"] > 0), "enter_long"] = 1
        df.loc[mr_entry & (df["volume"] > 0), "enter_tag"] = "mr_dip"

        return df

    # -------- Exits --------
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["exit_long"] = 0
        df["exit_tag"] = ""

        ema_cross_down = _crossed_below(df["ema20"], df["ema50"])
        lose_fast = df["close"] < df["ema20"]
        mr_complete = df["close"] >= df["bb_mid"]
        rsi_hot = df["rsi"] >= 72

        rule = (ema_cross_down | lose_fast | mr_complete | rsi_hot) & (df["volume"] > 0)

        df.loc[rule, "exit_long"] = 1
        df.loc[rule, "exit_tag"] = "rule_exit"
        return df

    # -------- Custom Stoploss (R:R) --------
    def _lookup_init_sl(self, pair: str, open_time: datetime) -> Optional[float]:
        hist = self.custom_info.get(pair)
        if hist is None or hist.empty:
            return None
        row = hist.loc[:open_time].tail(1)
        if row.empty:
            return None
        try:
            return float(row["init_sl_price"].iloc[0])
        except Exception:
            return None

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        init_sl_abs = self._lookup_init_sl(pair, trade.open_date_utc)
        if init_sl_abs is None or current_rate <= 0:
            return self.stoploss

        risk_abs = max(0.0, trade.open_rate - init_sl_abs)
        if risk_abs <= 0:
            return self.stoploss

        rr = float(self.rr_ratio.value)
        tp_abs = trade.open_rate + rr * risk_abs
        be_trigger = trade.open_rate + 0.7 * risk_abs
        lock_trigger = trade.open_rate + 1.5 * risk_abs

        result = (init_sl_abs / current_rate) - 1.0

        if current_rate >= be_trigger:
            be_abs = trade.open_rate * (1 + trade.fee_open + trade.fee_close)
            result = max(result, (be_abs / current_rate) - 1.0)

        if current_rate >= lock_trigger:
            lock_abs = trade.open_rate + 0.4 * risk_abs
            result = max(result, (lock_abs / current_rate) - 1.0)

        if current_rate >= tp_abs:
            tp_ratio = (tp_abs / current_rate) - 1.0
            result = max(result, tp_ratio)

        return max(result, self.stoploss)

    plot_config = {
        "main_plot": {
            "ema20": {"color": "orange"},
            "ema50": {"color": "blue"},
            "ema100": {"color": "teal"},
            "bb_upper": {"color": "grey"},
            "bb_mid": {"color": "lightblue"},
            "bb_lower": {"color": "grey"},
        },
        "subplots": {
            "RSI": {"rsi": {"color": "purple"}},
            "ADX": {"adx14": {"color": "black"}},
            "ATR": {"atr": {"color": "red"}},
        },
    }