# --- SwingEMA_RSI_ATR.py
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


class SwingEMA_RSI_ATR(IStrategy):
    """
    Swing trading baseline (1h).
    - Direction: EMA50 > EMA200 (uptrend).
    - Entry: pullback to SMA20 with RSI >= buy_min.
    - Exit: close < EMA50 OR RSI > sell_max (trend weakening/overbought).
    - Risk: initial SL = close - ATR * atr_mult; breakeven at ~0.7R; trail at TP (R:R).
    """

    INTERFACE_VERSION: int = 3
    timeframe = "1h"
    startup_candle_count = 240
    can_short = False

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_buying_expired_candle_after = 5

    # We manage exits via signals + custom stoploss (R:R). Disable ROI TP.
    minimal_roi = {"0": 0.99}
    trailing_stop = False
    use_custom_stoploss = True
    stoploss = -0.20  # hard floor, rarely hit

    # --- Hyperoptable knobs ---
    rsi_len = IntParameter(10, 18, default=14, space="buy")
    rsi_buy_min = IntParameter(40, 55, default=45, space="buy")
    rsi_sell_max = IntParameter(65, 78, default=72, space="sell")

    atr_mult = DecimalParameter(1.6, 2.6, default=2.0, decimals=2, space="stoploss")
    rr_ratio = DecimalParameter(1.6, 2.8, default=2.0, decimals=2, space="roi")

    pullback_tol_bp = IntParameter(0, 20, default=6, space="buy")  # tolerance around SMA20 in basis points

    # cache per pair for custom_stoploss
    custom_info: Dict[str, pd.DataFrame] = {}

    def informative_pairs(self):
        return []

    # ---------------- Indicators ----------------
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        if df.empty:
            return df

        for col in ("open", "high", "low", "close", "volume"):
            df[col] = _to_num(df[col])

        df["ema50"] = ta.EMA(df, timeperiod=50)
        df["ema200"] = ta.EMA(df, timeperiod=200)
        df["sma20"] = ta.SMA(df, timeperiod=20)
        df["rsi"] = ta.RSI(df, timeperiod=int(self.rsi_len.value))
        df["atr"] = ta.ATR(df, timeperiod=14)

        # initial stop ref at each bar
        df["init_sl_price"] = df["close"] - (df["atr"] * float(self.atr_mult.value))
        df["ema50"] = _to_num(df["ema50"])
        df["ema200"] = _to_num(df["ema200"])
        df["sma20"] = _to_num(df["sma20"])
        df["rsi"] = _to_num(df["rsi"])
        df["atr"] = _to_num(df["atr"])
        df["init_sl_price"] = _to_num(df["init_sl_price"])

        # store for custom_stoploss lookup
        self.custom_info[metadata["pair"]] = df[["date", "init_sl_price"]].copy().set_index("date")
        return df

    # ---------------- Entries ----------------
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["enter_long"] = 0
        df["enter_tag"] = ""

        uptrend = df["ema50"] > df["ema200"]

        # allow tiny pierce of SMA20, but require close > SMA20 (reclaim)
        tol = 1 + (int(self.pullback_tol_bp.value) / 10000.0)
        pullback = (df["low"] <= df["sma20"] * tol) & (df["close"] > df["sma20"])

        rsi_ok = df["rsi"] >= int(self.rsi_buy_min.value)

        mask = uptrend & pullback & rsi_ok & (df["volume"] > 0)
        df.loc[mask, ["enter_long", "enter_tag"]] = (1, "swing_pullback_sma20")
        return df

    # ---------------- Exits ----------------
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["exit_long"] = 0
        df["exit_tag"] = ""

        weaken = df["close"] < df["ema50"]
        overbought = df["rsi"] > int(self.rsi_sell_max.value)

        mask = (weaken | overbought) & (df["volume"] > 0)
        df.loc[mask, ["exit_long", "exit_tag"]] = (1, "swing_exit_weaken_or_hot")
        return df

    # ---------------- Custom Stoploss: ATR + R:R ----------------
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

        # base = initial SL projected to current price (negative ratio)
        result = (init_sl_abs / current_rate) - 1.0

        # move to breakeven (incl. fees) once ~0.7R reached
        if current_rate >= be_trigger:
            be_abs = trade.open_rate * (1 + trade.fee_open + trade.fee_close)
            result = max(result, (be_abs / current_rate) - 1.0)

        # lock some gains at >=1.5R
        if current_rate >= lock_trigger:
            lock_abs = trade.open_rate + 0.5 * risk_abs
            result = max(result, (lock_abs / current_rate) - 1.0)

        # at TP, trail at TP line
        if current_rate >= tp_abs:
            tp_ratio = (tp_abs / current_rate) - 1.0
            result = max(result, tp_ratio)

        # never looser than hard floor
        return max(result, self.stoploss)

    # Optional plotting
    plot_config = {
        "main_plot": {
            "ema50": {"color": "blue"},
            "ema200": {"color": "teal"},
            "sma20": {"color": "orange"},
        },
        "subplots": {
            "RSI": {"rsi": {"color": "purple"}},
            "ATR": {"atr": {"color": "red"}},
        },
    }