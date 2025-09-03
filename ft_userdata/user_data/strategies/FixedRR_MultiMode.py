# --- FixedRR_MultiMode.py
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging

logger = logging.getLogger(__name__)


def _to_num(s: pd.Series) -> pd.Series:
    """Coerce a series to numeric (float), preserving index and shape."""
    return pd.to_numeric(s, errors="coerce")


class FixedRR_MultiMode(IStrategy):
    """
    Two-mode strategy with ATR-based initial SL and fixed Risk/Reward trailing via custom_stoploss.

    Modes:
      - "mr"    (Mean-Reversion intraday): range-friendly, quick exits.
      - "trend" (Trend-Pullback): strong uptrend pullbacks.

    Switch via config JSON:
      "custom": { "mode": "mr" }   or   "custom": { "mode": "trend" }
    Default is "mr" if missing/invalid.
    """

    INTERFACE_VERSION: int = 3

    timeframe = "5m"
    startup_candle_count = 200

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

    # Defaults (overridden in bot_start based on mode)
    minimal_roi = {"0": 0.99}
    stoploss = -0.18
    use_custom_stoploss = True
    trailing_stop = False

    # Risk/Reward engine (overridden per mode)
    rr_ratio: float = 1.2
    break_even_R: float = 0.6
    atr_mult: float = 1.8
    rsi_len: int = 14
    rsi_overbought: int = 72

    _mode: str = "mr"
    custom_info: Dict[str, pd.DataFrame] = {}

    # -------------------- lifecycle --------------------
    def bot_start(self, **kwargs) -> None:
        # Robust mode parsing
        mode_raw = None
        if isinstance(self.config.get("custom"), dict):
            mode_raw = self.config["custom"].get("mode", None)
        mode = (str(mode_raw).strip().lower()) if mode_raw is not None else "mr"
        self._mode = mode if mode in ("mr", "trend") else "mr"

        if self._mode == "mr":
            self.rr_ratio = 1.2
            self.break_even_R = 0.6
            self.atr_mult = 1.8
            self.stoploss = -0.08
            # 5m: force exit after 1 day
            self.minimal_roi = {"0": 0.02, "720": 0.0, "1440": -1}
            self.rsi_overbought = 72
        else:  # trend
            self.rr_ratio = 2.0
            self.break_even_R = 0.8
            self.atr_mult = 2.2
            self.stoploss = -0.18
            self.minimal_roi = {"0": 0.99}
            self.rsi_overbought = 72

        logger.info(f"[FixedRR_MultiMode] Active mode: {self._mode}")

    # -------------------- helpers --------------------
    def informative_pairs(self):
        return []

    def _strong_trend(self, df: DataFrame) -> pd.Series:
        return (df["ema20"] > df["ema50"]) & (df["ema50"] > df["ema100"])

    def _lookup_open_sl_abs(self, pair: str, open_time: datetime) -> Optional[float]:
        hist = self.custom_info.get(pair)
        if hist is None or hist.empty:
            return None
        row = hist.loc[:open_time].tail(1)
        if row.empty:
            return None
        val = row["stoploss_rate"].iloc[0]
        try:
            return float(val)
        except Exception:
            return None

    # -------------------- indicators --------------------
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Compute core indicators
        df["ema20"] = ta.EMA(df, timeperiod=20)
        df["ema50"] = ta.EMA(df, timeperiod=50)
        df["ema100"] = ta.EMA(df, timeperiod=100)
        df["ema200"] = ta.EMA(df, timeperiod=200)
        df["rsi"] = ta.RSI(df, timeperiod=self.rsi_len)

        ha = qtpylib.heikinashi(df)
        df["ha_open"] = ha["open"]
        df["ha_close"] = ha["close"]

        ub, mb, lb = ta.BBANDS(df["close"], timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = ub, mb, lb

        df["adx14"] = ta.ADX(df, timeperiod=14)

        df["atr"] = ta.ATR(df, timeperiod=14)
        df["stoploss_rate"] = df["close"] - (df["atr"] * self.atr_mult)

        # --- numeric safety (prevents TypeError with <= / >=) ---
        for col in [
            "close", "low", "volume",
            "ema20", "ema50", "ema100", "ema200",
            "rsi", "bb_upper", "bb_mid", "bb_lower",
            "adx14", "atr", "stoploss_rate", "ha_open", "ha_close"
        ]:
            if col in df.columns:
                df[col] = _to_num(df[col])

        # NaN handling (avoid accidental early signals)
        df[["ema20", "ema50", "ema100", "ema200", "rsi", "bb_upper", "bb_mid", "bb_lower", "adx14", "atr"]] = \
            df[["ema20", "ema50", "ema100", "ema200", "rsi", "bb_upper", "bb_mid", "bb_lower", "adx14", "atr"]].fillna(method="ffill")

        # Store per-pair stoploss reference
        # Freqtrade dataframe provides 'date' column
        self.custom_info[metadata["pair"]] = df[["date", "stoploss_rate"]].copy().set_index("date")
        return df

    # -------------------- entries --------------------
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["enter_long"] = 0
        df["enter_tag"] = ""

        if self._mode == "mr":
            # Range preference: low ADX (fill NaN with high to suppress early signals)
            not_trending = df["adx14"].fillna(50) < 20
            # Avoid heavy downtrend context
            context_ok = df["close"] >= (df["ema200"] * 0.98)

            # Stretched down: BB lower tag OR RSI depressed
            stretch_down = (df["close"] <= df["bb_lower"]) | (df["rsi"] <= 40)

            mask = not_trending & context_ok & stretch_down & (df["volume"] > 0)
            df.loc[mask, ["enter_long", "enter_tag"]] = (1, "mr_dip_buy")
            return df

        else:
            strong = self._strong_trend(df)
            ha_green = df["ha_close"] > df["ha_open"]
            pullback_ema20 = (df["low"] <= df["ema20"]) & (df["close"] > df["ema20"])
            pullback_bblow = (df["close"] <= df["bb_lower"])

            mask = (strong & ha_green & (pullback_ema20 | pullback_bblow) & (df["volume"] > 0))
            df.loc[mask, ["enter_long", "enter_tag"]] = (1, "trend_pullback")
            return df

    # -------------------- exits --------------------
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["exit_long"] = 0
        df["exit_tag"] = ""

        if self._mode == "mr":
            mid_revert = df["close"] >= df["bb_mid"]
            rsi_relief = df["rsi"] >= 55
            trend_bad = df["close"] < df["ema20"]
            exit_mask = (mid_revert | rsi_relief | trend_bad) & (df["volume"] > 0)
            df.loc[exit_mask, ["exit_long", "exit_tag"]] = (1, "mr_revert_or_weak")
            return df

        else:
            ema_cross_down = qtpylib.crossed_below(df["ema20"], df["ema50"])
            rsi_hot = df["rsi"] > self.rsi_overbought
            price_below_ema20 = df["close"] < df["ema20"]
            exit_mask = (ema_cross_down | rsi_hot | price_below_ema20) & (df["volume"] > 0)
            df.loc[exit_mask, ["exit_long", "exit_tag"]] = (1, "trend_weaken_or_hot")
            return df

    # -------------------- custom stoploss (fixed R:R) --------------------
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

        # risk in price units
        risk_abs = max(0.0, trade.open_rate - float(init_sl_abs))
        if risk_abs <= 0:
            return self.stoploss

        tp_abs = trade.open_rate + self.rr_ratio * risk_abs
        be_abs = trade.open_rate * (1 + trade.fee_open + trade.fee_close)
        be_trigger_abs = trade.open_rate + self.break_even_R * risk_abs

        # default = initial SL projected to current price (negative ratio)
        result = (float(init_sl_abs) / current_rate) - 1.0

        # move to breakeven after X*R
        if current_rate >= be_trigger_abs:
            result = max(result, (be_abs / current_rate) - 1.0)

        # after TP, trail at TP line
        if current_rate >= tp_abs:
            result = max(result, (tp_abs / current_rate) - 1.0)

        # never looser than hard floor
        return max(result, self.stoploss)