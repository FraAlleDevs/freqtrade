# --- Sma20RsiBbAtrStrategy.py
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade

# Try TA-Lib first (fast, native); fallback to pandas/numpy if not available.
try:
    import talib.abstract as ta
    TA_AVAILABLE = True
except Exception:
    TA_AVAILABLE = False

# --------------------------- Fallback helpers (no TA-Lib) ---------------------------
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # neutral until enough data

def _bbands(close: pd.Series, window: int = 20, stds: float = 2.0):
    mid = _sma(close, window)
    std = close.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + stds * std
    lower = mid - stds * std
    return upper, mid, lower

def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
# -----------------------------------------------------------------------------------


class Sma20RsiBbAtrStrategy(IStrategy):
    """
    RSI + SMA20 + Bollinger Bands with ATR-adaptive stop-loss.

    Long-only:
      - Enter when RSI is oversold (<30) AND price > SMA20 (uptrend filter)
      - Exit when RSI is overbought (>70) AND price < SMA20 (trend weakening)
      - Custom stoploss adapts to ATR (wider in wild markets, tighter in calm)
    """

    timeframe = "15m"
    startup_candle_count = 50
    can_short = False

    minimal_roi = {"0": 0.05}  # let custom stoploss do most of the management
    stoploss = -0.10           # absolute worst-case floor

    use_custom_stoploss = True
    trailing_stop = False

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_buying_expired_candle_after = 5

    # Hyperopt-able knobs (optional)
    rsi_length = IntParameter(8, 21, default=14, space="buy")
    rsi_buy = IntParameter(20, 35, default=30, space="buy")
    rsi_sell = IntParameter(65, 85, default=70, space="sell")
    bb_window = IntParameter(18, 22, default=20, space="buy")
    bb_stds = DecimalParameter(1.8, 2.4, default=2.0, decimals=1, space="buy")

    def informative_pairs(self):
        return []

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if df.empty:
            return df

        # --- Momentum: RSI
        if TA_AVAILABLE:
            df["rsi"] = ta.RSI(df, timeperiod=int(self.rsi_length.value))
        else:
            df["rsi"] = _rsi(df["close"], int(self.rsi_length.value))

        # --- Trend: SMA20
        if TA_AVAILABLE:
            df["sma20"] = ta.SMA(df, timeperiod=20)
        else:
            df["sma20"] = _sma(df["close"], 20)

        # --- Volatility: Bollinger Bands
        if TA_AVAILABLE:
            upper, middle, lower = ta.BBANDS(
                df["close"],
                timeperiod=int(self.bb_window.value),
                nbdevup=float(self.bb_stds.value),
                nbdevdn=float(self.bb_stds.value),
                matype=0
            )
        else:
            upper, middle, lower = _bbands(
                df["close"],
                window=int(self.bb_window.value),
                stds=float(self.bb_stds.value)
            )
        df["bb_upperband"] = upper
        df["bb_middleband"] = middle
        df["bb_lowerband"] = lower
        df["bb_width_pct"] = ((df["bb_upperband"] - df["bb_lowerband"]) / df["bb_middleband"]).replace(
            [np.inf, -np.inf], np.nan
        )

        # --- Volatility: ATR(14)
        if TA_AVAILABLE:
            df["atr"] = ta.ATR(df, timeperiod=14)
        else:
            df["atr"] = _atr(df, 14)

        # --- Volume guard / diagnostic
        df["volume_mean_slow"] = df["volume"].rolling(30, min_periods=1).mean()

        return df

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df["enter_long"] = 0
        df["enter_tag"] = ""
        sma_slope_up = df["sma20"] > df["sma20"].shift(3)   # rising over ~3 candles

        c1 = (df["rsi"] < max(30, int(self.rsi_buy.value)))
        c2 = (df["rsi"].shift(1) < 30) & (df["rsi"] >= 30)
        c3 = (df["close"] <= df["bb_lowerband"])
        mask = (sma_slope_up & (c1 | c2 | c3) & (df["volume"] > 0))
        df.loc[mask, ["enter_long", "enter_tag"]] = (1, "revert_with_rising_sma")
        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df["exit_long"] = 0
        df["exit_tag"] = ""
        # slightly easier exit: either overbought OR trend weakening
        exit_mask = ((df["rsi"] > self.rsi_sell.value) | (df["close"] < df["sma20"])) & (df["volume"] > 0)
        df.loc[exit_mask, ["exit_long", "exit_tag"]] = (1, "rsi_overbought_or_below_sma")
        return df

    # ------------------------- ATR-adaptive custom stoploss -------------------------
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
        Return a negative ratio for stop below current_rate.
        - ATR% = ATR / close (latest candle)
        - Wild market (ATR% >= 2%): wider SL (~2.2 * ATR%, capped)
        - Calm market: tighter SL (~1.2 * ATR%, bounded)
        - If profit > 4%: lock ~50% of gains, but within sane limits
        - Never looser than the hard floor self.stoploss
        """
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last = df.iloc[-1]

        close = float(last["close"])
        atr_val = float(last.get("atr", 0.0)) if not np.isnan(last.get("atr", np.nan)) else 0.0
        atr_pct = atr_val / close if close > 0 else 0.0

        wild = atr_pct >= 0.02
        if wild:
            atr_based = -min(0.15, max(0.012, atr_pct * 2.2))
        else:
            atr_based = -min(0.08, max(0.008, atr_pct * 1.2))

        sl = atr_based

        if current_profit is not None and current_profit > 0.04:
            profit_trail = -max(0.015, min(0.05, current_profit * 0.5))
            sl = max(atr_based, profit_trail)  # pick the tighter (less negative)

        sl = max(sl, self.stoploss)
        return float(sl)

    # Pretty plotting for `freqtrade plot-dataframe`
    plot_config = {
        "main_plot": {
            "sma20": {"color": "orange"},
            "bb_upperband": {"color": "grey"},
            "bb_middleband": {"color": "blue"},
            "bb_lowerband": {"color": "grey"},
        },
        "subplots": {
            "RSI": {"rsi": {"color": "purple"}},
            "ATR": {"atr": {"color": "black"}},
        },
    }