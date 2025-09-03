# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime

from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class CustomStoplossWithPSAR(IStrategy):
    """
    Instrumented + tightened PSAR strategy for analysis:
    - Stricter entries to avoid chop.
    - enter_tag/exit_tag populated for clear analysis.
    - Winners exit via momentum TP (RSI>70 / BB upper).
    - Losers cut via PSAR bear flip + capped custom stop.
    - Trailing stop to protect gains.
    """
    INTERFACE_VERSION: int = 3

    timeframe = '15m'
    process_only_new_candles = True

    # Base stop (hard floor). Custom stop will tighten dynamically.
    stoploss = -0.08

    # Small ROI curve as backstop to close stale trades
    # (minutes -> ROI). Keeps exits happening even without signals.
    minimal_roi = {
        "0": 0.006,     # 0.6% early TP if it spikes
        "120": 0.003,   # after 2h accept 0.3%
        "360": 0.0      # after 6h accept anything >= 0
    }

    # Trailing stop (kicks in after ~1.2% run-up, trails at 0.6%)
    trailing_stop = True
    trailing_stop_positive = 0.006
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = True

    use_custom_stoploss = True
    startup_candle_count = 200

    custom_info = {}

    # ---------- Custom Stoploss (PSAR-based, clamped) ----------
    def custom_stoploss(
        self,
        pair: str,
        trade: 'Trade',
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> float:
        """
        Trail at PSAR distance converted to relative stop.
        Clamp to [-0.08, -0.02] so it doesn't get too loose/tight.
        """
        result = 1  # off unless computed
        if self.custom_info and pair in self.custom_info and trade and self.dp:
            df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            last = df.iloc[-1].squeeze()
            sar = last.get('sar', None)
            if sar is not None and current_rate > 0:
                rel = (current_rate - sar) / current_rate - 1  # negative number
                # Clamp between -8% and -2%
                result = max(min(rel, -0.02), -0.08)
        return result

    # ---------- Indicators ----------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Trend
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        # Momentum
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atrp'] = dataframe['atr'] / dataframe['close'].replace(0, np.nan)  # ATR as % of price

        # PSAR
        dataframe['sar'] = ta.SAR(dataframe)

        # Bollinger (for exits)
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        dataframe['bb_upper'] = bb['upperband']
        dataframe['bb_middle'] = bb['middleband']
        dataframe['bb_lower'] = bb['lowerband']

        # Volume sanity
        dataframe['vol_ma20'] = qtpylib.rolling_mean(dataframe['volume'], 20)

        # Save SAR for custom_stoploss in backtest/hyperopt
        if self.dp and self.dp.runmode.value in ('backtest', 'hyperopt'):
            self.custom_info[metadata['pair']] = dataframe[['date', 'sar']].copy().set_index('date')

        return dataframe

    # ---------- Entries (with tags) ----------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Uptrend filter
        strong_up = (
            (dataframe['ema20'] > dataframe['ema50']) &
            (dataframe['ema50'] > dataframe['ema200']) &
            (dataframe['close'] > dataframe['ema50'])
        )

        # Momentum filter
        mom_ok = (dataframe['rsi'] > 45) & (dataframe['macdhist'] > 0)

        # Volatility regime (avoid dead chop / extreme spikes)
        vol_ok = (dataframe['atrp'] > 0.005) & (dataframe['atrp'] < 0.03) & (dataframe['vol_ma20'] > 0)

        # PSAR bull flip (from above→below price)
        psar_bull_flip = (
            (dataframe['sar'] < dataframe['close']) &
            (dataframe['sar'].shift(1) >= dataframe['close'].shift(1))
        )

        cond_psar_strict = psar_bull_flip & strong_up & mom_ok & vol_ok

        # RSI rebound entry - relaxed to actually trigger on 15m
        rsi_rebound = (
            (dataframe['rsi'] > 35) & (dataframe['rsi'].shift(1) <= 35) &
            (dataframe['close'] > dataframe['ema20'])
        )
        cond_rsi = rsi_rebound & strong_up & vol_ok

        # Signals + tags
        dataframe.loc[cond_psar_strict, 'enter_long'] = 1
        dataframe.loc[cond_psar_strict, 'enter_tag'] = 'psar_bull_flip_strict'

        dataframe.loc[cond_rsi, 'enter_long'] = 1
        dataframe.loc[cond_rsi, 'enter_tag'] = 'rsi_rebound_35'

        return dataframe

    # ---------- Exits (with tags) ----------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Take profit on momentum exhaustion
        rsi_tp = dataframe['rsi'] > 70
        bb_tp = dataframe['close'] >= dataframe['bb_upper']

        # PSAR bear flip - use as protective exit (we'll only apply if trade is red via custom_exit)
        psar_bear_flip = (
            (dataframe['sar'] > dataframe['close']) &
            (dataframe['sar'].shift(1) <= dataframe['close'].shift(1))
        )

        # Exit signals
        dataframe.loc[rsi_tp, 'exit_long'] = 1
        dataframe.loc[rsi_tp, 'exit_tag'] = 'rsi_overbought_70'

        dataframe.loc[bb_tp, 'exit_long'] = 1
        dataframe.loc[bb_tp, 'exit_tag'] = 'bb_upper_tp'

        # We do not set psar_bear_flip as exit_long here to avoid dumping winners too early.
        # Loser handling is done in custom_stoploss clamp + ROI backstop + trailing stop.
        return dataframe