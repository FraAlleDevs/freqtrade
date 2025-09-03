"""
Safer variant of SwingHighToSky:
- Keeps the original CCI/RSI swing entries/exits (mean-reversion).
- Adds ATR-capped stoploss via custom_stoploss (volatility-aware).
- Optional trend and volatility filters (off by default to preserve behavior).
"""

from freqtrade.strategy import IStrategy, IntParameter
from pandas import DataFrame
from typing import List, Dict
from datetime import datetime

import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade


class SwingHighToSkyV2(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '15m'

    # Keep original ROI/SL (hyperopt will override if present in config)
    # NOTE: We'll **tighten** risk in custom_stoploss (ATR cap), without changing ROI logic.
    minimal_roi = {"0": 0.27058, "33": 0.0853, "64": 0.04093, "244": 0}
    stoploss = -0.34338

    # Manage exits ourselves (still keep signal exits).
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    process_only_new_candles = True
    ignore_buying_expired_candle_after = 2

    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 8
        },
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 96,   # ~1 day on 15m TF
            "trade_limit": 3,                # if >=3 SL hits in window -> pause
            "stop_duration_candles": 32,     # pause ~8 hours on 15m TF
            "only_per_pair": False           # protect globally (all pairs)
        },
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 288,  # ~3 days on 15m TF
            "trade_limit": 20,               # need some trades to judge
            "stop_duration_candles": 96,     # pause ~1 day
            "max_allowed_drawdown": 0.10     # 10% equity DD threshold
        }
    ]

    # -------- Original hyperopt knobs (kept) --------
    buy_cci = IntParameter(low=-200, high=200, default=100, space='buy', optimize=True)
    buy_cciTime = IntParameter(low=10, high=80, default=20, space='buy', optimize=True)
    buy_rsi = IntParameter(low=10, high=90, default=30, space='buy', optimize=True)
    buy_rsiTime = IntParameter(low=10, high=80, default=26, space='buy', optimize=True)

    sell_cci = IntParameter(low=-200, high=200, default=100, space='sell', optimize=True)
    sell_cciTime = IntParameter(low=10, high=80, default=20, space='sell', optimize=True)
    sell_rsi = IntParameter(low=10, high=90, default=30, space='sell', optimize=True)
    sell_rsiTime = IntParameter(low=10, high=80, default=26, space='sell', optimize=True)

    # Freeze-in the known good params as defaults (hyperopt can still override)
    buy_params = {"buy_cci": -175, "buy_cciTime": 72, "buy_rsi": 90, "buy_rsiTime": 36}
    sell_params = {"sell_cci": -106, "sell_cciTime": 66, "sell_rsi": 88, "sell_rsiTime": 45}

    # -------- Risk overlay knobs (light-touch, can be hyperopted later) --------
    atr_len = 14                    # ATR length
    atr_cap_mult = 1.8              # initial ATR multiple for SL cap near entry
    max_atr_stop = 0.045            # cap SL at 4.5% (never wider than this)
    be_trigger = 0.015              # move SL to breakeven once profit >= +1.5%
    # trailing only after decent profit (optional: set >0 to enable)
    trail_trigger = 0.035           # start trailing after +3.5%
    trail_atr_mult = 1.0            # trail distance = ATR * mult (as % of price)
    max_trail = 0.05                # cap trailing width at 5%

    # Optional filters (off by default to preserve original behavior)
    enable_trend_filter = False     # require fast EMA > slow EMA
    ema_fast = 50
    ema_slow = 200

    enable_volatility_filter = False  # skip entries when BB width is tiny (chop)
    bb_len = 20
    bb_stds = 2.0
    min_bb_width_pct = 0.01          # 1% band width

    # cache for ATR/time lookups
    _cache: Dict[str, DataFrame] = {}

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Core indicators (RSI/CCI at *all* candidate periods so hyperopt can switch)
        for val in self.buy_cciTime.range:
            dataframe[f'cci-{val}'] = ta.CCI(dataframe, timeperiod=val)
        for val in self.sell_cciTime.range:
            dataframe[f'cci-sell-{val}'] = ta.CCI(dataframe, timeperiod=val)
        for val in self.buy_rsiTime.range:
            dataframe[f'rsi-{val}'] = ta.RSI(dataframe, timeperiod=val)
        for val in self.sell_rsiTime.range:
            dataframe[f'rsi-sell-{val}'] = ta.RSI(dataframe, timeperiod=val)

        # ATR for volatility-aware stops
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_len)

        # Optional simple trend filter
        if self.enable_trend_filter:
            dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast)
            dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow)

        # Optional volatility filter (Bollinger width)
        if self.enable_volatility_filter:
            bb = qtpylib.bollinger_bands(dataframe['close'], window=self.bb_len, stds=self.bb_stds)
            dataframe['bbw_pct'] = (bb['upper'] - bb['lower']) / bb['mid']  # relative width

        # cache needed for custom_stoploss
        self._cache[metadata['pair']] = dataframe[['date', 'close', 'atr']].copy().set_index('date')
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = 0

        conds: List = [
            (dataframe[f'cci-{self.buy_cciTime.value}'] < self.buy_cci.value),
            (dataframe[f'rsi-{self.buy_rsiTime.value}'] < self.buy_rsi.value),
        ]

        if self.enable_trend_filter:
            conds += [
                (dataframe['ema_fast'] > dataframe['ema_slow']),
                (dataframe['close'] > dataframe['ema_fast']),
            ]

        if self.enable_volatility_filter:
            conds += [(dataframe['bbw_pct'] > self.min_bb_width_pct)]

        if conds:
            dataframe.loc[np.all(conds, axis=0), 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        conds: List = [
            (dataframe[f'cci-sell-{self.sell_cciTime.value}'] > self.sell_cci.value),
            (dataframe[f'rsi-sell-{self.sell_rsiTime.value}'] > self.sell_rsi.value),
        ]
        if conds:
            dataframe.loc[np.all(conds, axis=0), 'exit_long'] = 1
        return dataframe

    def _atr_at(self, pair: str, when: datetime, fallback_last=True) -> float:
        df = self._cache.get(pair)
        if df is None or df.empty:
            return 0.0
        try:
            idx = df.index.get_loc(when, method="ffill")
            return float(df.iloc[idx]['atr'])
        except Exception:
            return float(df['atr'].iloc[-1]) if fallback_last else 0.0

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
        Returns a negative ratio relative to current_rate.
        Layers:
          1) ATR-capped initial SL near entry (cap: max_atr_stop)
          2) Breakeven once profit >= be_trigger
          3) ATR trailing after profit >= trail_trigger
          4) Never looser than self.stoploss
        """
        # 1) ATR-based initial cap computed at entry
        atr_open = self._atr_at(pair, trade.open_date_utc)
        if trade.open_rate <= 0 or atr_open <= 0:
            atr_cap = self.stoploss  # fallback
        else:
            atr_frac = (atr_open / trade.open_rate) * self.atr_cap_mult
            atr_frac = min(atr_frac, self.max_atr_stop)      # cap e.g. <= 4.5%
            atr_cap = -max(0.005, atr_frac)                  # at least 0.5%

        sl = atr_cap  # negative

        # 2) Breakeven
        if current_profit is not None and current_profit >= self.be_trigger:
            be_price = trade.open_rate * (1 + trade.fee_open + trade.fee_close)
            be_ratio = (be_price / max(current_rate, 1e-9)) - 1.0  # negative tiny magnitude
            sl = max(sl, float(be_ratio))  # tighten if better

        # 3) Trail after profit threshold
        if current_profit is not None and current_profit >= self.trail_trigger:
            atr_now = self._atr_at(pair, current_time)
            if atr_now > 0 and current_rate > 0:
                trail_frac = min(self.max_trail, max(0.004, (atr_now / current_rate) * self.trail_atr_mult))
                trail_sl = -trail_frac
                sl = max(sl, trail_sl)

        # 4) Respect absolute floor
        sl = max(sl, self.stoploss)
        return float(sl)

    # Optional plot overlays
    plot_config = {
        "main_plot": {},
        "subplots": {
            "ATR": {"atr": {"color": "black"}},
        },
    }