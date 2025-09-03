
# Trading Bot Objectives

This document defines the objectives for evaluating and improving trading strategies with Freqtrade.

## Core Performance Targets

1. **Profitability**
   - Profit Factor (PF) > **1.3**
   - Sharpe Ratio > **1.0**
   - Positive expectancy per trade

2. **Annualized Performance**
   - Target CAGR: **+20% – +50%**
   - Stretch goal: **100%+ CAGR** (but must be sustainable)

3. **Risk Management**
   - Max Drawdown < **20%**
   - Single-trade stoploss between **-2% and -8%** (depending on volatility)

4. **Practical Constraints**
   - Focus on **1–3 high-liquidity pairs** initially (BTC/USDT, ETH/USDT, +1 altcoin)
   - Trade frequency: realistic, avoid overfitting
   - Only **1 open trade per pair** (as enforced by Freqtrade)

[glossary examples](./trading_glossary_examples_mathjax.md)