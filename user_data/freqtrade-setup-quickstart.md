# Freqtrade Quickstart Guide

This guide summarizes the essential steps to set up Freqtrade and run your first backtest.

---

## 1. Environment Setup

```bash
# Clone the Freqtrade repository (if not already done)
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade

# Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

---

## 2. Install Freqtrade and Dependencies

```bash
# Install Freqtrade (from source)
pip install -e .

# Install TA-Lib (required for indicators)
# On macOS:
brew install ta-lib
pip install TA-Lib

# On Linux:
sudo apt-get install -y build-essential
git clone https://github.com/ta-lib/ta-lib.git
cd ta-lib && ./configure --prefix=/usr && make && sudo make install
cd ..
pip install TA-Lib
```

---

## 3. Download Historical Data

```bash
# Download spot data (example: BTC/USDT, 1h timeframe)
freqtrade download-data --exchange binance --pairs BTC/USDT --timeframe 1h

# Download futures data (for short selling)
freqtrade download-data --exchange binance --pairs BTC/USDT:USDT --timeframe 1h
```

---

## 4. Create a Configuration File

```bash
# Create a new config file (interactive)
freqtrade new-config --config user_data/config_btc.json

# Or copy and edit an example config
cp config_examples/config_binance.example.json user_data/config_btc.json
# Edit user_data/config_btc.json as needed (API keys, pairs, stake amount, etc.)
```

---

## 5. Create or Add a Strategy

- Place your strategy file in `user_data/strategies/` (e.g., `SimpleRSIStrategy.py`).
- Example strategies can be found in the [Freqtrade documentation](https://www.freqtrade.io/en/stable/strategy-customization/).

---

## 6. List Available Strategies

```bash
freqtrade list-strategies
```

---

## 7. Run Your First Backtest

```bash
freqtrade backtesting --config user_data/config_btc.json --strategy SimpleRSIStrategy
```

- Results will be saved in `user_data/backtest_results/`.

---

## 8. Analyze Results (Optional)

- Use Jupyter Lab for in-depth analysis:

```bash
jupyter lab
```
- Open `user_data/notebooks/strategy_analysis.ipynb` for custom analysis and plots.

---

## Tips
- Always start with small amounts and dry-run mode.
- Review [Freqtrade Documentation](https://www.freqtrade.io) for advanced features.
- Use `freqtrade hyperopt` to optimize your strategy parameters.

---

**Happy Trading!** 