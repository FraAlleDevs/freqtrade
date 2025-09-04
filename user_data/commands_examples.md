docker compose run --rm freqtrade backtesting \
  -c user_data/strategies/StrategyMD001_backtest.json \
  -s CustomStoplossWithPSAR \
  --timeframe 15m \
  --timerange 20240717-20250816 \
  --cache none \                  # avoid reusing stale cached results
  --export trades \               # export trades.json (good for slicing later)
  --export signals \              # export signals/exits pkl (needed for analysis groups)
  --export-filename user_data/backtest_results/bt_md001


docker compose run --rm freqtrade backtesting-show \                                                                     (matthias-dev)freqtrade
  -c user_data/strategies/StrategyMD001_backtest.json \
  --backtest-filename user_data/backtest_results/backtest-result-2025-09-03_19-39-46.zip \
  --breakdown week

docker compose run --rm freqtrade backtesting-analysis \
  -c user_data/strategies/StrategyMD001_backtest.json \
  --backtest-filename user_data/backtest_results/backtest-result-2025-09-03_19-39-46.zip \
  --analysis-groups 0 1 2 3 4 5 \
  -vvv


point on latest backtest ouptup:
LATEST=$(ls -t1 user_data/backtest_results/bt_md001-*.zip | head -n1)