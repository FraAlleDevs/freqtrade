#!/bin/bash
cd ~/freqtrade

if [ -z "$1" ]; then
    echo "Usage: $0 <strategy_name>"
    echo "Example: $0 BtcEurActiveStrategy"
    exit 1
fi

docker compose run --rm freqtrade backtesting --config /freqtrade/user_data/config.json --strategy "$1"
