#!/bin/bash
cd ~/freqtrade
docker compose run --rm freqtrade download-data --config /freqtrade/user_data/config.json --days 30 --timeframes 1h 5m
