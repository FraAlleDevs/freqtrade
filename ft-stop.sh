#!/bin/bash
# Stop Freqtrade services
echo "Stopping Freqtrade services..."
cd ~/freqtrade
docker compose down
echo "Services stopped."
