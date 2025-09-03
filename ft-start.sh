#!/bin/bash
# Start Freqtrade services
echo "Starting Freqtrade services..."
cd ~/freqtrade
docker compose up -d
echo "Services started. Access FreqUI at http://localhost:8080"
echo "Check status with: ./ft-status.sh"
