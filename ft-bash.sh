#!/bin/bash
if docker ps --format "{{.Names}}" | grep -q "^freqtrade$"; then
    docker exec -it freqtrade bash
else
    echo "Freqtrade container is not running. Use ./ft-start.sh to start services."
fi
