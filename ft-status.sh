#!/bin/bash
echo "=== Docker Containers ==="
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(NAMES|freqtrade)"
echo ""
echo "=== Recent Logs ==="
if docker ps --format "{{.Names}}" | grep -q "^freqtrade$"; then
    docker logs freqtrade --tail 5
else
    echo "Freqtrade container is not running. Use ./ft-start.sh to start services."
fi
