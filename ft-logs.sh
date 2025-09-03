#!/bin/bash

# Get all running freqtrade containers
containers=($(docker ps --format "{{.Names}}" | grep "^freqtrade"))

if [ ${#containers[@]} -eq 0 ]; then
    echo "No Freqtrade containers are running. Use ./ft-start.sh to start services."
    exit 1
elif [ ${#containers[@]} -eq 1 ]; then
    # Single container - show logs directly
    echo "Showing logs for ${containers[0]}..."
    docker logs "${containers[0]}" --tail 100 -f
else
    # Multiple containers - show menu
    echo "Multiple Freqtrade containers found:"
    echo "0) Show logs from ALL containers (interleaved)"
    for i in "${!containers[@]}"; do
        echo "$((i+1))) ${containers[i]}"
    done
    
    read -p "Select option [0-${#containers[@]}]: " choice
    
    case $choice in
        0)
            echo "Showing logs from all containers (press Ctrl+C to stop)..."
            # Show logs from all containers with container name prefix
            for container in "${containers[@]}"; do
                docker logs "$container" --tail 50 -f --timestamps 2>&1 | sed "s/^/[$container] /" &
            done
            wait
            ;;
        [1-9]*)
            if [ "$choice" -gt 0 ] && [ "$choice" -le "${#containers[@]}" ]; then
                selected_container="${containers[$((choice-1))]}"
                echo "Showing logs for $selected_container..."
                docker logs "$selected_container" --tail 100 -f
            else
                echo "Invalid selection."
                exit 1
            fi
            ;;
        *)
            echo "Invalid selection."
            exit 1
            ;;
    esac
fi
