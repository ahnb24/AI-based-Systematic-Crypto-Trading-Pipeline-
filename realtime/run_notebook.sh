#!/bin/bash

while true; do
    echo "Starting notebook..."
    python -m jupyter nbconvert --execute --to notebook realtime.ipynb
    echo "Notebook exited. Restarting in 10 seconds..."
    sleep 10
done
