#!/bin/bash

# Sparse Autoencoder Training Script
# Runs training in background with nohup (unbuffered output)

OUTPUT_PREFIX="sparse_ae_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUTPUT_PREFIX}_training.log"

echo "Starting sparse autoencoder training..."
echo "Log file: $LOG_FILE"
echo ""

# Use -u flag to make Python unbuffered so logs appear immediately
nohup /data/b23_chiranjeevi/.venv/bin/python3 -u train.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "Training started with PID: $PID"
echo "Monitor with: tail -f $LOG_FILE"
echo "Kill with: kill $PID"
echo ""
echo "Output will appear in the log file within seconds..."

exit 0
