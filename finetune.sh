#!/bin/bash

# Sparse Autoencoder Finetuning Script
# Loads pre-trained sparse_ae.pth and runs Phase 2 & 3

OUTPUT_PREFIX="sparse_finetune_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUTPUT_PREFIX}_training.log"

echo "Starting sparse autoencoder finetuning..."
echo "Log file: $LOG_FILE"
echo ""

nohup /data/b23_chiranjeevi/.venv/bin/python3 -u finetune.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "Finetuning started with PID: $PID"
echo "Monitor with: tail -f $LOG_FILE"
echo "Kill with: kill $PID"
echo ""
echo "Output will appear in the log file within seconds..."

exit 0
