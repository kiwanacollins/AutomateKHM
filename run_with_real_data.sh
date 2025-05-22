#!/bin/bash
# Comprehensive script to process real production data
# This script directly runs the Python automation without needing run_automation.sh

# Load configuration
source ./config.sh

# Create logs directory if it doesn't exist
mkdir -p logs

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set log file path
LOG_FILE="logs/departure_automation_$(date +"%Y-%m-%d_%H-%M-%S").log"

echo "=== Starting Departure Status Automation at $(date) ===" | tee -a "$LOG_FILE"
echo "Using files:" | tee -a "$LOG_FILE"
echo "- Status file: $STATUS_FILE" | tee -a "$LOG_FILE"
echo "- History file: $HISTORY_FILE" | tee -a "$LOG_FILE"
echo "- Output will be saved to: $OUTPUT_FILE" | tee -a "$LOG_FILE"
echo "- Name mapping report will be saved to: $MAPPING_REPORT" | tee -a "$LOG_FILE"

# Run the Python script directly with your configured files
python automate_departure_status.py \
    --input-status "$STATUS_FILE" \
    --input-history "$HISTORY_FILE" \
    --output "$OUTPUT_FILE" \
    --generate-mapping-report \
    --mapping-report-path "$MAPPING_REPORT" 2>&1 | tee -a "$LOG_FILE"

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Departure status automation completed successfully!" | tee -a "$LOG_FILE"
else
    echo "Error: Departure status automation failed!" | tee -a "$LOG_FILE"
fi

echo "=== Completed Departure Status Automation at $(date) ===" | tee -a "$LOG_FILE"

echo ""
echo "==========================="
echo "Process completed!"
echo "==========================="
echo ""
echo "Output files:"
echo "1. Updated departure status: $OUTPUT_FILE"
echo "2. Name mapping report: $MAPPING_REPORT"
echo "3. Log file: $LOG_FILE"
echo ""
echo "To see the log file, run: less $LOG_FILE"
