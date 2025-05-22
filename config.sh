#!/bin/bash
# Configuration file for real production data

# Set your actual Excel file names here
# IMPORTANT: Change these to your actual file names!
STATUS_FILE="your_real_status_file.xlsx"    # CHANGE THIS to your actual status file name
HISTORY_FILE="your_real_history_file.xlsx"  # CHANGE THIS to your actual history file name

# Define output directory
OUTPUT_DIR="output"

# Output files - these will be generated with date stamps
DATE=$(date +"%Y-%m-%d")
OUTPUT_FILE="${OUTPUT_DIR}/updated_departure_status_${DATE}.xlsx"
MAPPING_REPORT="${OUTPUT_DIR}/name_mapping_report_${DATE}.xlsx"
