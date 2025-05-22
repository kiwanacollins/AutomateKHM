# Real Data Usage Guide

This guide explains how to use the departure status automation with your actual Excel files.

## Quick Start

1. **Place your Excel files in this folder**
   - Your departure status Excel file 
   - Your departure history Excel file

2. **Update the configuration file**
   ```bash
   nano config.sh
   ```
   - Change STATUS_FILE to your actual status file name
   - Change HISTORY_FILE to your actual history file name

3. **Run the automation**
   ```bash
   ./run_with_real_data.sh
   ```

4. **Check the results**
   - The script will create the following folders automatically:
     - `output` directory containing:
       1. An updated departure status Excel file
       2. A name mapping report Excel file 
     - `logs` directory containing:
       3. A detailed log file with timestamp

## File Requirements

### Departure Status File
Must have these columns:
- S/N
- First_name
- Last_name
- PASSPORT NO
- Departure_status
- Travel_date
- Arrival_date

### Departure History File
Must have these columns:
- S/N
- CANDIDATE NAME (full name in one column)
- PASSPORT NO
- PHN NO
- CONTRACT

## Output Files

After running the script, you'll find:

1. **Updated departure status file**: `updated_departure_status_YYYY-MM-DD.xlsx`
   - Contains your original status data with updates
   - People in both files will have status updated to "TRAVELLED"
   - People only in history will be added as new entries

2. **Name mapping report**: `name_mapping_report_YYYY-MM-DD.xlsx`
   - Shows how people were matched between files
   - Highlights who is in history but not in status
   - Includes a count of missing people

3. **Log file**: In the `logs` directory
   - Detailed information about the process
   - Statistics about matches and new entries

## Viewing Logs

You can view the log files in the `logs` directory:
```bash
less logs/departure_automation_YYYY-MM-DD_HH-MM-SS.log
```

## Troubleshooting

If you encounter issues:
1. Check the log file for specific error messages
2. Verify your Excel files have the correct column names
3. Make sure both files are not open in Excel when running the script

For detailed documentation, see the main README.md file.
