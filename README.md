# Departure Status Automation

This tool automates updating departure status information using Excel files:

1. Updates departure status from NULL to "TRAVELLED" for people in both files
2. Adds new entries to the departure status file for people only in the history file
3. Generates a mapping report showing who's missing in the status file

## For Real-World Usage

See the `REAL_DATA_GUIDE.md` file for specific instructions on using this tool with your actual data.

## Quick Start

1. Edit `config.sh` with your actual Excel file names
2. Run `./run_with_real_data.sh`
3. Check the output files:
   - Updated departure status Excel file in the `output` directory
   - Name mapping report Excel file in the `output` directory
   - Log file in the `logs` directory

## Directory Structure

All necessary directories are created automatically:

- `logs/` - Contains log files with timestamps
- `output/` - Contains all generated files (status updates and reports)

## Core Files

- `automate_departure_status.py` - Main script with the matching logic
- `name_mapping_report.py` - Module for mapping report generation
- `config.sh` - Configuration for your real data files
- `run_with_real_data.sh` - Script to run the automation with your data

## Usage with Your Actual Data

1. Rename your files to match expected names, or provide custom paths:
   - Your departure status file (with columns S/N, First_name, Last_name, PASSPORT NO, Departure_status, Travel_date, Arrival_date)
   - Your departure history file (with columns S/N, CANDIDATE NAME, PASSPORT NO, PHN NO, CONTRACT)

2. Run the script:

```bash
python automate_departure_status.py --input-status your_status_file.csv --input-history your_history_file.xlsx --output updated_file.csv
```

## Detailed Options

The script supports various command-line options for flexibility:

```bash
python automate_departure_status.py --help
```

Available options:
- `--input-status FILE` - Path to your departure status file (CSV or Excel)
- `--input-history FILE` - Path to your departure history file (CSV or Excel)
- `--output FILE` - Path to save the updated departure status file
- `--skip-validation` - Skip the validation step (not recommended)

## How the Matching Works

The script uses a sophisticated approach to match people between files:

1. **Primary Match**: Exact passport number matching
2. **Enhanced Name Matching**: Advanced name similarity detection when passport numbers aren't available, using:
   - Name normalization (handling capitalization, punctuation, etc.)
   - Token-based comparison (handling different name part orders)
   - Fuzzy string matching with SequenceMatcher
   - Initial matching for handling abbreviated names

This ensures high accuracy even when:
- Names are in different order (e.g., "JOHN SMITH" vs "SMITH JOHN")
- First name and last name are swapped between files
- Names have different formatting
- Names use abbreviations or initials
- Passport numbers are missing or inconsistent

## Name Matching Improvements

This tool now includes enhanced name matching capabilities to address the challenge of different name formats between files:

1. **Advanced Name Standardization**: Names are standardized (uppercase, removing titles, punctuation, etc.)
2. **Intelligent Name Splitting**: Properly splits full names into first and last names, handling complex cases
3. **Part-by-Part Matching**: Compares individual parts of names, not just the whole string
4. **Multi-algorithm Similarity Scoring**: Uses multiple algorithms to calculate name similarity
5. **Name Mapping Reports**: Can generate detailed reports showing how names are matched between files

### Using the Name Mapping Report

To better understand how names are matched between files, use the new mapping report feature:

```bash
python automate_departure_status.py --generate-mapping-report --mapping-report-path name_report.csv
```

This generates a CSV file showing:
- How each name from the history file maps to the status file
- The similarity score between matched names
- The type of match (Passport Match, High Confidence Match, etc.)

### Testing Name Matching

You can test the name matching functionality with:
```bash
./run_name_matching_tests.sh
```

This runs test cases against various name formats and processes the sample data files to verify matching accuracy.

## Advanced Name Matching

This tool includes sophisticated name matching capabilities to handle the challenge of different name formats between files:

1. **Problem Addressed**: 
   - In the departure status file, names are divided into first name and last name columns
   - In the departure history file, the entire name is in a single "CANDIDATE NAME" column
   - This makes it difficult to accurately match people between the two files

2. **Solution Implemented**:
   - **Name Standardization**: Names are standardized (uppercase, removing titles, punctuation)
   - **Intelligent Name Parsing**: Enhanced split_name function handles complex name patterns
   - **Multi-algorithm Matching**: Uses whole-string similarity and part-by-part matching
   - **Name Mapping Reports**: Generates detailed reports showing how names are matched

To generate a name mapping report:
```bash
python automate_departure_status.py --generate-mapping-report --mapping-report-path name_report.csv
```

## Handling NULL Values

The script properly handles various forms of NULL values in both files:
- Empty cells
- "NULL" text
- "NaN" values
- "0" values for passport numbers

## Example Workflow

1. Start with two files:
   - `departure_status.csv` with people awaiting travel approval
   - `departure_history.csv` with people who have traveled

2. Run the script:
   ```bash
   python automate_departure_status.py
   ```

3. The script will:
   - Update "MOSES MONDAY" (passport A00423291) from NULL to TRAVELLED because they appear in both files
   - Update "MBOGO HASSAN" (passport B00253946) from NULL to TRAVELLED for the same reason
   - Add new entries for "MUNONO BENJAMIN" and others who are in history but not status

4. Review the log file (`departure_automation.log`) for details on all operations performed

## Validation

The script includes a validation step that checks:
- All history records are properly reflected in the updated status file
- All status updates that should have happened did happen correctly

This ensures 100% accuracy of the automation process.

## Need Help?

Check the log file for detailed information about any issues. The script provides clear error messages and warnings to help troubleshoot any problems.

## Running the Tool with Excel or CSV Files

The tool now includes a unified script that supports both Excel and CSV files:

```bash
./run_automation.sh
```

### Command-line Options:

- `-s FILE`: Specify the input status file
- `-h FILE`: Specify the input history file
- `-o FILE`: Specify the output file  
- `-m FILE`: Specify the name mapping report file
- `-f FORMAT`: Specify the file format (xlsx or csv)
- `-H` or `--help`: Display help message

### Examples:

Run with default Excel files:
```bash
./run_automation.sh
```

Run with specific CSV files:
```bash
./run_automation.sh -f csv -s status.csv -h history.csv
```

Show help message:
```bash
./run_automation.sh --help
```
