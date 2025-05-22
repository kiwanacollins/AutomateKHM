#!/usr/bin/env python3
"""
Departure Status Automation Script

This script automates the process of:
1. Updating departure status from NULL to "TRAVELLED" for people who are in both departure_status.csv 
   and departure_history.csv files
2. Adding new entries to the departure status file for people who are only in the departure history file

Usage:
    python automate_departure_status.py [--input-status FILEPATH] [--input-history FILEPATH] [--output FILEPATH]

Example:
    python automate_departure_status.py --input-status departure_status.csv --input-history departure_history.csv --output updated_departure_status.csv
"""

import pandas as pd
import argparse
import os
import logging
import datetime
import re
from difflib import SequenceMatcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Console handler
    ]
)
logger = logging.getLogger(__name__)

def ensure_directory_exists(filepath):
    """
    Ensure the directory for the given filepath exists.
    Creates all necessary parent directories if they don't exist.
    
    Args:
        filepath: The path to a file for which to create the parent directory
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def load_csv_file(filepath, default_name=None):
    """
    Load a CSV file and return a pandas DataFrame.
    If file doesn't exist, log an error and exit.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"Could not find the file: {filepath}")
    
    try:
        # First try with comma separator
        df = pd.read_csv(filepath, na_values=['NULL', 'null', 'Null', 'NaN', 'nan', 'None', 'none'], 
                        keep_default_na=True)
        if len(df.columns) <= 1:
            # If only one column detected, try with tab separator
            df = pd.read_csv(filepath, sep='\t', na_values=['NULL', 'null', 'Null', 'NaN', 'nan', 'None', 'none'], 
                           keep_default_na=True)
        
        # If the file is loaded but empty
        if df.empty:
            logger.warning(f"The file {filepath} is empty.")
        else:
            logger.info(f"Successfully loaded {filepath} with {len(df)} rows.")
        
        # Convert all columns to string to avoid dtype issues
        for col in df.columns:
            if df[col].dtype != 'datetime64[ns]':
                df[col] = df[col].astype(str)
                df[col] = df[col].replace('nan', '')
                df[col] = df[col].replace('NULL', '')
                df[col] = df[col].replace('null', '')
                df[col] = df[col].replace('None', '')
        
        # Rename the DataFrame to help with debugging
        if default_name:
            df.name = default_name
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        # Try to load as Excel if CSV fails
        try:
            logger.info(f"Attempting to load {filepath} as Excel file...")
            df = pd.read_excel(filepath, na_values=['NULL', 'null', 'Null', 'NaN', 'nan', 'None', 'none'], 
                             keep_default_na=True)
            
            # Convert all columns to string to avoid dtype issues
            for col in df.columns:
                if df[col].dtype != 'datetime64[ns]':
                    df[col] = df[col].astype(str)
                    df[col] = df[col].replace('nan', '')
                    df[col] = df[col].replace('NULL', '')
                    df[col] = df[col].replace('null', '')
                    df[col] = df[col].replace('None', '')
            
            logger.info(f"Successfully loaded {filepath} as Excel with {len(df)} rows.")
            return df
        except Exception as excel_err:
            logger.error(f"Error loading {filepath} as Excel: {str(excel_err)}")
            raise

def clean_passport_numbers(df, passport_col):
    """
    Clean passport numbers by:
    - Converting to string
    - Removing any whitespace
    - Converting 'NULL', '0', and other non-valid values to empty string
    """
    # Handle NaN first
    df[passport_col] = df[passport_col].fillna('')
    
    # Convert to string
    df[passport_col] = df[passport_col].astype(str)
    
    # Strip whitespace
    df[passport_col] = df[passport_col].str.strip()
    
    # Convert 'NULL', '0', etc. to empty string
    invalid_values = ['NULL', 'null', 'Null', '0', 'nan', 'NaN', 'None', 'none']
    df.loc[df[passport_col].isin(invalid_values), passport_col] = ''
    
    return df

def split_name(name):
    """
    Split a full name into first name and last name.
    
    Uses a more sophisticated approach to handle different name formats:
    - If name has multiple parts, assumes first part is first name
    - If name has 2 parts, simple split
    - If name has 3+ parts, tries to identify common patterns
    """
    # Clean the name
    name = name.strip().upper()
    
    # Split the name by spaces
    parts = name.split()
    
    if len(parts) == 0:
        return "", ""
    elif len(parts) == 1:
        return parts[0], ""
    elif len(parts) == 2:
        return parts[0], parts[1]
    else:
        # For names with 3+ parts, there are a few common patterns
        
        # Check for common multi-word last names
        compound_last_names = ["DE", "VAN", "VON", "AL", "EL", "BIN", "BINTI"]
        
        # Find potential compound last name points
        for i in range(1, len(parts) - 1):
            if parts[i] in compound_last_names:
                # Found a compound last name indicator
                return " ".join(parts[:i]), " ".join(parts[i:])
        
        # Otherwise, use first part as first name and the rest as last name
        return parts[0], " ".join(parts[1:])

def extract_name_components(full_name):
    """
    Extract name components from a full name.
    Returns a dictionary with 'first_name' and 'last_name' keys.
    """
    first_name, last_name = split_name(full_name)
    return {
        'first_name': first_name,
        'last_name': last_name
    }

def standardize_name(name):
    """
    Standardize a name for better comparison by:
    - Converting to uppercase
    - Removing extra whitespace
    - Removing punctuation
    - Removing common titles and prefixes
    
    Returns a clean name string.
    """
    if not name:
        return ""
    
    # Convert to uppercase and strip whitespace
    name = name.upper().strip()
    
    # Remove common titles and prefixes
    titles = ["MR", "MR.", "MS", "MS.", "MRS", "MRS.", "DR", "DR.", "PROF", "PROF."]
    for title in titles:
        if name.startswith(title + " "):
            name = name[len(title):].strip()
    
    # Remove punctuation
    name = re.sub(r'[^\w\s]', '', name)
    
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def get_name_parts(name):
    """
    Break a name into parts (first, middle, last) for more flexible matching.
    Returns a list of name parts.
    """
    if not name:
        return []
    
    # Standardize first
    name = standardize_name(name)
    
    # Split into parts
    parts = name.split()
    return parts

def compare_names(name1, name2):
    """
    Compare two names using multiple techniques for more accurate matching.
    Returns a score between 0 and 1, with 1 being a perfect match.
    """
    if not name1 or not name2:
        return 0.0
    
    # Standardize both names
    std_name1 = standardize_name(name1)
    std_name2 = standardize_name(name2)
    
    # If standardized names are identical, it's a perfect match
    if std_name1 == std_name2:
        return 1.0
    
    # Get name parts
    parts1 = get_name_parts(std_name1)
    parts2 = get_name_parts(std_name2)
    
    # If either name has no parts, return 0
    if not parts1 or not parts2:
        return 0.0
    
    # Calculate overall sequence similarity
    sequence_sim = SequenceMatcher(None, std_name1, std_name2).ratio()
    
    # Calculate part-by-part matches
    matching_parts = 0
    total_parts = max(len(parts1), len(parts2))
    
    # Check if any parts match exactly between names
    for part1 in parts1:
        if any(part1 == part2 for part2 in parts2):
            matching_parts += 1
    
    # Calculate part matching ratio
    part_match_ratio = matching_parts / total_parts if total_parts > 0 else 0
    
    # Calculate match score as weighted average of sequence similarity and part matching
    score = (sequence_sim * 0.6) + (part_match_ratio * 0.4)
    
    return score

def name_similarity(name1, name2):
    """
    Calculate similarity between two names.
    Returns a value between 0 and 1, where 1 means exact match.
    """
    if not name1 or not name2:
        return 0.0
    
    # Use the enhanced comparison function
    return compare_names(name1, name2)

def find_matching_person(history_row, status_df, name_threshold=0.7):
    """
    Find a matching person in status_df for the given history_row.
    
    First tries to match by passport number.
    If no match found by passport, tries to match by name similarity.
    
    Returns the index of the matching row in status_df, or None if no match found.
    """
    hist_passport = history_row['PASSPORT NO']
    
    # If there's a valid passport number, try to match by passport
    if hist_passport and hist_passport != '0':
        # Check if this passport exists in the status dataframe
        matching_rows = status_df.loc[status_df['PASSPORT NO'] == hist_passport]
        
        if not matching_rows.empty:
            return matching_rows.index[0]  # Return the first matching row
    
    # If we get here, no passport match was found. Try matching by name.
    hist_full_name = history_row['CANDIDATE NAME']
    
    # Try different name matching strategies
    best_match_idx = None
    best_match_score = 0.0
    
    for idx, status_row in status_df.iterrows():
        status_first = status_row['First_name']
        status_last = status_row['Last_name']
        
        # Combine first and last names from status file
        status_full = f"{status_first} {status_last}".strip()
        
        # Calculate name similarity using enhanced method
        similarity = compare_names(hist_full_name, status_full)
        
        # Also try matching with name parts
        hist_parts = get_name_parts(hist_full_name)
        status_parts = get_name_parts(status_full)
        
        # Check for exact matches on individual name parts
        exact_match_bonus = 0.0
        if hist_parts and status_parts:
            for part in hist_parts:
                if part in status_parts and len(part) > 2:  # Only consider parts longer than 2 chars
                    exact_match_bonus += 0.1
            exact_match_bonus = min(0.2, exact_match_bonus)  # Cap bonus at 0.2
        
        # Apply bonus to similarity score
        adjusted_similarity = min(1.0, similarity + exact_match_bonus)
        
        # Check if this is the best match so far
        if adjusted_similarity > best_match_score and adjusted_similarity >= name_threshold:
            best_match_score = adjusted_similarity
            best_match_idx = idx
    
    if best_match_score >= name_threshold:
        logger.info(f"Found name match for '{hist_full_name}' with confidence {best_match_score:.2f}")
        return best_match_idx
    
    return None  # No good match found

def update_departure_status(status_df, history_df):
    """
    Update the departure status dataframe based on the history dataframe.
    
    1. For people in both dataframes, set departure status to "TRAVELLED"
    2. Add new entries for people who are only in the history dataframe
    """
    logger.info("Starting update process...")
    
    # Make a copy of the original dataframes to avoid modifying the originals
    status_df = status_df.copy()
    history_df = history_df.copy()
    
    # Clean passport numbers in both dataframes
    status_df = clean_passport_numbers(status_df, 'PASSPORT NO')
    history_df = clean_passport_numbers(history_df, 'PASSPORT NO')
    
    # List to track passport numbers that have been updated
    updated_passports = []
    
    # List to track passport numbers that have been added
    added_passports = []
    
    # Track statistics for validation
    stats = {
        'exact_passport_matches': 0,
        'name_matches': 0,
        'new_entries': 0,
        'no_action': 0
    }
    
    # 1. Update departure status for people in both dataframes
    for _, history_row in history_df.iterrows():
        hist_passport = history_row['PASSPORT NO']
        
        # Skip rows with invalid passport numbers
        if hist_passport == '' or hist_passport == '0':
            continue
        
        # Find a matching person in the status dataframe
        match_idx = find_matching_person(history_row, status_df)
        
        if match_idx is not None:
            # A match was found - check if it was an exact passport match
            if status_df.loc[match_idx, 'PASSPORT NO'] == hist_passport:
                stats['exact_passport_matches'] += 1
            else:
                stats['name_matches'] += 1
            
            # Check the current status and update if needed
            current_status = str(status_df.loc[match_idx, 'Departure_status']).upper()
            if current_status in ['NULL', '', 'NAN', 'NONE'] or current_status is None:
                status_df.at[match_idx, 'Departure_status'] = "TRAVELLED"
                updated_passports.append(hist_passport)
                logger.info(f"Updated status to TRAVELLED for passport {hist_passport}")
            else:
                stats['no_action'] += 1
                logger.info(f"No update needed for passport {hist_passport}, already has status: {current_status}")
        else:
            # No match found - this is a new entry
            stats['new_entries'] += 1
            
            # Get the next S/N value
            try:
                # Convert S/N to integers for correct numbering
                status_df['S/N_int'] = pd.to_numeric(status_df['S/N'], errors='coerce')
                next_sn = int(status_df['S/N_int'].max() + 1)
                # Remove temporary column
                status_df.drop('S/N_int', axis=1, inplace=True)
            except Exception as e:
                logger.warning(f"Error calculating next S/N value: {str(e)}. Using length+1.")
                next_sn = len(status_df) + 1
            
            # Split the candidate name into first and last name
            name_components = extract_name_components(history_row['CANDIDATE NAME'])
            first_name = name_components['first_name']
            last_name = name_components['last_name']
            
            # Create a new row
            new_row = {
                'S/N': next_sn,
                'First_name': first_name,
                'Last_name': last_name,
                'PASSPORT NO': hist_passport,
                'Departure_status': "TRAVELLED",
                'Travel_date': "",  # Use empty string for consistency
                'Arrival_date': ""
            }
            
            # Add the new row to the status dataframe
            status_df.loc[len(status_df)] = new_row
            added_passports.append(hist_passport)
            logger.info(f"Added new entry for passport {hist_passport}")
    
    # Log statistics
    logger.info(f"Statistics: {stats}")
    logger.info(f"Update process completed. Updated {len(updated_passports)} records and added {len(added_passports)} new records.")
    
    return status_df, stats

def validate_results(original_status_df, history_df, updated_status_df):
    """
    Validate that all history records are properly reflected in the updated status dataframe.
    
    Returns a tuple of (is_valid, validation_report)
    """
    logger.info("Validating results...")
    
    # Clean passport numbers in all dataframes
    original_status_df = clean_passport_numbers(original_status_df.copy(), 'PASSPORT NO')
    history_df = clean_passport_numbers(history_df.copy(), 'PASSPORT NO')
    updated_status_df = clean_passport_numbers(updated_status_df.copy(), 'PASSPORT NO')
    
    # Keep track of validation issues
    issues = []
    
    # 1. Check that all passport numbers from history are in updated status
    for _, history_row in history_df.iterrows():
        hist_passport = history_row['PASSPORT NO']
        
        # Skip rows with invalid passport numbers
        if hist_passport == '' or hist_passport == '0':
            continue
        
        # Check if this passport exists in the updated dataframe
        matching_rows = updated_status_df.loc[updated_status_df['PASSPORT NO'] == hist_passport]
        
        if matching_rows.empty:
            # No match found by passport - try matching by name
            hist_full_name = history_row['CANDIDATE NAME']
            
            found_name_match = False
            for _, status_row in updated_status_df.iterrows():
                status_first = status_row['First_name']
                status_last = status_row['Last_name']
                status_full = f"{status_first} {status_last}".strip()
                
                if compare_names(hist_full_name, status_full) >= 0.7:
                    found_name_match = True
                    break
            
            if not found_name_match:
                issues.append(f"Passport {hist_passport} ({history_row['CANDIDATE NAME']}) from history not found in updated status")
        else:
            # Check that the status is set to TRAVELLED
            for idx in matching_rows.index:
                status = str(updated_status_df.loc[idx, 'Departure_status']).upper()
                if status != 'TRAVELLED':
                    issues.append(f"Passport {hist_passport} in updated status has incorrect status: {status}")
    
    # 2. Check that all records that should have been updated were updated
    for _, original_row in original_status_df.iterrows():
        orig_passport = original_row['PASSPORT NO']
        
        # Skip rows with invalid passport numbers
        if orig_passport == '' or orig_passport == '0':
            continue
        
        # Check if this passport exists in the history dataframe
        matching_history = history_df.loc[history_df['PASSPORT NO'] == orig_passport]
        
        if not matching_history.empty:
            # This passport should have been updated to TRAVELLED
            matching_updated = updated_status_df.loc[updated_status_df['PASSPORT NO'] == orig_passport]
            
            if not matching_updated.empty:
                for idx in matching_updated.index:
                    status = str(updated_status_df.loc[idx, 'Departure_status']).upper()
                    if status != 'TRAVELLED':
                        issues.append(f"Original passport {orig_passport} should have been updated to TRAVELLED but has status: {status}")
    
    # Generate validation report
    validation_report = "Validation Report:\n"
    validation_report += f"Total records in original status: {len(original_status_df)}\n"
    validation_report += f"Total records in history: {len(history_df)}\n"
    validation_report += f"Total records in updated status: {len(updated_status_df)}\n"
    validation_report += f"Total issues found: {len(issues)}\n"
    
    if issues:
        validation_report += "\nIssues found:\n"
        for issue in issues:
            validation_report += f"- {issue}\n"
    else:
        validation_report += "\nNo issues found. All records processed correctly.\n"
    
    logger.info(f"Validation completed. {len(issues)} issues found.")
    return len(issues) == 0, validation_report

def build_name_mapping(status_df, history_df, similarity_threshold=0.7):
    """
    Build a mapping between names in the departure status and departure history files.
    
    This creates a dictionary that maps candidate names from the history file
    to corresponding entries in the status file.
    
    Args:
        status_df: DataFrame containing the departure status data
        history_df: DataFrame containing the departure history data
        similarity_threshold: Minimum similarity score to consider a match
        
    Returns:
        A dictionary mapping history names to status indices
    """
    logger.info("Building name mapping between departure files...")
    
    # Dictionary to store the mapping
    name_mapping = {}
    
    # Track statistics
    stats = {
        'exact_matches': 0,
        'high_confidence_matches': 0,
        'low_confidence_matches': 0,
        'no_matches': 0
    }
    
    # First, try matching by passport number
    passport_matches = {}
    for _, history_row in history_df.iterrows():
        hist_passport = history_row['PASSPORT NO']
        hist_name = history_row['CANDIDATE NAME']
        
        # Skip rows with invalid passport numbers
        if not hist_passport or hist_passport == '0':
            continue
        
        # Find matching passport in status file
        matching_rows = status_df.loc[status_df['PASSPORT NO'] == hist_passport]
        
        if not matching_rows.empty:
            # Found a passport match
            idx = matching_rows.index[0]
            passport_matches[hist_name] = idx
            name_mapping[hist_name] = idx
            stats['exact_matches'] += 1
    
    # For names without passport matches, try name matching
    for _, history_row in history_df.iterrows():
        hist_name = history_row['CANDIDATE NAME']
        
        # Skip if already matched by passport
        if hist_name in passport_matches:
            continue
        
        # Find the best name match
        best_match_idx = None
        best_match_score = 0.0
        
        for idx, status_row in status_df.iterrows():
            status_first = status_row['First_name']
            status_last = status_row['Last_name']
            status_full = f"{status_first} {status_last}".strip()
            
            # Calculate similarity score
            similarity = compare_names(hist_name, status_full)
            
            # Keep track of the best match
            if similarity > best_match_score:
                best_match_score = similarity
                best_match_idx = idx
        
        # Categorize the match based on confidence
        if best_match_score >= 0.9:
            # Very high confidence match
            name_mapping[hist_name] = best_match_idx
            stats['high_confidence_matches'] += 1
        elif best_match_score >= similarity_threshold:
            # Acceptable match based on threshold
            name_mapping[hist_name] = best_match_idx
            stats['low_confidence_matches'] += 1
        else:
            # No good match found
            stats['no_matches'] += 1
    
    # Log summary statistics only
    logger.info(f"Name mapping complete: {stats['exact_matches']} exact matches, {stats['high_confidence_matches']} high confidence, {stats['low_confidence_matches']} low confidence, {stats['no_matches']} no matches")
    
    return name_mapping

def main():
    """Main function to process the CSV files."""
    parser = argparse.ArgumentParser(description='Update departure status based on departure history.')
    parser.add_argument('--input-status', required=False, default='departure_status.csv',
                        help='Path to the input departure status CSV file')
    parser.add_argument('--input-history', required=False, default='departure_history.csv',
                        help='Path to the input departure history CSV file')
    parser.add_argument('--output', required=False, default='updated_departure_status.csv',
                        help='Path to the output CSV file')
    parser.add_argument('--skip-validation', action='store_true', 
                        help='Skip the validation step')
    parser.add_argument('--generate-mapping-report', action='store_true',
                        help='Generate a report showing how names are mapped between files')
    parser.add_argument('--mapping-report-path', default='name_mapping_report.csv',
                        help='Path to save the name mapping report (if --generate-mapping-report is specified)')
    parser.add_argument('--output-dir', default='output',
                        help='Directory to save output files if relative paths are provided')
    args = parser.parse_args()
    
    # If output is Excel, make mapping report Excel by default too
    if args.generate_mapping_report and args.output.lower().endswith('.xlsx') and args.mapping_report_path == 'name_mapping_report.csv':
        args.mapping_report_path = 'name_mapping_report.xlsx'
    
    # Convert relative output paths to include output directory
    if not os.path.isabs(args.output):
        if '/' not in args.output and '\\' not in args.output:  # If it's just a filename with no path
            args.output = os.path.join(args.output_dir, args.output)
    
    # Convert relative mapping report path to include output directory
    if args.generate_mapping_report and not os.path.isabs(args.mapping_report_path):
        if '/' not in args.mapping_report_path and '\\' not in args.mapping_report_path:
            args.mapping_report_path = os.path.join(args.output_dir, args.mapping_report_path)
    
    try:
        # Load the CSV files
        status_df = load_csv_file(args.input_status, "status")
        history_df = load_csv_file(args.input_history, "history")
        
        # Make a copy of the original status dataframe for validation
        original_status_df = status_df.copy()
        
        # Build name mapping if report is requested
        if args.generate_mapping_report:
            name_mapping = build_name_mapping(status_df, history_df)
            # Import the report generator if needed
            from name_mapping_report import generate_name_mapping_report
            report_df = generate_name_mapping_report(status_df, history_df, name_mapping, args.mapping_report_path)
            
            # Calculate missing people count
            missing_count = len([r for r in report_df['In Status File'].tolist() if r == 'No'])
            
            logger.info(f"Name mapping report generated with information on {len(history_df)} history records")
            logger.info(f"IMPORTANT: {missing_count} people from history are not in the status file")
            
            # These missing people will be added as new entries
            print(f"\nIMPORTANT: {missing_count} people from departure history are not in the departure status file")
            print(f"These individuals will be added as new entries with status 'TRAVELLED'")
            print(f"See {args.mapping_report_path} for detailed information\n")
        
        # Update the departure status
        updated_status_df, stats = update_departure_status(status_df, history_df)
        
        # Ensure output directory exists and save the updated dataframe
        output_path = args.output
        ensure_directory_exists(output_path)
        
        # Detect format based on file extension
        if output_path.lower().endswith('.xlsx') or output_path.lower().endswith('.xls'):
            updated_status_df.to_excel(output_path, index=False)
            logger.info(f"Updated departure status saved to Excel file: {output_path}")
        else:
            updated_status_df.to_csv(output_path, index=False)
            logger.info(f"Updated departure status saved to CSV file: {output_path}")
        
        # Validate the results if not skipped
        if not args.skip_validation:
            is_valid, validation_report = validate_results(original_status_df, history_df, updated_status_df)
            logger.info(validation_report)
            
            if not is_valid:
                logger.warning("Validation found issues. Please check the log file for details.")
        
        # Print summary
        print("\nSummary:")
        print(f"Total records in original status file: {len(original_status_df)}")
        print(f"Total records in history file: {len(history_df)}")
        print(f"Total records in updated status file: {len(updated_status_df)}")
        print(f"Records updated: {stats['exact_passport_matches'] + stats['name_matches'] - stats['no_action']}")
        print(f"New records added: {stats['new_entries']}")
        print(f"Output file saved to: {args.output}")
        if args.generate_mapping_report:
            print(f"Name mapping report saved to: {args.mapping_report_path}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    # Ensure logs directory exists
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created directory: {log_dir}")
    
    # Set up file logger with timestamp
    log_filename = f"{log_dir}/departure_automation_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("====== Departure Status Automation Started ======")
    logger.info(f"Script executed at {datetime.datetime.now()}")
    main()
    logger.info("====== Departure Status Automation Completed ======")
