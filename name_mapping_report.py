import pandas as pd
import logging

# Get the logger
logger = logging.getLogger(__name__)

def generate_name_mapping_report(status_df, history_df, name_mapping, output_path='name_mapping_report.csv'):
    """
    Generate a CSV report showing how names from the history file are mapped to the status file.
    Highlights people who are in history but not in status, and provides a count.
    
    Args:
        status_df: DataFrame containing the departure status data
        history_df: DataFrame containing the departure history data
        name_mapping: Dictionary mapping history names to status indices
        output_path: Path where the report will be saved
    """
    # Import the name comparison function
    from automate_departure_status import compare_names
    
    # Create a list to hold the report data
    report_data = []
    
    # Counter for people in history but not in status
    missing_in_status_count = 0
    
    for _, history_row in history_df.iterrows():
        hist_name = history_row['CANDIDATE NAME']
        hist_passport = history_row['PASSPORT NO']
        
        matched_idx = name_mapping.get(hist_name)
        
        if matched_idx is not None:
            # Found a match in the status file
            status_row = status_df.loc[matched_idx]
            status_first = status_row['First_name']
            status_last = status_row['Last_name']
            status_full = f"{status_first} {status_last}".strip()
            status_passport = status_row['PASSPORT NO']
            
            # Calculate similarity score
            similarity = compare_names(hist_name, status_full)
            
            # Determine match type
            if hist_passport and status_passport and hist_passport == status_passport:
                match_type = "Passport Match"
            elif similarity >= 0.9:
                match_type = "High Confidence Name Match"
            else:
                match_type = "Low Confidence Name Match"
            
            # Not missing in status
            in_status = "Yes"
        else:
            # No match found - missing in status
            status_full = "NO MATCH"
            status_passport = ""
            similarity = 0.0
            match_type = "No Match"
            in_status = "No"
            missing_in_status_count += 1
            
        report_data.append({
            'History Name': hist_name,
            'History Passport': hist_passport,
            'Status Name': status_full,
            'Status Passport': status_passport,
            'Similarity Score': f"{similarity:.2f}",
            'Match Type': match_type,
            'In Status File': in_status
        })
    
    # Create a DataFrame from the report data
    report_df = pd.DataFrame(report_data)
    
    # Add a summary row
    summary_row = pd.DataFrame([{
        'History Name': f"TOTAL MISSING IN STATUS: {missing_in_status_count}",
        'History Passport': f"{missing_in_status_count} of {len(history_df)} records",
        'Status Name': "",
        'Status Passport': "",
        'Similarity Score': "",
        'Match Type': "Summary",
        'In Status File': ""
    }])
    
    # Concatenate the summary row to the report
    report_df = pd.concat([report_df, summary_row], ignore_index=True)
    
    # Ensure the output directory exists
    from automate_departure_status import ensure_directory_exists
    ensure_directory_exists(output_path)
    
    # Save the report based on file extension
    if output_path.lower().endswith('.xlsx') or output_path.lower().endswith('.xls'):
        report_df.to_excel(output_path, index=False)
        logger.info(f"Name mapping report generated with {missing_in_status_count} people missing in status file")
    else:
        report_df.to_csv(output_path, index=False)
        logger.info(f"Name mapping report generated with {missing_in_status_count} people missing in status file")
    
    return report_df
