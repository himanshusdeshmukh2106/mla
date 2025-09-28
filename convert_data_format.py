#!/usr/bin/env python3
"""
Convert space-separated OHLCV data to proper CSV format
"""

def convert_to_csv(input_file, output_file):
    """
    Convert space-separated OHLCV data to CSV format
    
    Args:
        input_file: Path to input file with space-separated data
        output_file: Path to output CSV file
    """
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Write CSV header
        outfile.write("datetime,open,high,low,close,volume\n")
        
        for line in infile:
            line = line.strip()
            if line:  # Skip empty lines
                # Split by spaces and join with commas
                parts = line.split()
                if len(parts) >= 6:
                    # Combine date and time parts if they're separate
                    datetime_part = f"{parts[0]} {parts[1]}"
                    ohlcv_parts = parts[2:7]  # Take next 5 parts (OHLCV)
                    
                    # Create CSV line
                    csv_line = f"{datetime_part},{','.join(ohlcv_parts)}\n"
                    outfile.write(csv_line)

def convert_raw_data_string(data_string):
    """
    Convert raw data string to CSV format
    
    Args:
        data_string: Multi-line string with space-separated data
        
    Returns:
        CSV formatted string
    """
    lines = data_string.strip().split('\n')
    csv_lines = ["datetime,open,high,low,close,volume"]
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 6:
            datetime_part = f"{parts[0]} {parts[1]}"
            ohlcv_parts = parts[2:7]
            csv_line = f"{datetime_part},{','.join(ohlcv_parts)}"
            csv_lines.append(csv_line)
    
    return '\n'.join(csv_lines)

# Example usage with your data
sample_data = """2023-01-09 13:10:00+05:30 1235.7 1235.8 1233 1233.6 131024
2023-01-09 13:15:00+05:30 1233.6 1233.85 1231 1232 123192
2023-01-09 13:20:00+05:30 1231.8 1232.4 1231.3 1231.4 74662
2023-01-09 13:25:00+05:30 1231.4 1233.1 1231.05 1231.8 80562
2023-01-09 13:30:00+05:30 1231.8 1232.5 1230.2 1230.95 88324
2023-01-09 13:35:00+05:30 1230.2 1231.05 1230.15 1230.65 62066"""

if __name__ == "__main__":
    # Convert sample data
    converted_csv = convert_raw_data_string(sample_data)
    print("Converted CSV format:")
    print(converted_csv)
    
    # Save to file
    with open("sample_data_5min.csv", "w") as f:
        f.write(converted_csv)
    
    print("\nSaved to sample_data_5min.csv")
    
    # If you have a file to convert, uncomment and modify:
    # convert_to_csv("your_input_file.txt", "output_5min.csv")