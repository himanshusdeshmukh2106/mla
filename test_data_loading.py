#!/usr/bin/env python3
"""
Test the converted data format with the actual system
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append('src')

from data.loader import DataLoader
from data.preprocessor import DataPreprocessor

def test_data_format():
    """Test the converted CSV data with the system"""
    
    print("Testing converted data format with XGBoost trading system...")
    print("=" * 60)
    
    # Initialize components
    loader = DataLoader()
    preprocessor = DataPreprocessor()
    
    try:
        # Load the converted CSV
        print("1. Loading CSV data...")
        data = loader.load_csv_data("sample_data_5min.csv")
        print(f"   ✓ Loaded {len(data)} rows")
        print(f"   ✓ Columns: {list(data.columns)}")
        print(f"   ✓ Data types: {data.dtypes.to_dict()}")
        
        # Display sample data
        print("\n2. Sample data:")
        print(data.head())
        
        # Validate data integrity
        print("\n3. Validating data integrity...")
        validation_result = loader.validate_data_integrity(data)
        
        if validation_result.is_valid:
            print("   ✓ Data validation PASSED")
        else:
            print("   ✗ Data validation FAILED")
            print(f"   Errors: {validation_result.errors}")
        
        if validation_result.warnings:
            print(f"   Warnings: {validation_result.warnings}")
        
        # Test preprocessing
        print("\n4. Testing preprocessing pipeline...")
        processed_data = preprocessor.preprocess_pipeline(data, expected_freq='5min')
        print(f"   ✓ Preprocessing completed: {len(processed_data)} rows")
        print(f"   ✓ Index type: {type(processed_data.index)}")
        
        # Display processed sample
        print("\n5. Processed data sample:")
        print(processed_data.head())
        
        # Check data integrity after preprocessing
        print("\n6. Final integrity check...")
        final_integrity = preprocessor.validate_data_integrity(processed_data)
        
        if final_integrity['is_valid']:
            print("   ✓ Final integrity check PASSED")
        else:
            print("   ✗ Final integrity check FAILED")
            print(f"   Issues: {final_integrity['issues']}")
        
        print("\n" + "=" * 60)
        print("✓ DATA FORMAT TEST COMPLETED SUCCESSFULLY!")
        print("Your data format is compatible with the system.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        print("Data format needs adjustment.")
        return False

if __name__ == "__main__":
    test_data_format()