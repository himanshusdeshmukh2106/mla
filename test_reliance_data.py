#!/usr/bin/env python3
"""
Test script to validate Reliance 5-minute data loading and preprocessing
"""

import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append('src')

from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from features.engineer import FeatureEngineer
from features.target_generator import TargetGenerator
from config_manager import ConfigManager

def test_reliance_data_pipeline():
    """Test the complete data pipeline with Reliance data"""
    
    print("=" * 60)
    print("RELIANCE 5-MIN DATA PIPELINE TEST")
    print("=" * 60)
    
    # Initialize components
    config_manager = ConfigManager("config")
    data_config = config_manager.get_data_config()
    feature_config = config_manager.get_feature_config()
    
    print(f"\nData Config:")
    print(f"  Symbols: {data_config.symbols}")
    print(f"  Timeframe: {data_config.timeframe}")
    print(f"  Date Range: {data_config.start_date} to {data_config.end_date}")
    
    # Step 1: Load raw data
    print(f"\n1. Loading Reliance data...")
    loader = DataLoader()
    
    try:
        raw_data = loader.load_ohlcv_data(
            symbol="RELIANCE",
            timeframe="5min",
            start_date=datetime.strptime(data_config.start_date, '%Y-%m-%d'),
            end_date=datetime.strptime(data_config.end_date, '%Y-%m-%d')
        )
        print(f"   ✓ Loaded {len(raw_data)} rows of raw data")
        print(f"   ✓ Date range: {raw_data['datetime'].min()} to {raw_data['datetime'].max()}")
        print(f"   ✓ Columns: {list(raw_data.columns)}")
        
        # Display sample data
        print(f"\n   Sample data (first 5 rows):")
        print(raw_data.head().to_string())
        
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return False
    
    # Step 2: Validate data integrity
    print(f"\n2. Validating data integrity...")
    validation_result = loader.validate_data_integrity(raw_data)
    
    if validation_result.is_valid:
        print(f"   ✓ Data validation passed")
    else:
        print(f"   ⚠ Data validation issues:")
        for error in validation_result.errors:
            print(f"     - {error}")
        for warning in validation_result.warnings:
            print(f"     - WARNING: {warning}")
    
    # Step 3: Preprocess data
    print(f"\n3. Preprocessing data...")
    preprocessor = DataPreprocessor()
    
    try:
        processed_data = preprocessor.preprocess_pipeline(raw_data, expected_freq='5min')
        print(f"   ✓ Preprocessing completed: {len(processed_data)} rows")
        
        # Check for any data quality issues
        integrity_report = preprocessor.validate_data_integrity(processed_data)
        if integrity_report['is_valid']:
            print(f"   ✓ Data integrity validation passed")
        else:
            print(f"   ⚠ Data integrity issues found:")
            for issue in integrity_report['issues']:
                print(f"     - {issue}")
        
    except Exception as e:
        print(f"   ✗ Error preprocessing data: {e}")
        return False
    
    # Step 4: Engineer features
    print(f"\n4. Engineering features...")
    feature_engineer = FeatureEngineer(feature_config.__dict__)
    
    try:
        featured_data = feature_engineer.create_features(processed_data)
        feature_names = [col for col in featured_data.columns 
                        if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        print(f"   ✓ Created {len(feature_names)} features")
        print(f"   ✓ Final dataset: {len(featured_data)} rows")
        print(f"   ✓ Features created: {feature_names[:10]}..." if len(feature_names) > 10 else f"   ✓ Features: {feature_names}")
        
    except Exception as e:
        print(f"   ✗ Error engineering features: {e}")
        return False
    
    # Step 5: Generate targets
    print(f"\n5. Generating target variables...")
    target_config = {
        'lookahead_periods': 1,
        'profit_threshold': 0.002,  # 0.2% for 5-min intraday
        'loss_threshold': -0.002,   # -0.2% for 5-min intraday
        'method': 'next_period_return'
    }
    
    target_generator = TargetGenerator(target_config)
    
    try:
        final_data = target_generator.generate_targets(featured_data)
        
        # Get target distribution
        target_dist = target_generator.get_target_distribution(final_data)
        
        print(f"   ✓ Target generation completed: {len(final_data)} rows")
        print(f"   ✓ Target distribution:")
        for key, value in target_dist.items():
            if not key.endswith('_pct'):
                pct_key = f"{key}_pct"
                pct_value = target_dist.get(pct_key, 0)
                print(f"     - Class {key}: {value} samples ({pct_value}%)")
        
        # Validate targets
        is_valid, validation_info = target_generator.validate_targets(final_data)
        if is_valid:
            print(f"   ✓ Target validation passed")
        else:
            print(f"   ⚠ Target validation issues: {validation_info}")
        
    except Exception as e:
        print(f"   ✗ Error generating targets: {e}")
        return False
    
    # Step 6: Data summary
    print(f"\n6. Final Data Summary:")
    print(f"   - Total samples: {len(final_data)}")
    print(f"   - Features: {len(feature_names)}")
    print(f"   - Date range: {final_data.index.min()} to {final_data.index.max()}")
    print(f"   - Memory usage: {final_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Display sample of final data
    print(f"\n   Sample of final processed data:")
    sample_cols = ['close'] + feature_names[:5] + ['target', 'target_binary']
    available_cols = [col for col in sample_cols if col in final_data.columns]
    print(final_data[available_cols].head().to_string())
    
    print(f"\n" + "=" * 60)
    print("PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return True

def check_data_file_exists():
    """Check if the Reliance data file exists"""
    data_file = Path("data/reliance_data_5min_full_year.csv")
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        print("Please ensure the file exists in the correct location.")
        return False
    
    print(f"✓ Data file found: {data_file}")
    print(f"✓ File size: {data_file.stat().st_size / 1024**2:.2f} MB")
    return True

if __name__ == "__main__":
    print("Checking data file availability...")
    if not check_data_file_exists():
        sys.exit(1)
    
    print("\nStarting pipeline test...")
    success = test_reliance_data_pipeline()
    
    if success:
        print("\n🎉 All tests passed! Your Reliance data is ready for training.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)