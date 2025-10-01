#!/usr/bin/env python3
"""
Test script for EMA trap feature engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append('src')

from features.engineer import FeatureEngineer
from features.target_generator import TargetGenerator
from config_manager import ConfigManager

def create_sample_data():
    """Create sample 5-minute OHLCV data for testing"""
    
    # Create datetime index for 5-minute intervals - multiple days for enough data
    start_time = datetime(2024, 1, 10, 9, 15)  # Start earlier for more data
    end_time = datetime(2024, 1, 20, 15, 30)   # Multiple days
    
    # Generate 5-minute intervals
    time_range = pd.date_range(start=start_time, end=end_time, freq='5min')
    
    # Create sample price data with some EMA crossovers
    np.random.seed(42)
    n_periods = len(time_range)
    
    # Base price around 100
    base_price = 100.0
    
    # Generate price movements with some trend
    price_changes = np.random.normal(0, 0.002, n_periods)  # 0.2% std dev
    
    # Add some specific patterns for EMA trap testing
    # Create multiple bearish and bullish traps throughout the data
    if n_periods > 100:
        # Bearish trap pattern 1
        price_changes[50] = 0.005   # Break above EMA
        price_changes[51] = 0.001   # Continue up slightly
        price_changes[52] = -0.008  # Fall back below EMA (trap confirmed)
        
        # Bullish trap pattern 1
        price_changes[70] = -0.005  # Break below EMA
        price_changes[71] = -0.001  # Continue down slightly
        price_changes[72] = 0.008   # Rise back above EMA (trap confirmed)
        
        # Bearish trap pattern 2
        price_changes[120] = 0.004  # Break above EMA
        price_changes[121] = 0.002  # Continue up
        price_changes[122] = -0.007 # Fall back below EMA
        
        # Bullish trap pattern 2
        price_changes[150] = -0.004 # Break below EMA
        price_changes[151] = -0.002 # Continue down
        price_changes[152] = 0.007  # Rise back above EMA
    
    # Calculate cumulative prices
    cumulative_returns = np.cumsum(price_changes)
    close_prices = base_price * (1 + cumulative_returns)
    
    # Generate OHLCV data
    data = []
    for i, (timestamp, close) in enumerate(zip(time_range, close_prices)):
        # Generate realistic OHLC from close price
        volatility = 0.001  # 0.1% intraday volatility
        
        open_price = close_prices[i-1] if i > 0 else close
        high = max(open_price, close) + np.random.uniform(0, volatility * close)
        low = min(open_price, close) - np.random.uniform(0, volatility * close)
        volume = np.random.randint(10000, 100000)
        
        data.append({
            'datetime': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('datetime', inplace=True)
    
    return df

def test_ema_trap_features():
    """Test EMA trap feature engineering"""
    
    print("Testing EMA Trap Feature Engineering")
    print("=" * 50)
    
    # Load configuration
    config_manager = ConfigManager()
    feature_config = config_manager.get_config('feature')
    
    # Create sample data
    print("Creating sample 5-minute OHLCV data...")
    data = create_sample_data()
    print(f"Created {len(data)} data points from {data.index[0]} to {data.index[-1]}")
    
    # Initialize feature engineer
    print("\nInitializing feature engineer...")
    engineer = FeatureEngineer(feature_config)
    
    # Create features
    print("Creating features...")
    features_df = engineer.create_features(data)
    
    print(f"Features created: {len(features_df)} rows, {len(features_df.columns)} columns")
    
    # Display EMA trap specific features
    ema_trap_columns = [
        'EMA_21', 'ADX', 'Price_Above_EMA21', 'Distance_From_EMA21_Pct',
        'EMA21_Cross_Above', 'EMA21_Cross_Below',
        'Bearish_Trap_Setup', 'Bullish_Trap_Setup',
        'Bearish_Trap_Confirmed', 'Bullish_Trap_Confirmed',
        'ADX_In_Range', 'In_Entry_Window', 'Small_Candle'
    ]
    
    print("\nEMA Trap Features Summary:")
    print("-" * 30)
    
    available_columns = [col for col in ema_trap_columns if col in features_df.columns]
    
    for col in available_columns:
        if col in features_df.columns:
            if features_df[col].dtype in ['int64', 'bool']:
                # For binary features, show count of 1s
                count_ones = features_df[col].sum()
                print(f"{col}: {count_ones} signals out of {len(features_df)} candles")
            else:
                # For continuous features, show basic stats
                mean_val = features_df[col].mean()
                print(f"{col}: mean = {mean_val:.4f}")
    
    # Show some sample rows with trap signals
    print("\nSample rows with trap signals:")
    print("-" * 40)
    
    trap_signals = features_df[
        (features_df['Bearish_Trap_Confirmed'] == 1) | 
        (features_df['Bullish_Trap_Confirmed'] == 1)
    ]
    
    if len(trap_signals) > 0:
        display_cols = ['close', 'EMA_21', 'Bearish_Trap_Confirmed', 'Bullish_Trap_Confirmed', 
                       'In_Entry_Window', 'ADX_In_Range', 'Small_Candle']
        available_display_cols = [col for col in display_cols if col in trap_signals.columns]
        print(trap_signals[available_display_cols].head())
    else:
        print("No trap signals found in sample data")
    
    # Test target generation
    print("\nTesting EMA trap target generation...")
    target_config = {
        'method': 'ema_trap',
        'lookahead_periods': 1,
        'profit_threshold': 0.002,  # 0.2%
        'loss_threshold': -0.002    # -0.2%
    }
    
    target_generator = TargetGenerator(target_config)
    targets_df = target_generator.generate_targets(features_df)
    
    # Show target distribution
    distribution = target_generator.get_target_distribution(targets_df)
    print(f"Target distribution: {distribution}")
    
    # Show validation results
    is_valid, validation_info = target_generator.validate_targets(targets_df)
    print(f"Target validation: {'PASSED' if is_valid else 'FAILED'}")
    print(f"Validation info: {validation_info}")
    
    print("\nTest completed successfully!")
    return features_df, targets_df

if __name__ == "__main__":
    try:
        features_df, targets_df = test_ema_trap_features()
        
        # Save sample results for inspection
        features_df.to_csv('sample_ema_trap_features.csv')
        print("\nSample features saved to 'sample_ema_trap_features.csv'")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()