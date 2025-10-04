"""
EMA Crossover + Retest Strategy - Feature Engineering
Creates sophisticated features for 8/30 EMA crossover and retest setups
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')


def create_ema_crossover_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive features for EMA crossover + retest strategy
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with all features
    """
    
    print("Creating EMA Crossover Strategy features...")
    
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Ensure datetime column
    if 'datetime' not in data.columns and data.index.name != 'datetime':
        if 'timestamp' in data.columns:
            data['datetime'] = pd.to_datetime(data['timestamp'])
        else:
            print("Warning: No datetime column found")
    
    # Calculate EMAs
    print("  Calculating EMAs...")
    data['EMA_8'] = ta.ema(data['close'], length=8)
    data['EMA_30'] = ta.ema(data['close'], length=30)
    
    # =================================================================
    # CATEGORY 1: CORE EMA DYNAMIC FEATURES
    # =================================================================
    print("  Creating Core EMA Dynamic Features...")
    
    # EMA Spread (normalized by price)
    data['EMA_Spread'] = (data['EMA_8'] - data['EMA_30']) / data['close']
    
    # EMA Spread Rate of Change
    data['EMA_Spread_ROC'] = data['EMA_Spread'].diff()
    
    # EMA Slopes (rate of change over multiple periods)
    for periods in [3, 5]:
        data[f'EMA_8_Slope_{periods}'] = data['EMA_8'].diff(periods) / data['EMA_8'].shift(periods)
        data[f'EMA_30_Slope_{periods}'] = data['EMA_30'].diff(periods) / data['EMA_30'].shift(periods)
    
    # EMA Crossover Detection
    data['EMA_8_Above_30'] = (data['EMA_8'] > data['EMA_30']).astype(int)
    data['EMA_Cross_Above'] = ((data['EMA_8'] > data['EMA_30']) & 
                               (data['EMA_8'].shift(1) <= data['EMA_30'].shift(1))).astype(int)
    data['EMA_Cross_Below'] = ((data['EMA_8'] < data['EMA_30']) & 
                               (data['EMA_8'].shift(1) >= data['EMA_30'].shift(1))).astype(int)
    
    # =================================================================
    # CATEGORY 2: PRICE-TO-EMA RELATIONSHIP FEATURES
    # =================================================================
    print("  Creating Price-to-EMA Relationship Features...")
    
    # Price distances from EMAs (normalized)
    data['Price_Distance_EMA8'] = (data['close'] - data['EMA_8']) / data['close']
    data['Price_Distance_EMA30'] = (data['close'] - data['EMA_30']) / data['close']
    
    # Price position relative to EMAs
    def get_price_position(row):
        if row['close'] < row['EMA_8'] and row['close'] < row['EMA_30']:
            return 0  # Below both
        elif row['close'] > row['EMA_8'] and row['close'] > row['EMA_30']:
            return 2  # Above both
        else:
            return 1  # Between EMAs
    
    data['Price_Position_Flag'] = data.apply(get_price_position, axis=1)
    
    # EMA retest detection
    data['Close_Below_EMA8'] = (data['close'] < data['EMA_8']).astype(int)
    data['Close_Below_EMA30'] = (data['close'] < data['EMA_30']).astype(int)
    data['Close_Above_EMA8'] = (data['close'] > data['EMA_8']).astype(int)
    data['Close_Above_EMA30'] = (data['close'] > data['EMA_30']).astype(int)
    
    # Retest patterns (simplified)
    data['Near_EMA8'] = (abs(data['Price_Distance_EMA8']) < 0.002).astype(int)  # Within 0.2%
    data['Near_EMA30'] = (abs(data['Price_Distance_EMA30']) < 0.002).astype(int)  # Within 0.2%
    
    # =================================================================
    # CATEGORY 3: PRICE ACTION & VOLATILITY FEATURES
    # =================================================================
    print("  Creating Price Action & Volatility Features...")
    
    # Candle characteristics
    data['Candle_Body_Size'] = abs(data['close'] - data['open']) / data['close']
    data['Candle_Range'] = (data['high'] - data['low']) / data['close']
    
    # Wick-to-Body Ratio
    total_wick = data['Candle_Range'] - data['Candle_Body_Size']
    data['Wick_to_Body_Ratio'] = total_wick / (data['Candle_Body_Size'] + 1e-10)
    
    # Candle direction and strength
    data['Green_Candle'] = (data['close'] > data['open']).astype(int)
    data['Red_Candle'] = (data['close'] < data['open']).astype(int)
    data['Strong_Candle'] = (data['Candle_Body_Size'] > data['Candle_Body_Size'].rolling(20).mean()).astype(int)
    
    # Rolling price volatility
    data['Price_Returns'] = data['close'].pct_change()
    data['Rolling_Volatility'] = data['Price_Returns'].rolling(20).std()
    
    # =================================================================
    # CATEGORY 4: VOLUME & CONVICTION FEATURES
    # =================================================================
    print("  Creating Volume & Conviction Features...")
    
    # Volume features
    data['Volume_MA20'] = data['volume'].rolling(20).mean()
    data['Volume_Spike'] = data['volume'] / data['Volume_MA20']
    data['Volume_ROC'] = data['volume'].pct_change()
    
    # Volume-Price relationship
    data['Volume_Price_Trend'] = (data['Volume_Spike'] > 1.2) & (data['Strong_Candle'] == 1)
    data['Volume_Price_Trend'] = data['Volume_Price_Trend'].astype(int)
    
    # =================================================================
    # CATEGORY 5: HISTORICAL CONTEXT (LAG FEATURES)
    # =================================================================
    print("  Creating Historical Context Features...")
    
    # Lagged EMA Spread
    for lag in [1, 2, 3]:
        data[f'EMA_Spread_Lag_{lag}'] = data['EMA_Spread'].shift(lag)
        data[f'Price_Distance_EMA30_Lag_{lag}'] = data['Price_Distance_EMA30'].shift(lag)
    
    # Recent crossover history
    for lookback in [5, 10]:
        data[f'Cross_Above_Last_{lookback}'] = data['EMA_Cross_Above'].rolling(lookback).sum()
        data[f'Cross_Below_Last_{lookback}'] = data['EMA_Cross_Below'].rolling(lookback).sum()
    
    # =================================================================
    # SWING HIGH/LOW FEATURES (for stop loss calculation)
    # =================================================================
    print("  Creating Swing High/Low Features...")
    
    # Calculate swing highs and lows
    for periods in [5, 10, 20]:
        data[f'Swing_High_{periods}'] = data['high'].rolling(periods).max()
        data[f'Swing_Low_{periods}'] = data['low'].rolling(periods).min()
    
    # Distance to swing levels
    data['Distance_to_Swing_High'] = (data['Swing_High_10'] - data['close']) / data['close']
    data['Distance_to_Swing_Low'] = (data['close'] - data['Swing_Low_10']) / data['close']
    
    # =================================================================
    # TIME-BASED FEATURES
    # =================================================================
    print("  Creating Time-based Features...")
    
    if 'datetime' in data.columns:
        data['Hour'] = data['datetime'].dt.hour
        data['Minute'] = data['datetime'].dt.minute
        data['Time_Slot'] = data['Hour'] * 60 + data['Minute']
        
        # Trading session flags
        data['Morning_Session'] = ((data['Hour'] >= 9) & (data['Hour'] < 12)).astype(int)
        data['Afternoon_Session'] = ((data['Hour'] >= 12) & (data['Hour'] < 15)).astype(int)
        data['Best_Hours'] = ((data['Hour'] >= 10) & (data['Hour'] < 14)).astype(int)
    
    # =================================================================
    # COMPOSITE SIGNALS
    # =================================================================
    print("  Creating Composite Signals...")
    
    # EMA Crossover Signals
    data['Bullish_Cross_Signal'] = (
        (data['EMA_Cross_Above'] == 1) & 
        (data['EMA_30_Slope_5'] > 0) & 
        (data['Volume_Spike'] > 1.1)
    ).astype(int)
    
    data['Bearish_Cross_Signal'] = (
        (data['EMA_Cross_Below'] == 1) & 
        (data['EMA_30_Slope_5'] < 0) & 
        (data['Volume_Spike'] > 1.1)
    ).astype(int)
    
    # Retest Signals
    data['Bullish_Retest_Signal'] = (
        (data['Price_Position_Flag'] == 0) &  # Below both EMAs
        (data['Near_EMA30'] == 1) &  # Near 30 EMA
        (data['Green_Candle'] == 1) &  # Strong bullish candle
        (data['Strong_Candle'] == 1)
    ).astype(int)
    
    data['Bearish_Retest_Signal'] = (
        (data['Price_Position_Flag'] == 2) &  # Above both EMAs
        (data['Near_EMA30'] == 1) &  # Near 30 EMA
        (data['Red_Candle'] == 1) &  # Strong bearish candle
        (data['Strong_Candle'] == 1)
    ).astype(int)
    
    # =================================================================
    # CLEAN UP AND RETURN
    # =================================================================
    print("  Cleaning up features...")
    
    # Remove rows with NaN values (from indicators)
    initial_rows = len(data)
    data = data.dropna()
    final_rows = len(data)
    
    print(f"  Removed {initial_rows - final_rows} rows with NaN values")
    print(f"  Final dataset: {final_rows} rows")
    
    # Get feature columns (exclude OHLCV and datetime)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'datetime', 'timestamp']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    print(f"  Created {len(feature_cols)} features")
    
    return data


def get_feature_names() -> list:
    """Return list of all feature names for reference"""
    
    features = [
        # Core EMA Dynamic Features
        'EMA_8', 'EMA_30', 'EMA_Spread', 'EMA_Spread_ROC',
        'EMA_8_Slope_3', 'EMA_8_Slope_5', 'EMA_30_Slope_3', 'EMA_30_Slope_5',
        'EMA_8_Above_30', 'EMA_Cross_Above', 'EMA_Cross_Below',
        
        # Price-to-EMA Relationship
        'Price_Distance_EMA8', 'Price_Distance_EMA30', 'Price_Position_Flag',
        'Close_Below_EMA8', 'Close_Below_EMA30', 'Close_Above_EMA8', 'Close_Above_EMA30',
        'Near_EMA8', 'Near_EMA30',
        
        # Price Action & Volatility
        'Candle_Body_Size', 'Candle_Range', 'Wick_to_Body_Ratio',
        'Green_Candle', 'Red_Candle', 'Strong_Candle',
        'Price_Returns', 'Rolling_Volatility',
        
        # Volume & Conviction
        'Volume_MA20', 'Volume_Spike', 'Volume_ROC', 'Volume_Price_Trend',
        
        # Historical Context
        'EMA_Spread_Lag_1', 'EMA_Spread_Lag_2', 'EMA_Spread_Lag_3',
        'Price_Distance_EMA30_Lag_1', 'Price_Distance_EMA30_Lag_2', 'Price_Distance_EMA30_Lag_3',
        'Cross_Above_Last_5', 'Cross_Above_Last_10', 'Cross_Below_Last_5', 'Cross_Below_Last_10',
        
        # Swing Levels
        'Swing_High_5', 'Swing_High_10', 'Swing_High_20',
        'Swing_Low_5', 'Swing_Low_10', 'Swing_Low_20',
        'Distance_to_Swing_High', 'Distance_to_Swing_Low',
        
        # Time Features
        'Hour', 'Minute', 'Time_Slot', 'Morning_Session', 'Afternoon_Session', 'Best_Hours',
        
        # Composite Signals
        'Bullish_Cross_Signal', 'Bearish_Cross_Signal',
        'Bullish_Retest_Signal', 'Bearish_Retest_Signal'
    ]
    
    return features


if __name__ == "__main__":
    # Test the feature creation
    print("Testing EMA Crossover Feature Creation...")
    
    # Load test data
    try:
        df = pd.read_csv("testing data/reliance_data_5min_full_year_testing.csv")
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"Loaded {len(df)} rows of test data")
        
        # Create features
        featured_df = create_ema_crossover_features(df)
        
        # Display results
        print(f"\nFeature creation complete!")
        print(f"Original columns: {len(df.columns)}")
        print(f"Final columns: {len(featured_df.columns)}")
        print(f"Features created: {len(featured_df.columns) - len(df.columns)}")
        
        # Show sample of features
        feature_cols = get_feature_names()
        available_features = [col for col in feature_cols if col in featured_df.columns]
        
        print(f"\nSample of features:")
        print(featured_df[available_features[:10]].head())
        
        # Save sample
        featured_df.to_csv('ema_crossover_features_sample.csv', index=False)
        print(f"\nSaved sample to ema_crossover_features_sample.csv")
        
    except FileNotFoundError:
        print("Test data file not found. Please ensure the data file exists.")
    except Exception as e:
        print(f"Error: {e}")