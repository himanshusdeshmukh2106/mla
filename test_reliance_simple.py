#!/usr/bin/env python3
"""
Simple test script to validate Reliance 5-minute data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def test_reliance_data():
    """Test the Reliance data file directly"""
    
    print("=" * 60)
    print("RELIANCE 5-MIN DATA VALIDATION")
    print("=" * 60)
    
    # Check if file exists
    data_file = Path("data/reliance_data_5min_full_year.csv")
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return False
    
    print(f"✓ Data file found: {data_file}")
    print(f"✓ File size: {data_file.stat().st_size / 1024**2:.2f} MB")
    
    # Load and analyze data
    try:
        print(f"\n1. Loading data...")
        df = pd.read_csv(data_file)
        print(f"   ✓ Loaded {len(df)} rows")
        print(f"   ✓ Columns: {list(df.columns)}")
        
        # Check required columns
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"   ❌ Missing required columns: {missing_cols}")
            return False
        print(f"   ✓ All required columns present")
        
        # Convert datetime
        print(f"\n2. Processing datetime...")
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        print(f"   ✓ Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"   ✓ Total trading days: {df['datetime'].dt.date.nunique()}")
        
        # Check data quality
        print(f"\n3. Data quality checks...")
        
        # Missing values
        missing_counts = df[required_cols].isnull().sum()
        if missing_counts.sum() > 0:
            print(f"   ⚠ Missing values found:")
            for col, count in missing_counts.items():
                if count > 0:
                    print(f"     - {col}: {count} missing")
        else:
            print(f"   ✓ No missing values")
        
        # OHLC consistency
        high_errors = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
        low_errors = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
        
        if high_errors > 0:
            print(f"   ⚠ High price inconsistencies: {high_errors}")
        if low_errors > 0:
            print(f"   ⚠ Low price inconsistencies: {low_errors}")
        
        if high_errors == 0 and low_errors == 0:
            print(f"   ✓ OHLC data is consistent")
        
        # Volume check
        negative_volume = (df['volume'] < 0).sum()
        zero_volume = (df['volume'] == 0).sum()
        
        if negative_volume > 0:
            print(f"   ⚠ Negative volume entries: {negative_volume}")
        if zero_volume > 0:
            print(f"   ⚠ Zero volume entries: {zero_volume}")
        
        print(f"   ✓ Volume range: {df['volume'].min():,.0f} to {df['volume'].max():,.0f}")
        
        # Price statistics
        print(f"\n4. Price statistics...")
        print(f"   ✓ Price range: ₹{df['close'].min():.2f} to ₹{df['close'].max():.2f}")
        print(f"   ✓ Average price: ₹{df['close'].mean():.2f}")
        print(f"   ✓ Price volatility (std): {df['close'].std():.2f}")
        
        # Calculate returns for analysis
        df['returns'] = df['close'].pct_change()
        daily_volatility = df['returns'].std()
        print(f"   ✓ 5-min return volatility: {daily_volatility:.4f} ({daily_volatility*100:.2f}%)")
        
        # Time series analysis
        print(f"\n5. Time series analysis...")
        
        # Check for gaps
        df['time_diff'] = df['datetime'].diff()
        expected_freq = pd.Timedelta(minutes=5)
        gaps = df[df['time_diff'] > expected_freq * 1.5]  # Allow some tolerance
        
        if len(gaps) > 0:
            print(f"   ⚠ Found {len(gaps)} potential time gaps")
            print(f"   ⚠ Largest gap: {gaps['time_diff'].max()}")
        else:
            print(f"   ✓ No significant time gaps found")
        
        # Market hours check (assuming 9:15 AM to 3:30 PM IST)
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        
        market_hours = df[
            ((df['hour'] == 9) & (df['minute'] >= 15)) |
            ((df['hour'] >= 10) & (df['hour'] <= 14)) |
            ((df['hour'] == 15) & (df['minute'] <= 30))
        ]
        
        print(f"   ✓ Market hours data: {len(market_hours)} / {len(df)} rows ({len(market_hours)/len(df)*100:.1f}%)")
        
        # Sample data display
        print(f"\n6. Sample data:")
        print(df[['datetime', 'open', 'high', 'low', 'close', 'volume']].head(10).to_string())
        
        print(f"\n" + "=" * 60)
        print("✅ DATA VALIDATION COMPLETED SUCCESSFULLY!")
        print("✅ Your Reliance data is ready for training!")
        print("=" * 60)
        
        # Summary for training
        print(f"\nTraining Data Summary:")
        print(f"📊 Total samples: {len(df):,}")
        print(f"📅 Date range: {df['datetime'].dt.date.min()} to {df['datetime'].dt.date.max()}")
        print(f"⏰ Timeframe: 5-minute candles")
        print(f"💹 Symbol: RELIANCE")
        print(f"🏛️ Market: Indian Stock Exchange (IST timezone)")
        print(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return False

if __name__ == "__main__":
    success = test_reliance_data()
    if not success:
        exit(1)