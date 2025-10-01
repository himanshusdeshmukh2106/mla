#!/usr/bin/env python3
"""
Complete EMA Trap Strategy Implementation Example
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

def create_realistic_data():
    """Create more realistic 5-minute OHLCV data for EMA trap testing"""
    
    # Create multiple trading days
    start_date = datetime(2024, 1, 15)
    end_date = datetime(2024, 1, 25)
    
    all_data = []
    
    for day_offset in range((end_date - start_date).days):
        current_date = start_date + timedelta(days=day_offset)
        
        # Skip weekends
        if current_date.weekday() >= 5:
            continue
            
        # Create trading hours: 9:15 AM to 3:30 PM
        start_time = current_date.replace(hour=9, minute=15)
        end_time = current_date.replace(hour=15, minute=30)
        
        # Generate 5-minute intervals for this day
        time_range = pd.date_range(start=start_time, end=end_time, freq='5min')
        
        # Base price for this day (with some daily variation)
        base_price = 100.0 + day_offset * 0.5 + np.random.normal(0, 2)
        
        # Generate intraday price movements
        n_periods = len(time_range)
        price_changes = np.random.normal(0, 0.001, n_periods)  # 0.1% std dev
        
        # Add some EMA trap patterns during entry windows
        for i, timestamp in enumerate(time_range):
            hour = timestamp.hour
            minute = timestamp.minute
            
            # Add trap patterns during entry windows with some probability
            if (hour == 9 and 15 <= minute <= 30) or (hour == 10 and 0 <= minute <= 60):
                if np.random.random() < 0.1:  # 10% chance of trap pattern
                    if i < n_periods - 3:
                        if np.random.random() < 0.5:  # Bearish trap
                            price_changes[i] = 0.003     # Break above
                            price_changes[i+1] = 0.001   # Continue up
                            price_changes[i+2] = -0.005  # Fall back (trap)
                        else:  # Bullish trap
                            price_changes[i] = -0.003    # Break below
                            price_changes[i+1] = -0.001  # Continue down
                            price_changes[i+2] = 0.005   # Rise back (trap)
        
        # Calculate cumulative prices for this day
        cumulative_returns = np.cumsum(price_changes)
        close_prices = base_price * (1 + cumulative_returns)
        
        # Generate OHLCV data for this day
        for i, (timestamp, close) in enumerate(zip(time_range, close_prices)):
            open_price = close_prices[i-1] if i > 0 else close
            
            # Generate realistic OHLC with controlled volatility
            volatility = 0.0005  # 0.05% intraday volatility
            high = max(open_price, close) + np.random.uniform(0, volatility * close)
            low = min(open_price, close) - np.random.uniform(0, volatility * close)
            volume = np.random.randint(50000, 200000)
            
            all_data.append({
                'datetime': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
    
    df = pd.DataFrame(all_data)
    df.set_index('datetime', inplace=True)
    
    return df

def run_ema_trap_strategy():
    """Run complete EMA trap strategy example"""
    
    print("EMA Trap Trading Strategy - Complete Example")
    print("=" * 60)
    
    # 1. Create realistic market data
    print("1. Creating realistic 5-minute market data...")
    data = create_realistic_data()
    print(f"   Created {len(data)} data points across {data.index.date[0]} to {data.index.date[-1]}")
    
    # 2. Load configuration and initialize feature engineer
    print("\n2. Initializing feature engineering...")
    config_manager = ConfigManager()
    feature_config = config_manager.get_config('feature')
    
    engineer = FeatureEngineer(feature_config)
    
    # 3. Create features
    print("   Creating EMA trap features...")
    features_df = engineer.create_features(data)
    print(f"   Features created: {len(features_df)} rows, {len(features_df.columns)} columns")
    
    # 4. Generate targets
    print("\n3. Generating trading targets...")
    target_config = {
        'method': 'ema_trap',
        'lookahead_periods': 2,  # Look ahead 2 candles (10 minutes)
        'profit_threshold': 0.003,  # 0.3% profit target
        'loss_threshold': -0.003    # -0.3% loss threshold
    }
    
    target_generator = TargetGenerator(target_config)
    targets_df = target_generator.generate_targets(features_df)
    
    # 5. Analyze entry signals
    print("\n4. Analyzing entry signals...")
    
    # Strict entry conditions
    strict_bearish = targets_df[
        (targets_df['Bearish_Trap_Confirmed'] == 1) &
        (targets_df['In_Entry_Window'] == 1) &
        (targets_df['ADX_In_Range'] == 1) &
        (targets_df['Small_Candle'] == 1)
    ]
    
    strict_bullish = targets_df[
        (targets_df['Bullish_Trap_Confirmed'] == 1) &
        (targets_df['In_Entry_Window'] == 1) &
        (targets_df['ADX_In_Range'] == 1) &
        (targets_df['Small_Candle'] == 1)
    ]
    
    print(f"   Strict conditions - Bearish entries: {len(strict_bearish)}, Bullish entries: {len(strict_bullish)}")
    
    # Relaxed entry conditions (for more signals)
    relaxed_bearish = targets_df[
        (targets_df['Bearish_Trap_Confirmed'] == 1) &
        (targets_df['ADX_In_Range'] == 1) &
        (targets_df['Candle_Body_Size_Pct'] <= 0.5)  # Relaxed candle size
    ]
    
    relaxed_bullish = targets_df[
        (targets_df['Bullish_Trap_Confirmed'] == 1) &
        (targets_df['ADX_In_Range'] == 1) &
        (targets_df['Candle_Body_Size_Pct'] <= 0.5)  # Relaxed candle size
    ]
    
    print(f"   Relaxed conditions - Bearish entries: {len(relaxed_bearish)}, Bullish entries: {len(relaxed_bullish)}")
    
    # 6. Show sample entry signals
    if len(relaxed_bearish) > 0 or len(relaxed_bullish) > 0:
        print("\n5. Sample Entry Signals:")
        print("-" * 40)
        
        if len(relaxed_bearish) > 0:
            print("Bearish Entry Signals (Short):")
            cols = ['close', 'EMA_21', 'ADX', 'Candle_Body_Size_Pct', 'future_return']
            sample_bearish = relaxed_bearish[cols].head(3)
            print(sample_bearish.round(4))
            
        if len(relaxed_bullish) > 0:
            print("\nBullish Entry Signals (Long):")
            cols = ['close', 'EMA_21', 'ADX', 'Candle_Body_Size_Pct', 'future_return']
            sample_bullish = relaxed_bullish[cols].head(3)
            print(sample_bullish.round(4))
    
    # 7. Calculate strategy performance metrics
    print("\n6. Strategy Performance Analysis:")
    print("-" * 40)
    
    if len(relaxed_bearish) > 0:
        bearish_success = (relaxed_bearish['future_return'] <= -0.003).sum()
        bearish_success_rate = bearish_success / len(relaxed_bearish) * 100
        avg_bearish_return = relaxed_bearish['future_return'].mean() * 100
        print(f"Bearish signals: {len(relaxed_bearish)}, Success rate: {bearish_success_rate:.1f}%, Avg return: {avg_bearish_return:.2f}%")
    
    if len(relaxed_bullish) > 0:
        bullish_success = (relaxed_bullish['future_return'] >= 0.003).sum()
        bullish_success_rate = bullish_success / len(relaxed_bullish) * 100
        avg_bullish_return = relaxed_bullish['future_return'].mean() * 100
        print(f"Bullish signals: {len(relaxed_bullish)}, Success rate: {bullish_success_rate:.1f}%, Avg return: {avg_bullish_return:.2f}%")
    
    # 8. Feature importance for ML
    print("\n7. Key Features for Machine Learning:")
    print("-" * 40)
    
    key_features = [
        'EMA_21', 'ADX', 'Distance_From_EMA21_Pct',
        'Bearish_Trap_Confirmed', 'Bullish_Trap_Confirmed',
        'In_Entry_Window', 'ADX_In_Range', 'Candle_Body_Size_Pct',
        'RSI', 'MACD', 'BB_Position', 'Volume_Ratio_20'
    ]
    
    print("Selected features for model training:")
    for i, feature in enumerate(key_features, 1):
        print(f"{i:2d}. {feature}")
    
    # 9. Save results
    print("\n8. Saving Results:")
    print("-" * 40)
    
    # Save feature data
    targets_df.to_csv('ema_trap_strategy_data.csv')
    print("   Full dataset saved to 'ema_trap_strategy_data.csv'")
    
    # Save entry signals
    if len(relaxed_bearish) > 0 or len(relaxed_bullish) > 0:
        entry_signals = pd.concat([relaxed_bearish, relaxed_bullish])
        entry_signals.to_csv('ema_trap_entry_signals.csv')
        print("   Entry signals saved to 'ema_trap_entry_signals.csv'")
    
    print(f"\n✅ EMA Trap Strategy analysis complete!")
    print(f"   Dataset ready for XGBoost model training with {len(targets_df)} samples")
    
    return targets_df

if __name__ == "__main__":
    try:
        results = run_ema_trap_strategy()
        
    except Exception as e:
        print(f"❌ Strategy analysis failed: {e}")
        import traceback
        traceback.print_exc()