#!/usr/bin/env python3
"""
Diagnostic script to analyze EMA trap conditions and suggest improvements
"""

import pandas as pd
import numpy as np
import sys
import os

# Setup environment
sys.path.insert(0, 'src')

def load_and_analyze_data():
    """Load data and analyze EMA trap conditions"""
    
    print("🔍 Diagnosing EMA Trap Conditions")
    print("=" * 50)
    
    # Load data
    data_path = "data/reliance_data_5min_full_year.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    print(f"📊 Loaded {len(df)} data points")
    
    # Create basic features for analysis
    import pandas_ta as ta
    
    # EMA and ADX
    df['EMA_21'] = ta.ema(df['close'], length=21)
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    
    # EMA crosses
    df['Price_Above_EMA21'] = (df['close'] > df['EMA_21']).astype(int)
    df['EMA21_Cross_Above'] = ((df['close'].shift(1) <= df['EMA_21'].shift(1)) & 
                               (df['close'] > df['EMA_21'])).astype(int)
    df['EMA21_Cross_Below'] = ((df['close'].shift(1) >= df['EMA_21'].shift(1)) & 
                               (df['close'] < df['EMA_21'])).astype(int)
    
    # Time features
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    df['Entry_Window_1'] = ((df['Hour'] == 9) & (df['Minute'].between(15, 30))).astype(int)
    df['Entry_Window_2'] = ((df['Hour'] == 10) & (df['Minute'] <= 60)).astype(int)
    df['In_Entry_Window'] = (df['Entry_Window_1'] | df['Entry_Window_2']).astype(int)
    
    # ADX condition
    df['ADX_In_Range'] = ((df['ADX'] >= 20) & (df['ADX'] <= 36)).astype(int)
    
    # Candle size
    df['Candle_Body_Size_Pct'] = abs(df['close'] - df['open']) / df['open'] * 100
    df['Small_Candle'] = (df['Candle_Body_Size_Pct'] <= 0.20).astype(int)
    
    # Trap detection (simplified)
    df['Bearish_Trap_Setup'] = df['EMA21_Cross_Above']
    df['Bullish_Trap_Setup'] = df['EMA21_Cross_Below']
    
    # Simple trap confirmation (next candle crosses back)
    df['Bearish_Trap_Confirmed'] = ((df['Bearish_Trap_Setup'].shift(1) == 1) & 
                                    (df['EMA21_Cross_Below'] == 1)).astype(int)
    df['Bullish_Trap_Confirmed'] = ((df['Bullish_Trap_Setup'].shift(1) == 1) & 
                                    (df['EMA21_Cross_Above'] == 1)).astype(int)
    
    # Future returns
    df['future_return_1'] = df['close'].shift(-1) / df['close'] - 1
    df['future_return_2'] = df['close'].shift(-2) / df['close'] - 1
    df['future_return_5'] = df['close'].shift(-5) / df['close'] - 1
    
    # Remove NaN
    df.dropna(inplace=True)
    
    print(f"📊 After feature creation: {len(df)} data points")
    
    # Analyze each condition
    print("\n🔍 CONDITION ANALYSIS:")
    print("-" * 30)
    
    # 1. EMA crosses
    bearish_setups = df['Bearish_Trap_Setup'].sum()
    bullish_setups = df['Bullish_Trap_Setup'].sum()
    print(f"1. EMA Crosses:")
    print(f"   Bearish setups (cross above): {bearish_setups}")
    print(f"   Bullish setups (cross below): {bullish_setups}")
    
    # 2. Trap confirmations
    bearish_confirmed = df['Bearish_Trap_Confirmed'].sum()
    bullish_confirmed = df['Bullish_Trap_Confirmed'].sum()
    print(f"2. Trap Confirmations:")
    print(f"   Bearish confirmed: {bearish_confirmed}")
    print(f"   Bullish confirmed: {bullish_confirmed}")
    
    # 3. Time windows
    in_window = df['In_Entry_Window'].sum()
    window1 = df['Entry_Window_1'].sum()
    window2 = df['Entry_Window_2'].sum()
    print(f"3. Time Windows:")
    print(f"   Total in windows: {in_window}")
    print(f"   Window 1 (9:15-9:30): {window1}")
    print(f"   Window 2 (10:00-11:00): {window2}")
    
    # 4. ADX condition
    adx_in_range = df['ADX_In_Range'].sum()
    adx_mean = df['ADX'].mean()
    adx_std = df['ADX'].std()
    print(f"4. ADX Condition (20-36):")
    print(f"   Samples in range: {adx_in_range} ({adx_in_range/len(df)*100:.1f}%)")
    print(f"   ADX mean: {adx_mean:.1f}, std: {adx_std:.1f}")
    
    # 5. Candle size
    small_candles = df['Small_Candle'].sum()
    candle_size_mean = df['Candle_Body_Size_Pct'].mean()
    candle_size_median = df['Candle_Body_Size_Pct'].median()
    print(f"5. Small Candle (<=0.20%):")
    print(f"   Small candles: {small_candles} ({small_candles/len(df)*100:.1f}%)")
    print(f"   Body size mean: {candle_size_mean:.3f}%, median: {candle_size_median:.3f}%")
    
    # Combined conditions
    print(f"\n🎯 COMBINED CONDITIONS:")
    print("-" * 30)
    
    # Bearish entries
    bearish_entries = df[
        (df['Bearish_Trap_Confirmed'] == 1) &
        (df['In_Entry_Window'] == 1) &
        (df['ADX_In_Range'] == 1) &
        (df['Small_Candle'] == 1)
    ]
    
    # Bullish entries
    bullish_entries = df[
        (df['Bullish_Trap_Confirmed'] == 1) &
        (df['In_Entry_Window'] == 1) &
        (df['ADX_In_Range'] == 1) &
        (df['Small_Candle'] == 1)
    ]
    
    print(f"Bearish entries (all conditions): {len(bearish_entries)}")
    print(f"Bullish entries (all conditions): {len(bullish_entries)}")
    print(f"Total entries: {len(bearish_entries) + len(bullish_entries)}")
    
    # Analyze profitability with different thresholds
    print(f"\n💰 PROFITABILITY ANALYSIS:")
    print("-" * 30)
    
    all_entries = pd.concat([bearish_entries, bullish_entries])
    if len(all_entries) > 0:
        for threshold in [0.001, 0.002, 0.003, 0.004, 0.005]:
            profitable_1 = (abs(all_entries['future_return_1']) >= threshold).sum()
            profitable_2 = (abs(all_entries['future_return_2']) >= threshold).sum()
            profitable_5 = (abs(all_entries['future_return_5']) >= threshold).sum()
            
            print(f"Threshold {threshold:.1%}:")
            print(f"  1-candle: {profitable_1}/{len(all_entries)} ({profitable_1/len(all_entries)*100:.1f}%)")
            print(f"  2-candle: {profitable_2}/{len(all_entries)} ({profitable_2/len(all_entries)*100:.1f}%)")
            print(f"  5-candle: {profitable_5}/{len(all_entries)} ({profitable_5/len(all_entries)*100:.1f}%)")
    
    # Suggestions
    print(f"\n💡 SUGGESTIONS:")
    print("-" * 30)
    
    if small_candles / len(df) < 0.1:
        print(f"1. 🔧 Relax candle size: Only {small_candles/len(df)*100:.1f}% are <=0.20%")
        print(f"   Try: <=0.5% or <=1.0%")
    
    if adx_in_range / len(df) < 0.5:
        print(f"2. 🔧 Adjust ADX range: Only {adx_in_range/len(df)*100:.1f}% in 20-36 range")
        print(f"   Try: 15-40 or 10-50")
    
    if bearish_confirmed + bullish_confirmed < 100:
        print(f"3. 🔧 Simplify trap detection: Only {bearish_confirmed + bullish_confirmed} confirmations")
        print(f"   Try: Allow 2-3 candle lookback for confirmation")
    
    if in_window / len(df) < 0.1:
        print(f"4. 🔧 Expand time windows: Only {in_window/len(df)*100:.1f}% in entry windows")
        print(f"   Try: 9:15-10:00 and 10:00-12:00")
    
    print(f"\n🎯 RECOMMENDED RELAXED CONDITIONS:")
    print("-" * 40)
    print(f"1. Candle size: <= 0.5% (instead of 0.20%)")
    print(f"2. ADX range: 15-40 (instead of 20-36)")
    print(f"3. Time windows: 9:15-10:00 and 10:00-12:00")
    print(f"4. Profit threshold: 0.2% (instead of 0.4%)")
    print(f"5. Lookback: 3 candles for trap confirmation")

if __name__ == "__main__":
    load_and_analyze_data()