#!/usr/bin/env python3
"""
Check overlap between trap confirmations and entry windows
"""

import pandas as pd
import sys
sys.path.append('src')

# Load the generated features
df = pd.read_csv('sample_ema_trap_features.csv', index_col=0, parse_dates=True)

# Check trap confirmations during entry windows
bearish_in_window = df[
    (df['Bearish_Trap_Confirmed'] == 1) & 
    (df['In_Entry_Window'] == 1)
]

bullish_in_window = df[
    (df['Bullish_Trap_Confirmed'] == 1) & 
    (df['In_Entry_Window'] == 1)
]

print(f"Bearish trap confirmations in entry windows: {len(bearish_in_window)}")
print(f"Bullish trap confirmations in entry windows: {len(bullish_in_window)}")

# Check ADX condition for those in windows
if len(bearish_in_window) > 0:
    bearish_with_adx = bearish_in_window[bearish_in_window['ADX_In_Range'] == 1]
    print(f"Bearish traps in window with ADX 20-36: {len(bearish_with_adx)}")
    
    if len(bearish_with_adx) > 0:
        bearish_with_small_candle = bearish_with_adx[bearish_with_adx['Small_Candle'] == 1]
        print(f"Bearish traps with all conditions: {len(bearish_with_small_candle)}")

if len(bullish_in_window) > 0:
    bullish_with_adx = bullish_in_window[bullish_in_window['ADX_In_Range'] == 1]
    print(f"Bullish traps in window with ADX 20-36: {len(bullish_with_adx)}")
    
    if len(bullish_with_adx) > 0:
        bullish_with_small_candle = bullish_with_adx[bullish_with_adx['Small_Candle'] == 1]
        print(f"Bullish traps with all conditions: {len(bullish_with_small_candle)}")

# Let's relax the time window to see more signals
print("\n" + "="*50)
print("RELAXED CONDITIONS (any time between 9:15-15:30)")

# Create a broader time window for testing
df['Broad_Entry_Window'] = ((df['Hour'] >= 9) & (df['Hour'] <= 15)).astype(int)

bearish_broad = df[
    (df['Bearish_Trap_Confirmed'] == 1) &
    (df['Broad_Entry_Window'] == 1) &
    (df['ADX_In_Range'] == 1) &
    (df['Small_Candle'] == 1)
]

bullish_broad = df[
    (df['Bullish_Trap_Confirmed'] == 1) &
    (df['Broad_Entry_Window'] == 1) &
    (df['ADX_In_Range'] == 1) &
    (df['Small_Candle'] == 1)
]

print(f"Bearish entries (broad window): {len(bearish_broad)}")
print(f"Bullish entries (broad window): {len(bullish_broad)}")

if len(bearish_broad) > 0:
    print("\nSample bearish entries:")
    cols = ['close', 'EMA_21', 'ADX', 'Candle_Body_Size_Pct', 'Hour', 'Minute']
    print(bearish_broad[cols].head())

if len(bullish_broad) > 0:
    print("\nSample bullish entries:")
    cols = ['close', 'EMA_21', 'ADX', 'Candle_Body_Size_Pct', 'Hour', 'Minute']
    print(bullish_broad[cols].head())