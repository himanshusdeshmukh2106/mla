#!/usr/bin/env python3
"""
Debug script for EMA trap feature engineering
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('src')

# Load the generated features
df = pd.read_csv('sample_ema_trap_features.csv', index_col=0, parse_dates=True)

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Check trap signals
print("\nTrap Signal Summary:")
print(f"Bearish_Trap_Setup: {df['Bearish_Trap_Setup'].sum()}")
print(f"Bullish_Trap_Setup: {df['Bullish_Trap_Setup'].sum()}")
print(f"Bearish_Trap_Confirmed: {df['Bearish_Trap_Confirmed'].sum()}")
print(f"Bullish_Trap_Confirmed: {df['Bullish_Trap_Confirmed'].sum()}")

# Check entry conditions
print(f"\nEntry Conditions:")
print(f"In_Entry_Window: {df['In_Entry_Window'].sum()}")
print(f"ADX_In_Range: {df['ADX_In_Range'].sum()}")
print(f"Small_Candle: {df['Small_Candle'].sum()}")

# Find rows with trap confirmations
bearish_confirmed = df[df['Bearish_Trap_Confirmed'] == 1]
bullish_confirmed = df[df['Bullish_Trap_Confirmed'] == 1]

print(f"\nBearish trap confirmations: {len(bearish_confirmed)}")
if len(bearish_confirmed) > 0:
    print("Sample bearish confirmations:")
    cols = ['close', 'EMA_21', 'In_Entry_Window', 'ADX_In_Range', 'Small_Candle', 'Hour', 'Minute']
    print(bearish_confirmed[cols].head())

print(f"\nBullish trap confirmations: {len(bullish_confirmed)}")
if len(bullish_confirmed) > 0:
    print("Sample bullish confirmations:")
    cols = ['close', 'EMA_21', 'In_Entry_Window', 'ADX_In_Range', 'Small_Candle', 'Hour', 'Minute']
    print(bullish_confirmed[cols].head())

# Check combined entry conditions
bearish_entries = df[
    (df['Bearish_Trap_Confirmed'] == 1) &
    (df['In_Entry_Window'] == 1) &
    (df['ADX_In_Range'] == 1) &
    (df['Small_Candle'] == 1)
]

bullish_entries = df[
    (df['Bullish_Trap_Confirmed'] == 1) &
    (df['In_Entry_Window'] == 1) &
    (df['ADX_In_Range'] == 1) &
    (df['Small_Candle'] == 1)
]

print(f"\nCombined entry conditions:")
print(f"Bearish entries: {len(bearish_entries)}")
print(f"Bullish entries: {len(bullish_entries)}")

# Check time windows
print(f"\nTime analysis:")
print(f"Entry Window 1 (9:15-9:30): {df['Entry_Window_1'].sum()}")
print(f"Entry Window 2 (10:00-11:00): {df['Entry_Window_2'].sum()}")

# Show some sample times
sample_times = df[df['In_Entry_Window'] == 1][['Hour', 'Minute', 'Time_Minutes']].head(10)
print(f"\nSample entry window times:")
print(sample_times)