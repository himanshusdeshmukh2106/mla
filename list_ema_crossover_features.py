"""
List all 78 features used in EMA Crossover model
"""

# Complete list of all 78 features used for training

features = [
    # ===== CORE EMA FEATURES (11) =====
    'EMA_8',
    'EMA_30',
    'EMA_Spread',
    'EMA_Spread_ROC',
    'EMA_8_Slope_3',
    'EMA_8_Slope_5',
    'EMA_8_Slope_10',
    'EMA_30_Slope_3',
    'EMA_30_Slope_5',
    'EMA_30_Slope_10',
    'EMA_8_Above_30',
    
    # ===== EMA ACCELERATION (2) =====
    'EMA_8_Acceleration',
    'EMA_30_Acceleration',
    
    # ===== CROSSOVER FEATURES (2) =====
    'EMA_Cross_Above',
    'EMA_Cross_Below',
    
    # ===== PRICE-EMA RELATIONSHIP (5) =====
    'Price_Distance_EMA8',
    'Price_Distance_EMA30',
    'Price_Distance_EMA8_ROC',
    'Price_Distance_EMA30_ROC',
    'Price_Position_Flag',
    
    # ===== PRICE ACTION (10) =====
    'Candle_Body_Size',
    'Candle_Range',
    'Wick_to_Body_Ratio',
    'Upper_Wick',
    'Lower_Wick',
    'Green_Candle',
    'Red_Candle',
    'Strong_Candle_10',
    'Strong_Candle_20',
    'Strong_Candle_50',
    
    # ===== VOLATILITY (3) =====
    'Price_Returns',
    'Rolling_Volatility',
    'Volatility_Ratio',
    
    # ===== VOLUME (6) =====
    'Volume_MA20',
    'Volume_MA50',
    'Volume_Spike',
    'Volume_ROC',
    'Volume_Trend',
    'Volume_Price_Corr',
    
    # ===== HISTORICAL CONTEXT - LAG FEATURES (16) =====
    'EMA_Spread_Lag_1',
    'EMA_Spread_Lag_2',
    'EMA_Spread_Lag_3',
    'EMA_Spread_Lag_5',
    'Price_Distance_EMA30_Lag_1',
    'Price_Distance_EMA30_Lag_2',
    'Price_Distance_EMA30_Lag_3',
    'Price_Distance_EMA30_Lag_5',
    'Returns_Lag_1',
    'Returns_Lag_2',
    'Returns_Lag_3',
    'Returns_Lag_5',
    'Cross_Above_Last_5',
    'Cross_Above_Last_10',
    'Cross_Above_Last_20',
    'Cross_Below_Last_5',
    
    # ===== CROSSOVER HISTORY (2) =====
    'Cross_Below_Last_10',
    'Cross_Below_Last_20',
    
    # ===== SWING LEVELS (18) =====
    'Swing_High_5',
    'Swing_High_10',
    'Swing_High_20',
    'Swing_Low_5',
    'Swing_Low_10',
    'Swing_Low_20',
    'Distance_to_Swing_High_5',
    'Distance_to_Swing_High_10',
    'Distance_to_Swing_High_20',
    'Distance_to_Swing_Low_5',
    'Distance_to_Swing_Low_10',
    'Distance_to_Swing_Low_20',
    
    # ===== TIME FEATURES (6) =====
    'Hour',
    'Minute',
    'Time_Slot',
    'Best_Hours',
    'Opening_Hour',
    'Closing_Hour',
    
    # ===== PATTERN RECOGNITION (3) =====
    'Doji',
    'Hammer',
    'Shooting_Star',
]

# Print organized list
print("="*80)
print("COMPLETE LIST OF 78 FEATURES")
print("="*80)

categories = {
    'Core EMA Features': features[0:11],
    'EMA Acceleration': features[11:13],
    'Crossover Features': features[13:15],
    'Price-EMA Relationship': features[15:20],
    'Price Action': features[20:30],
    'Volatility': features[30:33],
    'Volume': features[33:39],
    'Historical Context (Lags)': features[39:55],
    'Crossover History': features[55:57],
    'Swing Levels': features[57:69],
    'Time Features': features[69:75],
    'Pattern Recognition': features[75:78],
}

for category, feature_list in categories.items():
    print(f"\n{category} ({len(feature_list)} features):")
    for i, feature in enumerate(feature_list, 1):
        print(f"  {i}. {feature}")

print(f"\n{'='*80}")
print(f"TOTAL: {len(features)} features")
print(f"{'='*80}")

# Verify count
assert len(features) == 78, f"Expected 78 features, got {len(features)}"
print("\n✅ Feature count verified: 78 features")

# Save to file
with open('ema_crossover_features_list.txt', 'w') as f:
    f.write("EMA CROSSOVER MODEL - COMPLETE FEATURE LIST\n")
    f.write("="*80 + "\n\n")
    
    for category, feature_list in categories.items():
        f.write(f"\n{category} ({len(feature_list)} features):\n")
        for i, feature in enumerate(feature_list, 1):
            f.write(f"  {i}. {feature}\n")
    
    f.write(f"\n{'='*80}\n")
    f.write(f"TOTAL: {len(features)} features\n")

print("\n📄 Saved to: ema_crossover_features_list.txt")
