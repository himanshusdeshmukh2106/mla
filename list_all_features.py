"""List all 51 features used in training with their importance"""
import json

# Load metadata
meta = json.load(open('models/ema_trap_balanced_ml_metadata.json'))

print("="*70)
print("ALL 51 FEATURES USED IN TRAINING")
print("="*70)

print(f"\nTotal Features: {meta['features_count']}")
print(f"Training Date: {meta['training_date']}")
print(f"Model Type: {meta['model_type']}")

print("\n" + "="*70)
print("FEATURES BY IMPORTANCE (Top 30)")
print("="*70)

for i, (feat, imp) in enumerate(meta['feature_importance'][:30], 1):
    imp_val = float(imp) * 100
    
    # Categorize
    if 'EMA' in feat or 'Distance' in feat or 'Cross' in feat:
        category = '🔵 EMA'
    elif 'Hour' in feat or 'Minute' in feat or 'Time' in feat or 'Is_' in feat:
        category = '⏰ Time'
    elif 'Candle' in feat or 'Green' in feat or 'Red' in feat:
        category = '📏 Candle'
    elif 'Volume' in feat:
        category = '📊 Volume'
    elif 'ADX' in feat:
        category = '💪 ADX'
    elif 'Price' in feat:
        category = '📈 Price'
    else:
        category = '🔗 Other'
    
    print(f"{i:2d}. {feat:30s} {imp_val:5.2f}%  {category}")

print("\n" + "="*70)
print("ALL 51 FEATURES (Alphabetical)")
print("="*70)

features = sorted(meta['features'])
for i, feat in enumerate(features, 1):
    if i % 3 == 1:
        print(f"\n{i:2d}. {feat:25s}", end="")
    else:
        print(f" {i:2d}. {feat:25s}", end="")

print("\n\n" + "="*70)
print("SUMMARY BY CATEGORY")
print("="*70)

categories = {
    'EMA': [],
    'Time': [],
    'Candle': [],
    'Volume': [],
    'ADX': [],
    'Price': [],
    'Other': []
}

for feat, imp in meta['feature_importance']:
    imp_val = float(imp)
    if 'EMA' in feat or 'Distance' in feat or 'Cross' in feat:
        categories['EMA'].append(imp_val)
    elif 'Hour' in feat or 'Minute' in feat or 'Time' in feat or 'Is_' in feat:
        categories['Time'].append(imp_val)
    elif 'Candle' in feat or 'Green' in feat or 'Red' in feat:
        categories['Candle'].append(imp_val)
    elif 'Volume' in feat:
        categories['Volume'].append(imp_val)
    elif 'ADX' in feat:
        categories['ADX'].append(imp_val)
    elif 'Price' in feat:
        categories['Price'].append(imp_val)
    else:
        categories['Other'].append(imp_val)

print("\nCategory          Features  Total Importance")
print("-" * 50)
for cat, imps in sorted(categories.items(), key=lambda x: sum(x[1]), reverse=True):
    total = sum(imps) * 100
    count = len(imps)
    print(f"{cat:15s}  {count:3d}       {total:5.2f}%")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("\n1. Time features are MOST important (28%+)")
print("   → When you trade matters more than what you trade")
print("\n2. Candle size is #1 single feature (10.11%)")
print("   → Volatility/movement is key predictor")
print("\n3. EMA features total only ~18%")
print("   → NOT primarily an EMA trap detector")
print("\n4. Top 10 features = 48% of importance")
print("   → Model focuses on key signals")
print("\n5. All 51 features work together")
print("   → No single feature dominates")
print("="*70)
