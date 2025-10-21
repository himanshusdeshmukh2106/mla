"""
Diagnose Overfitting Issues
Identifies data leakage, target problems, and overfitting causes
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def load_and_analyze():
    """Load data and models to diagnose issues"""
    
    print("="*80)
    print("🔍 OVERFITTING DIAGNOSIS")
    print("="*80)
    
    # Load data
    print("\n1️⃣ Loading data...")
    try:
        df = pd.read_csv('ema_crossover_with_targets.csv')
        print(f"   ✅ Loaded {len(df):,} rows")
    except FileNotFoundError:
        print("   ❌ Data file not found!")
        return
    
    # Check target distribution
    print("\n2️⃣ Analyzing target distribution...")
    target_cols = [col for col in df.columns if 'target_' in col]
    
    for target_col in target_cols[:3]:  # Check first 3 targets
        target = df[target_col]
        positive = target.sum()
        negative = len(target) - positive
        ratio = positive / len(target) * 100
        
        print(f"\n   {target_col}:")
        print(f"      Positive: {positive:,} ({ratio:.1f}%)")
        print(f"      Negative: {negative:,} ({100-ratio:.1f}%)")
        
        if ratio > 90 or ratio < 10:
            print(f"      ⚠️  SEVERE CLASS IMBALANCE!")
        elif ratio > 80 or ratio < 20:
            print(f"      ⚠️  High class imbalance")
    
    # Check for data leakage
    print("\n3️⃣ Checking for data leakage...")
    
    # Common leakage patterns
    leakage_features = []
    
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'datetime', 'timestamp']
    exclude_cols.extend(target_cols)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Check for future-looking features
    future_keywords = ['future', 'next', 'forward', 'ahead', 'target']
    for feat in feature_cols:
        if any(keyword in feat.lower() for keyword in future_keywords):
            leakage_features.append(feat)
            print(f"   ⚠️  Suspicious feature: {feat}")
    
    # Check for perfect correlations with target
    print("\n4️⃣ Checking feature-target correlations...")
    
    target_col = 'target_combined_1.0R' if 'target_combined_1.0R' in df.columns else target_cols[0]
    
    high_corr_features = []
    for feat in feature_cols[:20]:  # Check first 20 features
        try:
            corr = df[feat].corr(df[target_col])
            if abs(corr) > 0.95:
                high_corr_features.append((feat, corr))
                print(f"   🚨 VERY HIGH CORRELATION: {feat} = {corr:.4f}")
            elif abs(corr) > 0.8:
                print(f"   ⚠️  High correlation: {feat} = {corr:.4f}")
        except:
            pass
    
    # Check model predictions
    print("\n5️⃣ Analyzing model predictions...")
    
    try:
        # Load metadata
        with open('models/ema_crossover_ensemble_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        selected_features = metadata['feature_names']
        print(f"   Features used: {len(selected_features)}")
        
        # Load model
        model = joblib.load('models/ema_crossover_ensemble.pkl')
        
        # Prepare data
        X = df[selected_features].fillna(0).values
        y = df[target_col].values
        
        # Split same way as training
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Confusion matrices
        train_cm = confusion_matrix(y_train, train_pred)
        test_cm = confusion_matrix(y_test, test_pred)
        
        print(f"\n   Training Set:")
        print(f"      Confusion Matrix:\n{train_cm}")
        print(f"      Accuracy: {(train_pred == y_train).mean():.4f}")
        
        print(f"\n   Test Set:")
        print(f"      Confusion Matrix:\n{test_cm}")
        print(f"      Accuracy: {(test_pred == y_test).mean():.4f}")
        
        # Check if all predictions are same class
        if len(np.unique(train_pred)) == 1:
            print(f"\n   🚨 MODEL PREDICTS ONLY ONE CLASS: {np.unique(train_pred)[0]}")
        
        if len(np.unique(test_pred)) == 1:
            print(f"\n   🚨 MODEL PREDICTS ONLY ONE CLASS ON TEST: {np.unique(test_pred)[0]}")
        
    except Exception as e:
        print(f"   ❌ Could not analyze model: {e}")
    
    # Check for temporal leakage
    print("\n6️⃣ Checking for temporal leakage...")
    
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        
        # Check if target is based on future data
        print("   Checking if targets use future information...")
        
        # Simple test: if target at time t depends on data at t+1, t+2, etc.
        # This would show up as perfect predictions
        
    # Summary
    print("\n" + "="*80)
    print("📋 DIAGNOSIS SUMMARY")
    print("="*80)
    
    issues_found = []
    
    if high_corr_features:
        issues_found.append("🚨 CRITICAL: Features with >0.95 correlation to target")
    
    if leakage_features:
        issues_found.append("⚠️  Suspicious feature names suggesting future data")
    
    # Check target balance
    if 'target_combined_1.0R' in df.columns:
        ratio = df['target_combined_1.0R'].sum() / len(df) * 100
        if ratio > 90 or ratio < 10:
            issues_found.append(f"🚨 CRITICAL: Severe class imbalance ({ratio:.1f}% positive)")
    
    if issues_found:
        print("\n❌ ISSUES FOUND:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
    else:
        print("\n✅ No obvious issues detected")
        print("   But 100% accuracy is still suspicious!")
    
    return df, metadata if 'metadata' in locals() else None


def recommend_fixes(df, metadata):
    """Recommend specific fixes"""
    
    print("\n" + "="*80)
    print("💡 RECOMMENDED FIXES")
    print("="*80)
    
    print("\n1️⃣ FIX TARGET DEFINITION")
    print("-" * 60)
    print("Current target might be too easy or have data leakage.")
    print("\nRecommended changes:")
    print("  • Use FUTURE returns only (not current candle)")
    print("  • Add minimum holding period (e.g., 3-5 candles)")
    print("  • Make target harder (e.g., 1% instead of 0.5%)")
    print("  • Add stop loss in target calculation")
    
    print("\n2️⃣ FIX FEATURE ENGINEERING")
    print("-" * 60)
    print("Remove any features that use future information:")
    print("  • No 'future_' features")
    print("  • No forward-looking indicators")
    print("  • Lag features should only look backward")
    print("  • Remove features with >0.9 correlation to target")
    
    print("\n3️⃣ ADD PROPER VALIDATION")
    print("-" * 60)
    print("  • Use time-based split (not random)")
    print("  • Add out-of-sample test on completely new data")
    print("  • Implement walk-forward with gaps between train/test")
    print("  • Test on different market conditions")
    
    print("\n4️⃣ ADJUST MODEL COMPLEXITY")
    print("-" * 60)
    print("  • Reduce max_depth (try 3-4 instead of 6)")
    print("  • Increase regularization (higher reg_alpha, reg_lambda)")
    print("  • Use fewer features (20-30 instead of 62)")
    print("  • Add early stopping with validation set")
    
    print("\n5️⃣ CHECK DATA QUALITY")
    print("-" * 60)
    print("  • Ensure enough samples (need 1000+ positive examples)")
    print("  • Check for duplicate rows")
    print("  • Verify datetime ordering")
    print("  • Look for data entry errors")
    
    print("\n" + "="*80)
    print("🚀 NEXT STEPS")
    print("="*80)
    print("\n1. Review target generation code")
    print("2. Check for data leakage in features")
    print("3. Retrain with stricter validation")
    print("4. Test on completely new data")
    print("5. If still 100% accuracy → investigate data source")


if __name__ == "__main__":
    try:
        df, metadata = load_and_analyze()
        if df is not None:
            recommend_fixes(df, metadata)
        
        print("\n" + "="*80)
        print("⚠️  IMPORTANT: Do NOT use this model for live trading!")
        print("   100% accuracy = overfitting = will lose money")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
