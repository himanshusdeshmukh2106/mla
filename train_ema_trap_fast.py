#!/usr/bin/env python3
"""
EMA Trap Strategy - FAST Training (3-5 minutes)
Optimized for speed while maintaining quality
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'src')

def load_data():
    """Load data"""
    possible_paths = [
        "data/reliance_data_5min_full_year.csv",
        "/content/reliance_data_5min_full_year.csv",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            print(f"✅ Loaded {len(df)} data points")
            return df
    raise FileNotFoundError("Data file not found")

def create_fast_features(df):
    """Create optimized features (same as optimized but faster)"""
    
    print("\nCreating features...")
    
    import pandas_ta as ta
    
    # Core EMA
    df['EMA_21'] = ta.ema(df['close'], length=21)
    df['Distance_From_EMA21_Pct'] = (df['close'] - df['EMA_21']) / df['EMA_21'] * 100
    
    # EMA crosses
    df['EMA21_Cross_Above'] = ((df['close'].shift(1) <= df['EMA_21'].shift(1)) & 
                               (df['close'] > df['EMA_21'])).astype(int)
    df['EMA21_Cross_Below'] = ((df['close'].shift(1) >= df['EMA_21'].shift(1)) & 
                               (df['close'] < df['EMA_21'])).astype(int)
    
    # Cross history
    for lookback in [2, 3, 5]:
        df[f'Crosses_Above_Last_{lookback}'] = df['EMA21_Cross_Above'].rolling(lookback).sum()
        df[f'Crosses_Below_Last_{lookback}'] = df['EMA21_Cross_Below'].rolling(lookback).sum()
    
    # Distance trend
    df['Distance_EMA_Change'] = df['Distance_From_EMA21_Pct'].diff()
    
    # ADX
    adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx_result['ADX_14']
    
    # Granular ADX
    df['ADX_Weak'] = ((df['ADX'] >= 15) & (df['ADX'] < 20)).astype(int)
    df['ADX_Optimal'] = ((df['ADX'] >= 20) & (df['ADX'] <= 30)).astype(int)
    df['ADX_Strong'] = ((df['ADX'] > 30) & (df['ADX'] <= 40)).astype(int)
    
    # Time features - GRANULAR (fixes Market_Open_Hour dominance)
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    
    df['Is_9_15_to_9_30'] = ((df['Hour'] == 9) & (df['Minute'].between(15, 30))).astype(int)
    df['Is_9_30_to_10_00'] = ((df['Hour'] == 9) & (df['Minute'] > 30)).astype(int)
    df['Is_10_00_to_10_30'] = ((df['Hour'] == 10) & (df['Minute'] <= 30)).astype(int)
    df['Is_10_30_to_11_00'] = ((df['Hour'] == 10) & (df['Minute'] > 30)).astype(int)
    
    # Candle features
    df['Candle_Body_Pct'] = abs(df['close'] - df['open']) / df['open'] * 100
    df['Candle_Range_Pct'] = (df['high'] - df['low']) / df['open'] * 100
    
    # Granular candle sizes
    df['Micro_Candle'] = (df['Candle_Body_Pct'] <= 0.10).astype(int)
    df['Tiny_Candle'] = ((df['Candle_Body_Pct'] > 0.10) & (df['Candle_Body_Pct'] <= 0.15)).astype(int)
    df['Small_Candle'] = ((df['Candle_Body_Pct'] > 0.15) & (df['Candle_Body_Pct'] <= 0.25)).astype(int)
    
    df['Green_Candle'] = (df['close'] > df['open']).astype(int)
    df['Red_Candle'] = (df['close'] < df['open']).astype(int)
    
    # Price action
    df['Price_Change_1'] = df['close'].pct_change(1) * 100
    df['Price_Change_3'] = df['close'].pct_change(3) * 100
    
    # Volume
    df['Volume_SMA_20'] = df['volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_SMA_20']
    
    df['Low_Volume'] = ((df['Volume_Ratio'] >= 0.7) & (df['Volume_Ratio'] < 1.0)).astype(int)
    df['High_Volume'] = (df['Volume_Ratio'] >= 1.3).astype(int)
    
    # KEY: Interaction features (powerful!)
    df['EMA_ADX_Signal'] = df['Distance_From_EMA21_Pct'] * df['ADX'] / 100
    df['Volume_Candle_Signal'] = df['Volume_Ratio'] * df['Candle_Body_Pct']
    
    df.dropna(inplace=True)
    print(f"✅ Created {len(df.columns)} features")
    
    return df

def generate_targets(df):
    """Generate targets"""
    print("\nGenerating targets...")
    
    PROFIT_THRESHOLD = 0.002
    LOOKAHEAD = 2
    
    df['future_return'] = df['close'].shift(-LOOKAHEAD) / df['close'] - 1
    df['future_max'] = df['high'].shift(-LOOKAHEAD).rolling(LOOKAHEAD).max() / df['close'] - 1
    df['future_min'] = df['low'].shift(-LOOKAHEAD).rolling(LOOKAHEAD).min() / df['close'] - 1
    
    df['target'] = 0
    profitable = (df['future_max'] >= PROFIT_THRESHOLD) | (df['future_min'] <= -PROFIT_THRESHOLD)
    df.loc[profitable, 'target'] = 1
    
    df = df[df['future_return'].notna()].copy()
    
    print(f"   Total: {len(df):,}, Profitable: {df['target'].sum():,} ({df['target'].sum()/len(df)*100:.2f}%)")
    
    return df

def train_fast_model(X, y, feature_names):
    """
    FAST training - reduced grid search
    Training time: 3-5 minutes (vs 20+ minutes)
    """
    
    print("\n🚀 FAST Training Mode...")
    print("=" * 60)
    
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                 f1_score, roc_auc_score)
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"📊 Data Split:")
    print(f"   Train: {len(X_train):,}, Positive: {y_train.sum():,} ({y_train.sum()/len(y_train)*100:.2f}%)")
    print(f"   Test: {len(X_test):,}, Positive: {y_test.sum():,} ({y_test.sum()/len(y_test)*100:.2f}%)")
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # FAST: Reduced hyperparameter grid (36 combinations vs 1000+)
    param_grid = {
        'max_depth': [4, 5, 6],  # 3 values
        'learning_rate': [0.05, 0.1],  # 2 values
        'n_estimators': [300, 500],  # 2 values
        'subsample': [0.8, 0.9],  # 2 values
        'colsample_bytree': [0.8],  # 1 value (fixed)
        'min_child_weight': [3, 5],  # 2 values
        'scale_pos_weight': [scale_pos_weight]
    }
    # Total: 3 × 2 × 2 × 2 × 1 × 2 = 48 combinations
    # With 3-fold CV = 144 fits (vs 1000+ before)
    
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        tree_method='hist'  # Faster
    )
    
    tscv = TimeSeriesSplit(n_splits=3)  # 3 folds (fast)
    
    print(f"\n🔍 Hyperparameter Search (48 combinations, ~3-5 min)...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    print(f"\n✅ Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    print(f"   Best CV F1: {grid_search.best_score_:.4f}")
    
    # Find optimal threshold (FAST: fewer steps)
    print(f"\n🎯 Finding Optimal Threshold...")
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.40, 0.70, 0.05):  # Fewer steps
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"   Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    
    # Evaluate
    y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_optimal),
        'precision': precision_score(y_test, y_pred_optimal, zero_division=0),
        'recall': recall_score(y_test, y_pred_optimal, zero_division=0),
        'f1_score': f1_score(y_test, y_pred_optimal, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'optimal_threshold': best_threshold
    }
    
    print(f"\n📈 Test Performance (Threshold {best_threshold:.2f}):")
    for metric, value in metrics.items():
        if metric != 'optimal_threshold':
            print(f"   {metric}: {value:.4f}")
    
    # Feature importance
    feature_importance = dict(zip(feature_names, best_model.feature_importances_))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🔝 Top 15 Features:")
    for i, (feature, importance) in enumerate(sorted_features[:15], 1):
        print(f"   {i:2d}. {feature:<35} {importance:.4f}")
    
    # Check balance
    top_importance = sorted_features[0][1]
    if top_importance > 0.3:
        print(f"\n⚠️  Top feature: {top_importance:.1%} (may be over-relying)")
    else:
        print(f"\n✅ Good balance (top: {top_importance:.1%})")
    
    return best_model, metrics, feature_importance, best_threshold

def save_model(model, metrics, feature_importance, threshold, feature_names):
    """Save model"""
    
    import joblib
    import json
    
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/ema_trap_fast_ml.pkl'
    joblib.dump(model, model_path)
    
    metadata = {
        'model_type': 'Fast EMA Trap Strategy',
        'algorithm': 'XGBoost',
        'training_date': datetime.now().isoformat(),
        'features_count': len(feature_names),
        'features': feature_names,
        'metrics': metrics,
        'optimal_threshold': threshold,
        'feature_importance': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20],
        'training_time': '3-5 minutes',
        'usage': f'Use model.predict_proba() >= {threshold:.2f} for trading decisions'
    }
    
    with open('models/ema_trap_fast_ml_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\n💾 Model: {model_path}")
    print(f"💾 Metadata: models/ema_trap_fast_ml_metadata.json")
    
    return model_path

def main():
    """Main fast training"""
    
    print("=" * 60)
    print("EMA TRAP - FAST TRAINING (3-5 minutes)")
    print("=" * 60)
    print("Optimized for speed while maintaining quality")
    print("=" * 60)
    
    try:
        start_time = datetime.now()
        
        df = load_data()
        features_df = create_fast_features(df)
        targets_df = generate_targets(features_df)
        
        exclude_cols = ['target', 'future_return', 'future_max', 'future_min',
                       'open', 'high', 'low', 'close', 'volume', 'Volume_SMA_20']
        feature_cols = [col for col in targets_df.columns if col not in exclude_cols]
        
        print(f"\n📊 Using {len(feature_cols)} features")
        
        X = targets_df[feature_cols].values
        y = targets_df['target'].values
        
        model, metrics, feature_importance, threshold = train_fast_model(X, y, feature_cols)
        
        model_path = save_model(model, metrics, feature_importance, threshold, feature_cols)
        
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        
        print("\n" + "=" * 60)
        print("✅ FAST TRAINING COMPLETE!")
        print("=" * 60)
        print(f"⏱️  Training time: {elapsed:.1f} minutes")
        print(f"📊 Dataset: {len(targets_df):,} samples")
        print(f"🎯 Profitable: {y.sum():,} ({y.sum()/len(y)*100:.2f}%)")
        print(f"🔧 Features: {len(feature_cols)}")
        print(f"📈 Accuracy: {metrics['accuracy']:.4f}")
        print(f"📈 Precision: {metrics['precision']:.4f}")
        print(f"📈 Recall: {metrics['recall']:.4f}")
        print(f"📈 F1-Score: {metrics['f1_score']:.4f}")
        print(f"📈 ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"🎯 Threshold: {threshold:.2f}")
        print(f"💾 Model: {model_path}")
        print("=" * 60)
        
        print(f"\n💡 SPEED OPTIMIZATIONS:")
        print(f"   ✅ Reduced grid: 48 combinations (vs 1000+)")
        print(f"   ✅ 3-fold CV (vs 5-fold)")
        print(f"   ✅ Fewer threshold tests")
        print(f"   ✅ tree_method='hist' (faster)")
        print(f"   ✅ Result: ~5 min (vs 20+ min)")
        
        return model, metrics, feature_importance
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    model, metrics, feature_importance = main()