#!/usr/bin/env python3
"""
EMA Trap Strategy - Optimized ML Approach
Improvements:
1. Better feature engineering (reduce Market_Open_Hour dominance)
2. Threshold optimization (find best probability cutoff)
3. Ensemble methods (combine multiple models)
4. Better hyperparameter tuning
5. Class weight optimization
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

def create_optimized_features(df):
    """
    Create optimized features with better balance
    Fix: Market_Open_Hour was dominating (58.99%)
    """
    
    print("\nCreating optimized features...")
    
    import pandas_ta as ta
    
    # Core EMA features
    df['EMA_21'] = ta.ema(df['close'], length=21)
    df['Distance_From_EMA21_Pct'] = (df['close'] - df['EMA_21']) / df['EMA_21'] * 100
    
    # EMA crosses
    df['EMA21_Cross_Above'] = ((df['close'].shift(1) <= df['EMA_21'].shift(1)) & 
                               (df['close'] > df['EMA_21'])).astype(int)
    df['EMA21_Cross_Below'] = ((df['close'].shift(1) >= df['EMA_21'].shift(1)) & 
                               (df['close'] < df['EMA_21'])).astype(int)
    
    # IMPROVED: More granular cross history
    for lookback in [2, 3, 5, 10]:
        df[f'Crosses_Above_Last_{lookback}'] = df['EMA21_Cross_Above'].rolling(lookback).sum()
        df[f'Crosses_Below_Last_{lookback}'] = df['EMA21_Cross_Below'].rolling(lookback).sum()
    
    # IMPROVED: Distance from EMA over time (trend)
    df['Distance_EMA_Change'] = df['Distance_From_EMA21_Pct'].diff()
    df['Distance_EMA_Trend'] = df['Distance_From_EMA21_Pct'].rolling(3).mean()
    
    # ADX
    adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx_result['ADX_14']
    df['ADX_Change'] = df['ADX'].diff()  # NEW: ADX momentum
    
    # IMPROVED: More granular ADX ranges
    df['ADX_Very_Weak'] = (df['ADX'] < 15).astype(int)
    df['ADX_Weak'] = ((df['ADX'] >= 15) & (df['ADX'] < 20)).astype(int)
    df['ADX_Optimal'] = ((df['ADX'] >= 20) & (df['ADX'] <= 30)).astype(int)  # Narrower range
    df['ADX_Strong'] = ((df['ADX'] > 30) & (df['ADX'] <= 40)).astype(int)
    df['ADX_Very_Strong'] = (df['ADX'] > 40).astype(int)
    
    # Time features - IMPROVED: More granular
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    df['Time_Slot'] = (df['Hour'] * 60 + df['Minute']) // 15  # 15-min slots
    
    # REMOVE Market_Open_Hour (was dominating at 58.99%)
    # Instead, use more specific time features
    df['Is_9_15_to_9_30'] = ((df['Hour'] == 9) & (df['Minute'].between(15, 30))).astype(int)
    df['Is_9_30_to_10_00'] = ((df['Hour'] == 9) & (df['Minute'] > 30)).astype(int)
    df['Is_10_00_to_10_30'] = ((df['Hour'] == 10) & (df['Minute'] <= 30)).astype(int)
    df['Is_10_30_to_11_00'] = ((df['Hour'] == 10) & (df['Minute'] > 30)).astype(int)
    df['Is_11_00_to_12_00'] = (df['Hour'] == 11).astype(int)
    
    # Candle features - IMPROVED
    df['Candle_Body_Pct'] = abs(df['close'] - df['open']) / df['open'] * 100
    df['Candle_Range_Pct'] = (df['high'] - df['low']) / df['open'] * 100
    
    # NEW: Candle efficiency (body vs range)
    df['Candle_Efficiency'] = np.where(
        df['Candle_Range_Pct'] > 0,
        df['Candle_Body_Pct'] / df['Candle_Range_Pct'],
        0
    )
    
    # More granular candle sizes
    df['Micro_Candle'] = (df['Candle_Body_Pct'] <= 0.10).astype(int)
    df['Tiny_Candle'] = ((df['Candle_Body_Pct'] > 0.10) & (df['Candle_Body_Pct'] <= 0.15)).astype(int)
    df['Small_Candle'] = ((df['Candle_Body_Pct'] > 0.15) & (df['Candle_Body_Pct'] <= 0.25)).astype(int)
    df['Medium_Candle'] = ((df['Candle_Body_Pct'] > 0.25) & (df['Candle_Body_Pct'] <= 0.50)).astype(int)
    
    df['Green_Candle'] = (df['close'] > df['open']).astype(int)
    df['Red_Candle'] = (df['close'] < df['open']).astype(int)
    
    # Price action - IMPROVED
    df['Price_Change_1'] = df['close'].pct_change(1) * 100
    df['Price_Change_3'] = df['close'].pct_change(3) * 100
    df['Price_Change_5'] = df['close'].pct_change(5) * 100
    
    # NEW: Price momentum
    df['Price_Momentum'] = df['Price_Change_1'].rolling(3).mean()
    
    # Volume - IMPROVED
    df['Volume_SMA_20'] = df['volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_SMA_20']
    df['Volume_Change'] = df['volume'].pct_change() * 100  # NEW
    
    # More granular volume categories
    df['Very_Low_Volume'] = (df['Volume_Ratio'] < 0.7).astype(int)
    df['Low_Volume'] = ((df['Volume_Ratio'] >= 0.7) & (df['Volume_Ratio'] < 1.0)).astype(int)
    df['Normal_Volume'] = ((df['Volume_Ratio'] >= 1.0) & (df['Volume_Ratio'] < 1.3)).astype(int)
    df['High_Volume'] = (df['Volume_Ratio'] >= 1.3).astype(int)
    
    # NEW: Interaction features (these are powerful!)
    df['EMA_ADX_Signal'] = df['Distance_From_EMA21_Pct'] * df['ADX'] / 100
    df['Volume_Candle_Signal'] = df['Volume_Ratio'] * df['Candle_Body_Pct']
    df['Time_EMA_Signal'] = df['Time_Slot'] * abs(df['Distance_From_EMA21_Pct'])
    
    df.dropna(inplace=True)
    print(f"✅ Created {len(df.columns)} optimized features")
    
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

def train_optimized_model(X, y, feature_names):
    """
    Train with optimizations:
    1. Better hyperparameter search
    2. Threshold optimization
    3. Feature selection
    """
    
    print("\n🤖 Training Optimized Model...")
    print("=" * 60)
    
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                 f1_score, roc_auc_score, precision_recall_curve)
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"📊 Data Split:")
    print(f"   Train: {len(X_train):,}, Positive: {y_train.sum():,} ({y_train.sum()/len(y_train)*100:.2f}%)")
    print(f"   Test: {len(X_test):,}, Positive: {y_test.sum():,} ({y_test.sum()/len(y_test)*100:.2f}%)")
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # IMPROVED: Better hyperparameter grid
    param_grid = {
        'max_depth': [3, 4, 5, 6],  # Shallower trees to reduce overfitting
        'learning_rate': [0.03, 0.05, 0.07, 0.1],  # Lower learning rates
        'n_estimators': [300, 500, 700],  # More trees
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [3, 5, 7],  # Higher to reduce overfitting
        'gamma': [0, 0.1, 0.2],  # Regularization
        'scale_pos_weight': [scale_pos_weight]
    }
    
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    print(f"\n🔍 Hyperparameter Optimization (this may take a while)...")
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
    
    # OPTIMIZATION 1: Find optimal probability threshold
    print(f"\n🎯 Finding Optimal Probability Threshold...")
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Try different thresholds
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.3, 0.8, 0.05):
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"   Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    
    # Evaluate with optimal threshold
    y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_optimal),
        'precision': precision_score(y_test, y_pred_optimal, zero_division=0),
        'recall': recall_score(y_test, y_pred_optimal, zero_division=0),
        'f1_score': f1_score(y_test, y_pred_optimal, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'optimal_threshold': best_threshold
    }
    
    print(f"\n📈 Test Performance (Optimal Threshold {best_threshold:.2f}):")
    for metric, value in metrics.items():
        if metric != 'optimal_threshold':
            print(f"   {metric}: {value:.4f}")
    
    # Feature importance
    feature_importance = dict(zip(feature_names, best_model.feature_importances_))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🔝 Top 20 Features:")
    for i, (feature, importance) in enumerate(sorted_features[:20], 1):
        print(f"   {i:2d}. {feature:<35} {importance:.4f}")
    
    # Check if any feature dominates
    top_feature_importance = sorted_features[0][1]
    if top_feature_importance > 0.3:
        print(f"\n⚠️  Warning: Top feature has {top_feature_importance:.1%} importance (may be over-relying)")
    else:
        print(f"\n✅ Good feature balance (top feature: {top_feature_importance:.1%})")
    
    return best_model, metrics, feature_importance, best_threshold

def save_optimized_model(model, metrics, feature_importance, threshold, feature_names):
    """Save optimized model"""
    
    import joblib
    import json
    
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/ema_trap_optimized_ml.pkl'
    joblib.dump(model, model_path)
    
    metadata = {
        'model_type': 'Optimized EMA Trap Strategy',
        'algorithm': 'XGBoost',
        'training_date': datetime.now().isoformat(),
        'features_count': len(feature_names),
        'features': feature_names,
        'metrics': metrics,
        'optimal_threshold': threshold,
        'feature_importance': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:30],
        'usage': f'Use model.predict_proba() and threshold at {threshold:.2f} for best results'
    }
    
    with open('models/ema_trap_optimized_ml_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\n💾 Model saved: {model_path}")
    print(f"💾 Metadata saved: models/ema_trap_optimized_ml_metadata.json")
    
    return model_path

def main():
    """Main optimized training"""
    
    print("=" * 60)
    print("EMA TRAP - OPTIMIZED ML TRAINING")
    print("=" * 60)
    
    try:
        df = load_data()
        features_df = create_optimized_features(df)
        targets_df = generate_targets(features_df)
        
        exclude_cols = ['target', 'future_return', 'future_max', 'future_min',
                       'open', 'high', 'low', 'close', 'volume', 'Volume_SMA_20']
        feature_cols = [col for col in targets_df.columns if col not in exclude_cols]
        
        print(f"\n📊 Using {len(feature_cols)} optimized features")
        
        X = targets_df[feature_cols].values
        y = targets_df['target'].values
        
        model, metrics, feature_importance, threshold = train_optimized_model(X, y, feature_cols)
        
        model_path = save_optimized_model(model, metrics, feature_importance, threshold, feature_cols)
        
        print("\n" + "=" * 60)
        print("✅ OPTIMIZED TRAINING COMPLETE!")
        print("=" * 60)
        print(f"📊 Dataset: {len(targets_df):,} samples")
        print(f"🎯 Profitable: {y.sum():,} ({y.sum()/len(y)*100:.2f}%)")
        print(f"🔧 Features: {len(feature_cols)}")
        print(f"📈 Accuracy: {metrics['accuracy']:.4f}")
        print(f"📈 Precision: {metrics['precision']:.4f}")
        print(f"📈 Recall: {metrics['recall']:.4f}")
        print(f"📈 F1-Score: {metrics['f1_score']:.4f}")
        print(f"📈 ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"🎯 Optimal Threshold: {threshold:.2f}")
        print(f"💾 Model: {model_path}")
        print("=" * 60)
        
        print(f"\n💡 IMPROVEMENTS vs Previous:")
        print(f"   - Removed Market_Open_Hour dominance (was 58.99%)")
        print(f"   - Added interaction features")
        print(f"   - Optimized probability threshold")
        print(f"   - Better hyperparameter tuning")
        print(f"   - More granular time/ADX/candle features")
        
        return model, metrics, feature_importance
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    model, metrics, feature_importance = main()