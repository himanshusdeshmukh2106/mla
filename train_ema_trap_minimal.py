#!/usr/bin/env python3
"""
EMA Trap Strategy - Minimal ML Approach
Only features directly related to the EMA trap strategy rules:
- 21 EMA
- ADX (14-period)
- Time windows
- Candle size
- Price action around EMA
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup environment
sys.path.insert(0, 'src')

def load_data():
    """Load Reliance 5-minute data"""
    possible_paths = [
        "data/reliance_data_5min_full_year.csv",
        "/content/reliance_data_5min_full_year.csv",
        "/content/data/reliance_data_5min_full_year.csv",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            print(f"✅ Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
            return df
    
    raise FileNotFoundError("Data file not found")

def create_minimal_ema_trap_features(df):
    """
    Create ONLY the minimal features needed for EMA trap strategy
    Based on your exact rules
    """
    
    print("\nCreating minimal EMA trap features...")
    
    import pandas_ta as ta
    
    # ============================================
    # 1. CORE: 21-PERIOD EMA (The Foundation)
    # ============================================
    
    df['EMA_21'] = ta.ema(df['close'], length=21)
    
    # Price-EMA relationship
    df['Price_Above_EMA21'] = (df['close'] > df['EMA_21']).astype(int)
    df['Price_Below_EMA21'] = (df['close'] < df['EMA_21']).astype(int)
    df['Distance_From_EMA21_Pct'] = (df['close'] - df['EMA_21']) / df['EMA_21'] * 100
    
    # EMA crosses (trap detection)
    df['EMA21_Cross_Above'] = ((df['close'].shift(1) <= df['EMA_21'].shift(1)) & 
                               (df['close'] > df['EMA_21'])).astype(int)
    df['EMA21_Cross_Below'] = ((df['close'].shift(1) >= df['EMA_21'].shift(1)) & 
                               (df['close'] < df['EMA_21'])).astype(int)
    
    # Recent cross history (trap context - minimal)
    df['Crosses_Above_Last_3'] = df['EMA21_Cross_Above'].rolling(3).sum()
    df['Crosses_Below_Last_3'] = df['EMA21_Cross_Below'].rolling(3).sum()
    df['Crosses_Above_Last_5'] = df['EMA21_Cross_Above'].rolling(5).sum()
    df['Crosses_Below_Last_5'] = df['EMA21_Cross_Below'].rolling(5).sum()
    
    # How long price stayed on each side
    df['Candles_Above_EMA_Last_5'] = df['Price_Above_EMA21'].rolling(5).sum()
    df['Candles_Below_EMA_Last_5'] = df['Price_Below_EMA21'].rolling(5).sum()
    
    # ============================================
    # 2. ADX (14-period) - Trend Strength Filter
    # ============================================
    
    adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx_result['ADX_14']
    
    # ADX as continuous feature (let ML learn optimal range)
    # Also add binary indicators for reference
    df['ADX_In_Range_20_36'] = ((df['ADX'] >= 20) & (df['ADX'] <= 36)).astype(int)
    df['ADX_Strong'] = (df['ADX'] > 25).astype(int)
    df['ADX_Weak'] = (df['ADX'] < 20).astype(int)
    
    # ============================================
    # 3. TIME FEATURES - Entry Windows
    # ============================================
    
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    
    # Your specific entry windows
    df['Entry_Window_1'] = ((df['Hour'] == 9) & (df['Minute'].between(15, 30))).astype(int)
    df['Entry_Window_2'] = ((df['Hour'] == 10) & (df['Minute'] <= 60)).astype(int)
    df['In_Entry_Window'] = (df['Entry_Window_1'] | df['Entry_Window_2']).astype(int)
    
    # Market session context (minimal)
    df['Market_Open_Hour'] = (df['Hour'] == 9).astype(int)
    df['First_Hour'] = ((df['Hour'] == 9) | ((df['Hour'] == 10) & (df['Minute'] == 0))).astype(int)
    
    # ============================================
    # 4. CANDLE SIZE - Entry Candle Filter
    # ============================================
    
    df['Candle_Body_Pct'] = abs(df['close'] - df['open']) / df['open'] * 100
    
    # Your specific rule
    df['Small_Candle_0_20'] = (df['Candle_Body_Pct'] <= 0.20).astype(int)
    
    # Let ML learn if slightly different thresholds work better
    df['Tiny_Candle_0_10'] = (df['Candle_Body_Pct'] <= 0.10).astype(int)
    df['Small_Candle_0_30'] = (df['Candle_Body_Pct'] <= 0.30).astype(int)
    df['Small_Candle_0_50'] = (df['Candle_Body_Pct'] <= 0.50).astype(int)
    
    # Candle direction (for trap context)
    df['Green_Candle'] = (df['close'] > df['open']).astype(int)
    df['Red_Candle'] = (df['close'] < df['open']).astype(int)
    
    # ============================================
    # 5. PRICE ACTION (Minimal - for trap context)
    # ============================================
    
    # Recent price movement
    df['Price_Change_1'] = df['close'].pct_change(1) * 100
    df['Price_Change_2'] = df['close'].pct_change(2) * 100
    
    # Candle range (for volatility context)
    df['Candle_Range_Pct'] = (df['high'] - df['low']) / df['open'] * 100
    
    # ============================================
    # 6. VOLUME (Minimal - trap confirmation)
    # ============================================
    
    df['Volume_SMA_20'] = df['volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_SMA_20']
    df['Low_Volume'] = (df['Volume_Ratio'] < 1.0).astype(int)
    df['High_Volume'] = (df['Volume_Ratio'] > 1.5).astype(int)
    
    # Remove NaN rows
    initial_rows = len(df)
    df.dropna(inplace=True)
    final_rows = len(df)
    
    print(f"✅ Created {len(df.columns)} features (minimal set)")
    print(f"✅ Data: {initial_rows} → {final_rows} rows ({initial_rows - final_rows} NaN removed)")
    
    return df

def generate_ml_targets(df):
    """
    Generate targets for EVERY candle based on future returns
    NO rule-based filtering
    """
    
    print("\nGenerating ML targets for ALL candles...")
    
    # Strategy parameters
    PROFIT_THRESHOLD = 0.002  # 0.2% profit target
    LOOKAHEAD_PERIODS = 2     # 2 candles = 10 minutes
    
    # Calculate future returns
    df['future_return'] = df['close'].shift(-LOOKAHEAD_PERIODS) / df['close'] - 1
    
    # Calculate max favorable and adverse moves
    df['future_max_return'] = df['high'].shift(-LOOKAHEAD_PERIODS).rolling(LOOKAHEAD_PERIODS).max() / df['close'] - 1
    df['future_min_return'] = df['low'].shift(-LOOKAHEAD_PERIODS).rolling(LOOKAHEAD_PERIODS).min() / df['close'] - 1
    
    # TARGET: Can we make 0.2% profit in either direction?
    df['target'] = 0
    
    profitable_long = (df['future_max_return'] >= PROFIT_THRESHOLD)
    profitable_short = (df['future_min_return'] <= -PROFIT_THRESHOLD)
    
    df.loc[profitable_long | profitable_short, 'target'] = 1
    
    # Direction indicator (for analysis)
    df['target_direction'] = 0
    df.loc[profitable_long & ~profitable_short, 'target_direction'] = 1   # Long
    df.loc[profitable_short & ~profitable_long, 'target_direction'] = -1  # Short
    
    # Remove rows without future data
    df = df[df['future_return'].notna()].copy()
    
    # Analysis
    total_samples = len(df)
    positive_samples = df['target'].sum()
    positive_pct = positive_samples / total_samples * 100
    
    long_opportunities = (df['target_direction'] == 1).sum()
    short_opportunities = (df['target_direction'] == -1).sum()
    
    print(f"\n📊 TARGET ANALYSIS:")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Profitable opportunities: {positive_samples:,} ({positive_pct:.2f}%)")
    print(f"   Long opportunities: {long_opportunities:,}")
    print(f"   Short opportunities: {short_opportunities:,}")
    
    return df

def train_minimal_ml_model(X, y, feature_names):
    """
    Train ML model with minimal features
    """
    
    print("\n🤖 Training Minimal ML Model...")
    print("=" * 60)
    
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                 f1_score, roc_auc_score, classification_report, confusion_matrix)
    
    # Time-series split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"📊 Data Split:")
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test: {len(X_test):,} samples")
    print(f"   Train positive: {y_train.sum():,} ({y_train.sum()/len(y_train)*100:.2f}%)")
    print(f"   Test positive: {y_test.sum():,} ({y_test.sum()/len(y_test)*100:.2f}%)")
    
    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")
    
    # Simplified hyperparameter grid (faster training)
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'n_estimators': [200, 300, 500],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'scale_pos_weight': [scale_pos_weight]
    }
    
    # Base model
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    
    # Time series CV
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Grid search
    print(f"\n🔍 Hyperparameter Optimization...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Train
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    print(f"\n✅ Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    print(f"   Best CV F1-Score: {grid_search.best_score_:.4f}")
    
    # Evaluate
    print(f"\n📈 Test Set Evaluation:")
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"   TN: {cm[0,0]:,} | FP: {cm[0,1]:,}")
    print(f"   FN: {cm[1,0]:,} | TP: {cm[1,1]:,}")
    
    # Feature importance
    feature_importance = dict(zip(feature_names, best_model.feature_importances_))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🔝 Feature Importance (All {len(sorted_features)} features):")
    for i, (feature, importance) in enumerate(sorted_features, 1):
        print(f"   {i:2d}. {feature:<30} {importance:.4f}")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return best_model, metrics, feature_importance, (X_train, X_test, y_train, y_test)

def save_model(model, metrics, feature_importance, feature_names):
    """Save model and metadata"""
    
    print(f"\n💾 Saving Model...")
    
    import joblib
    import json
    
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/ema_trap_minimal_ml.pkl'
    joblib.dump(model, model_path)
    print(f"   ✅ Model: {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'Minimal EMA Trap Strategy',
        'algorithm': 'XGBoost',
        'training_date': datetime.now().isoformat(),
        'features_count': len(feature_names),
        'features': feature_names,
        'metrics': metrics,
        'feature_importance': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True),
        'strategy_rules': {
            'ema_period': 21,
            'adx_period': 14,
            'adx_range': [20, 36],
            'entry_windows': ['9:15-9:30', '10:00-11:00'],
            'max_candle_body': 0.20,
            'profit_threshold': 0.002,
            'lookahead_periods': 2
        }
    }
    
    with open('models/ema_trap_minimal_ml_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"   ✅ Metadata: models/ema_trap_minimal_ml_metadata.json")
    
    return model_path

def main():
    """Main minimal ML training pipeline"""
    
    print("=" * 60)
    print("EMA TRAP STRATEGY - MINIMAL ML APPROACH")
    print("=" * 60)
    print("Features: Only 21-EMA, ADX, Time, Candle Size, Volume")
    print("No RSI, MACD, Stochastic, Bollinger Bands")
    print("=" * 60)
    
    try:
        # 1. Load data
        print("\n📂 STEP 1: Loading Data")
        df = load_data()
        
        # 2. Create minimal features
        print("\n🔧 STEP 2: Creating Minimal Features")
        features_df = create_minimal_ema_trap_features(df)
        
        # 3. Generate targets
        print("\n🎯 STEP 3: Generating Targets")
        targets_df = generate_ml_targets(features_df)
        
        # 4. Prepare training data
        print("\n📊 STEP 4: Preparing Training Data")
        
        # Select only minimal features
        exclude_cols = ['target', 'target_direction', 'future_return', 'future_max_return', 'future_min_return',
                       'open', 'high', 'low', 'close', 'volume', 'Volume_SMA_20']
        feature_cols = [col for col in targets_df.columns if col not in exclude_cols]
        
        print(f"   Using {len(feature_cols)} minimal features:")
        for i, feat in enumerate(feature_cols, 1):
            print(f"      {i:2d}. {feat}")
        
        X = targets_df[feature_cols].values
        y = targets_df['target'].values
        
        # 5. Train model
        print("\n🚀 STEP 5: Training Model")
        model, metrics, feature_importance, data_split = train_minimal_ml_model(X, y, feature_cols)
        
        # 6. Save
        print("\n💾 STEP 6: Saving Model")
        model_path = save_model(model, metrics, feature_importance, feature_cols)
        
        # 7. Summary
        print("\n" + "=" * 60)
        print("✅ MINIMAL ML TRAINING COMPLETE!")
        print("=" * 60)
        print(f"📊 Dataset: {len(targets_df):,} samples")
        print(f"🎯 Profitable: {y.sum():,} ({y.sum()/len(y)*100:.2f}%)")
        print(f"🔧 Features: {len(feature_cols)} (minimal)")
        print(f"📈 Accuracy: {metrics['accuracy']:.4f}")
        print(f"📈 F1-Score: {metrics['f1_score']:.4f}")
        print(f"📈 ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"💾 Model: {model_path}")
        print("=" * 60)
        
        return model, metrics, feature_importance
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    model, metrics, feature_importance = main()