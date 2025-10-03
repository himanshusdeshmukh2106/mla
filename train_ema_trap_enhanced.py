#!/usr/bin/env python3
"""
EMA Trap Strategy - ENHANCED Training with More Features
Adds: Multiple EMAs (9, 21, 50, 200), RSI, MACD, Stochastic
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

def create_enhanced_features(df):
    """Create ENHANCED features with multiple EMAs and RSI"""
    
    print("\n🚀 Creating ENHANCED features...")
    
    import pandas_ta as ta
    
    # ========== MULTIPLE EMAs (WIDELY USED) ==========
    print("Adding multiple EMAs...")
    ema_periods = [9, 21, 50, 200]  # Common EMAs
    for period in ema_periods:
        df[f'EMA_{period}'] = ta.ema(df['close'], length=period)
        df[f'Distance_From_EMA{period}_Pct'] = (df['close'] - df[f'EMA_{period}']) / df[f'EMA_{period}'] * 100
    
    # EMA crosses for EMA 21 (primary)
    df['EMA21_Cross_Above'] = ((df['close'].shift(1) <= df['EMA_21'].shift(1)) & 
                               (df['close'] > df['EMA_21'])).astype(int)
    df['EMA21_Cross_Below'] = ((df['close'].shift(1) >= df['EMA_21'].shift(1)) & 
                               (df['close'] < df['EMA_21'])).astype(int)
    
    # EMA relationships (golden cross, death cross indicators)
    df['EMA9_Above_EMA21'] = (df['EMA_9'] > df['EMA_21']).astype(int)
    df['EMA21_Above_EMA50'] = (df['EMA_21'] > df['EMA_50']).astype(int)
    df['EMA50_Above_EMA200'] = (df['EMA_50'] > df['EMA_200']).astype(int)
    
    # Cross history for EMA 21
    for lookback in [2, 3, 5, 10]:
        df[f'Crosses_Above_Last_{lookback}'] = df['EMA21_Cross_Above'].rolling(lookback).sum()
        df[f'Crosses_Below_Last_{lookback}'] = df['EMA21_Cross_Below'].rolling(lookback).sum()
    
    # Distance trends
    df['Distance_EMA21_Change'] = df['Distance_From_EMA21_Pct'].diff()
    df['Distance_EMA21_Trend'] = df['Distance_From_EMA21_Pct'].rolling(3).mean()
    
    # ========== RSI (MULTIPLE PERIODS) ==========
    print("Adding RSI indicators...")
    rsi_periods = [14, 21]  # Standard and longer period
    for period in rsi_periods:
        df[f'RSI_{period}'] = ta.rsi(df['close'], length=period)
        
        # RSI levels
        df[f'RSI{period}_Oversold'] = (df[f'RSI_{period}'] < 30).astype(int)
        df[f'RSI{period}_Overbought'] = (df[f'RSI_{period}'] > 70).astype(int)
        df[f'RSI{period}_Neutral'] = ((df[f'RSI_{period}'] >= 40) & (df[f'RSI_{period}'] <= 60)).astype(int)
    
    # RSI momentum
    df['RSI_14_Change'] = df['RSI_14'].diff()
    df['RSI_14_Momentum'] = df['RSI_14_Change'].rolling(3).mean()
    
    # ========== MACD ==========
    print("Adding MACD...")
    macd_result = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd_result['MACD_12_26_9']
    df['MACD_Signal'] = macd_result['MACDs_12_26_9']
    df['MACD_Hist'] = macd_result['MACDh_12_26_9']
    df['MACD_Bullish'] = (df['MACD'] > df['MACD_Signal']).astype(int)
    df['MACD_Bearish'] = (df['MACD'] < df['MACD_Signal']).astype(int)
    
    # ========== ADX (TREND STRENGTH) ==========
    print("Adding ADX...")
    adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx_result['ADX_14']
    df['ADX_Change'] = df['ADX'].diff()
    
    # ADX ranges
    df['ADX_Very_Weak'] = (df['ADX'] < 15).astype(int)
    df['ADX_Weak'] = ((df['ADX'] >= 15) & (df['ADX'] < 20)).astype(int)
    df['ADX_Optimal'] = ((df['ADX'] >= 20) & (df['ADX'] <= 30)).astype(int)
    df['ADX_Strong'] = ((df['ADX'] > 30) & (df['ADX'] <= 40)).astype(int)
    df['ADX_Very_Strong'] = (df['ADX'] > 40).astype(int)
    
    # ========== TIME FEATURES ==========
    print("Adding time features...")
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    df['Time_Slot'] = (df['Hour'] * 60 + df['Minute']) // 15
    
    df['Is_9_15_to_9_30'] = ((df['Hour'] == 9) & (df['Minute'].between(15, 30))).astype(int)
    df['Is_9_30_to_10_00'] = ((df['Hour'] == 9) & (df['Minute'] > 30)).astype(int)
    df['Is_10_00_to_10_30'] = ((df['Hour'] == 10) & (df['Minute'] <= 30)).astype(int)
    df['Is_10_30_to_11_00'] = ((df['Hour'] == 10) & (df['Minute'] > 30)).astype(int)
    df['Is_11_00_to_12_00'] = (df['Hour'] == 11).astype(int)
    
    # ========== CANDLE FEATURES ==========
    print("Adding candle features...")
    df['Candle_Body_Pct'] = abs(df['close'] - df['open']) / df['open'] * 100
    df['Candle_Range_Pct'] = (df['high'] - df['low']) / df['open'] * 100
    
    df['Candle_Efficiency'] = np.where(
        df['Candle_Range_Pct'] > 0,
        df['Candle_Body_Pct'] / df['Candle_Range_Pct'],
        0
    )
    
    # Candle sizes
    df['Micro_Candle'] = (df['Candle_Body_Pct'] <= 0.10).astype(int)
    df['Tiny_Candle'] = ((df['Candle_Body_Pct'] > 0.10) & (df['Candle_Body_Pct'] <= 0.15)).astype(int)
    df['Small_Candle'] = ((df['Candle_Body_Pct'] > 0.15) & (df['Candle_Body_Pct'] <= 0.25)).astype(int)
    df['Medium_Candle'] = ((df['Candle_Body_Pct'] > 0.25) & (df['Candle_Body_Pct'] <= 0.50)).astype(int)
    
    df['Green_Candle'] = (df['close'] > df['open']).astype(int)
    df['Red_Candle'] = (df['close'] < df['open']).astype(int)
    
    # ========== PRICE MOMENTUM ==========
    print("Adding price momentum...")
    df['Price_Change_1'] = df['close'].pct_change(1) * 100
    df['Price_Change_3'] = df['close'].pct_change(3) * 100
    df['Price_Change_5'] = df['close'].pct_change(5) * 100
    df['Price_Momentum'] = df['Price_Change_1'].rolling(3).mean()
    
    # ========== VOLUME FEATURES ==========
    print("Adding volume features...")
    df['Volume_SMA_20'] = df['volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_SMA_20']
    df['Volume_Change'] = df['volume'].pct_change() * 100
    
    df['Very_Low_Volume'] = (df['Volume_Ratio'] < 0.5).astype(int)
    df['Low_Volume'] = ((df['Volume_Ratio'] >= 0.5) & (df['Volume_Ratio'] < 0.8)).astype(int)
    df['Normal_Volume'] = ((df['Volume_Ratio'] >= 0.8) & (df['Volume_Ratio'] <= 1.2)).astype(int)
    df['High_Volume'] = (df['Volume_Ratio'] > 1.2).astype(int)
    
    # ========== INTERACTION FEATURES ==========
    print("Adding interaction features...")
    df['EMA_ADX_Signal'] = df['Distance_From_EMA21_Pct'] * df['ADX'] / 100
    df['RSI_ADX_Signal'] = (df['RSI_14'] - 50) * df['ADX'] / 100
    df['Volume_Candle_Signal'] = df['Volume_Ratio'] * df['Candle_Body_Pct']
    df['Time_EMA_Signal'] = df['Time_Slot'] * abs(df['Distance_From_EMA21_Pct'])
    df['MACD_RSI_Signal'] = df['MACD_Hist'] * (df['RSI_14'] - 50) / 100
    
    df.dropna(inplace=True)
    
    feature_count = len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])
    print(f"✅ Created {feature_count} ENHANCED features")
    print(f"   - EMAs: 9, 21, 50, 200")
    print(f"   - RSI: 14, 21 periods")
    print(f"   - MACD: 12/26/9")
    print(f"   - ADX: 14 period")
    print(f"   - Plus all original features")
    
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

def train_enhanced_model(X, y, feature_names):
    """Train enhanced model"""
    
    print("\n🚀 ENHANCED Training Mode...")
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
    print(f"   Features: {len(feature_names)}")
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Moderate hyperparameter grid
    param_grid = {
        'max_depth': [4, 5, 6],
        'learning_rate': [0.05, 0.07, 0.1],
        'n_estimators': [300, 500],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'min_child_weight': [3, 5],
        'gamma': [0, 0.1],
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
    
    print("\n⚙️  Running GridSearchCV...")
    print(f"   Combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=tscv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    
    # Train final model
    best_model = grid_search.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📊 Model Performance:")
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}% (Win Rate)")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1 Score:  {f1*100:.2f}%")
    print(f"   ROC-AUC:   {roc_auc*100:.2f}%")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 Top 20 Features:")
    for idx, row in feature_importance.head(20).iterrows():
        print(f"   {row['feature']:30s} {row['importance']*100:5.2f}%")
    
    return best_model, feature_importance

def save_model(model, feature_importance, feature_names, metrics):
    """Save model and metadata"""
    import joblib
    import json
    
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/ema_trap_enhanced.pkl'
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'Enhanced EMA Trap Strategy',
        'algorithm': 'XGBoost',
        'training_date': datetime.now().isoformat(),
        'features_count': len(feature_names),
        'features': feature_names,
        'metrics': metrics,
        'feature_importance': feature_importance.head(30).values.tolist(),
        'enhancements': [
            'Multiple EMAs (9, 21, 50, 200)',
            'RSI (14, 21 periods)',
            'MACD (12/26/9)',
            'EMA relationships',
            'RSI levels and momentum',
            'Enhanced interactions'
        ]
    }
    
    metadata_path = 'models/ema_trap_enhanced_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Metadata saved: {metadata_path}")

if __name__ == "__main__":
    print("="*60)
    print("🚀 ENHANCED EMA TRAP TRAINING")
    print("   With Multiple EMAs + RSI + MACD")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Create enhanced features
    df = create_enhanced_features(df)
    
    # Generate targets
    df = generate_targets(df)
    
    # Prepare features
    feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 
                                                         'target', 'future_return', 'future_max', 'future_min',
                                                         'Volume_SMA_20']]
    
    X = df[feature_cols].values
    y = df['target'].values
    
    print(f"\n📊 Final Dataset:")
    print(f"   Samples: {len(X):,}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Positive: {y.sum():,} ({y.sum()/len(y)*100:.2f}%)")
    
    # Train model
    model, feature_importance = train_enhanced_model(X, y, feature_cols)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
    }
    
    # Save model
    save_model(model, feature_importance, feature_cols, metrics)
    
    print("\n" + "="*60)
    print("✅ ENHANCED TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel: models/ema_trap_enhanced.pkl")
    print(f"Features: {len(feature_cols)}")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print("\n🎯 Next: Run backtest with enhanced model")
    print("="*60)
