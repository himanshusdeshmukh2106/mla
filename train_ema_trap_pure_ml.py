#!/usr/bin/env python3
"""
EMA Trap Strategy - Pure ML Approach
Let ML learn EMA trap patterns directly from ALL candles, not pre-filtered signals
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

def create_comprehensive_features(df):
    """
    Create ALL features for ML to learn from
    No pre-filtering - let ML decide what's important
    """
    
    print("Creating comprehensive feature set for ML...")
    
    import pandas_ta as ta
    
    # ============================================
    # 1. CORE EMA TRAP FEATURES
    # ============================================
    
    # EMA indicators
    df['EMA_12'] = ta.ema(df['close'], length=12)
    df['EMA_21'] = ta.ema(df['close'], length=21)
    df['EMA_26'] = ta.ema(df['close'], length=26)
    df['EMA_50'] = ta.ema(df['close'], length=50)
    
    # Price-EMA relationships
    df['Price_Above_EMA21'] = (df['close'] > df['EMA_21']).astype(int)
    df['Price_Below_EMA21'] = (df['close'] < df['EMA_21']).astype(int)
    df['Distance_From_EMA21'] = df['close'] - df['EMA_21']
    df['Distance_From_EMA21_Pct'] = (df['close'] - df['EMA_21']) / df['EMA_21'] * 100
    
    # EMA crosses (the trap setup indicators)
    df['EMA21_Cross_Above'] = ((df['close'].shift(1) <= df['EMA_21'].shift(1)) & 
                               (df['close'] > df['EMA_21'])).astype(int)
    df['EMA21_Cross_Below'] = ((df['close'].shift(1) >= df['EMA_21'].shift(1)) & 
                               (df['close'] < df['EMA_21'])).astype(int)
    
    # Recent cross history (trap context)
    df['Crosses_Above_Last_5'] = df['EMA21_Cross_Above'].rolling(5).sum()
    df['Crosses_Below_Last_5'] = df['EMA21_Cross_Below'].rolling(5).sum()
    df['Crosses_Above_Last_10'] = df['EMA21_Cross_Above'].rolling(10).sum()
    df['Crosses_Below_Last_10'] = df['EMA21_Cross_Below'].rolling(10).sum()
    
    # Price position relative to EMA over time
    df['Above_EMA_Last_5'] = df['Price_Above_EMA21'].rolling(5).sum()
    df['Above_EMA_Last_10'] = df['Price_Above_EMA21'].rolling(10).sum()
    
    # ============================================
    # 2. TREND STRENGTH (ADX and related)
    # ============================================
    
    adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx_result['ADX_14']
    df['DI_Plus'] = adx_result['DMP_14']
    df['DI_Minus'] = adx_result['DMN_14']
    
    # ADX conditions (as features, not filters)
    df['ADX_Strong'] = (df['ADX'] > 25).astype(int)
    df['ADX_Weak'] = (df['ADX'] < 20).astype(int)
    df['ADX_In_Range_20_36'] = ((df['ADX'] >= 20) & (df['ADX'] <= 36)).astype(int)
    
    # ============================================
    # 3. MOMENTUM INDICATORS
    # ============================================
    
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
    df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
    
    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Histogram'] = macd['MACDh_12_26_9']
    df['MACD_Bullish'] = (df['MACD'] > df['MACD_Signal']).astype(int)
    
    # Stochastic
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    df['Stoch_K'] = stoch['STOCHk_14_3_3']
    df['Stoch_D'] = stoch['STOCHd_14_3_3']
    
    # Rate of Change
    df['ROC'] = ta.roc(df['close'], length=10)
    
    # ============================================
    # 4. VOLATILITY INDICATORS
    # ============================================
    
    # Bollinger Bands
    bb = ta.bbands(df['close'], length=20, std=2)
    if bb is not None and len(bb.columns) >= 3:
        df['BB_Upper'] = bb.iloc[:, 0]
        df['BB_Middle'] = bb.iloc[:, 1]
        df['BB_Lower'] = bb.iloc[:, 2]
    else:
        sma_20 = ta.sma(df['close'], length=20)
        std_20 = df['close'].rolling(20).std()
        df['BB_Upper'] = sma_20 + (2 * std_20)
        df['BB_Middle'] = sma_20
        df['BB_Lower'] = sma_20 - (2 * std_20)
    
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['BB_Upper_Touch'] = (df['high'] >= df['BB_Upper']).astype(int)
    df['BB_Lower_Touch'] = (df['low'] <= df['BB_Lower']).astype(int)
    
    # ATR
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ATR_Pct'] = df['ATR'] / df['close'] * 100
    
    # Volatility
    df['Returns'] = df['close'].pct_change()
    df['Volatility_5'] = df['Returns'].rolling(5).std() * 100
    df['Volatility_10'] = df['Returns'].rolling(10).std() * 100
    df['Volatility_20'] = df['Returns'].rolling(20).std() * 100
    
    # ============================================
    # 5. VOLUME INDICATORS
    # ============================================
    
    df['Volume_SMA_20'] = ta.sma(df['volume'], length=20)
    df['Volume_Ratio'] = df['volume'] / df['Volume_SMA_20']
    df['Volume_Spike'] = (df['Volume_Ratio'] > 1.5).astype(int)
    df['Volume_Dry'] = (df['Volume_Ratio'] < 0.7).astype(int)
    
    df['OBV'] = ta.obv(df['close'], df['volume'])
    df['OBV_SMA'] = df['OBV'].rolling(20).mean()
    df['OBV_Trend'] = (df['OBV'] > df['OBV_SMA']).astype(int)
    
    # ============================================
    # 6. TIME-BASED FEATURES (Critical for EMA trap)
    # ============================================
    
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    df['Time_Minutes'] = df['Hour'] * 60 + df['Minute']
    
    # Entry windows (as features, not filters)
    df['Entry_Window_1'] = ((df['Hour'] == 9) & (df['Minute'].between(15, 30))).astype(int)
    df['Entry_Window_2'] = ((df['Hour'] == 10) & (df['Minute'] <= 60)).astype(int)
    df['In_Entry_Window'] = (df['Entry_Window_1'] | df['Entry_Window_2']).astype(int)
    
    # Market session features
    df['Market_Open'] = (df['Hour'] == 9).astype(int)
    df['First_Hour'] = ((df['Hour'] == 9) | ((df['Hour'] == 10) & (df['Minute'] == 0))).astype(int)
    df['Mid_Session'] = ((df['Hour'] >= 11) & (df['Hour'] <= 13)).astype(int)
    df['Last_Hour'] = (df['Hour'] >= 15).astype(int)
    
    # ============================================
    # 7. CANDLE PATTERN FEATURES
    # ============================================
    
    # Candle body and shadows
    df['Candle_Body'] = abs(df['close'] - df['open'])
    df['Candle_Body_Pct'] = df['Candle_Body'] / df['open'] * 100
    df['Candle_Range'] = df['high'] - df['low']
    df['Candle_Range_Pct'] = df['Candle_Range'] / df['open'] * 100
    
    # Candle size categories (as features)
    df['Tiny_Candle'] = (df['Candle_Body_Pct'] <= 0.1).astype(int)
    df['Small_Candle'] = (df['Candle_Body_Pct'] <= 0.2).astype(int)
    df['Medium_Candle'] = ((df['Candle_Body_Pct'] > 0.2) & (df['Candle_Body_Pct'] <= 0.5)).astype(int)
    df['Large_Candle'] = (df['Candle_Body_Pct'] > 0.5).astype(int)
    
    # Candle direction
    df['Green_Candle'] = (df['close'] > df['open']).astype(int)
    df['Red_Candle'] = (df['close'] < df['open']).astype(int)
    df['Doji'] = (abs(df['close'] - df['open']) < 0.0001 * df['open']).astype(int)
    
    # Shadows
    df['Upper_Shadow'] = df['high'] - np.maximum(df['open'], df['close'])
    df['Lower_Shadow'] = np.minimum(df['open'], df['close']) - df['low']
    df['Upper_Shadow_Pct'] = df['Upper_Shadow'] / df['open'] * 100
    df['Lower_Shadow_Pct'] = df['Lower_Shadow'] / df['open'] * 100
    
    # Body to range ratio
    df['Body_To_Range_Ratio'] = np.where(
        df['Candle_Range'] > 0,
        df['Candle_Body'] / df['Candle_Range'],
        0
    )
    
    # ============================================
    # 8. PRICE ACTION FEATURES
    # ============================================
    
    # Recent price changes
    df['Price_Change_1'] = df['close'].pct_change(1) * 100
    df['Price_Change_2'] = df['close'].pct_change(2) * 100
    df['Price_Change_5'] = df['close'].pct_change(5) * 100
    
    # High/Low relative to recent history
    df['High_Of_Last_5'] = df['high'].rolling(5).max()
    df['Low_Of_Last_5'] = df['low'].rolling(5).min()
    df['Near_High'] = (df['close'] >= df['High_Of_Last_5'] * 0.995).astype(int)
    df['Near_Low'] = (df['close'] <= df['Low_Of_Last_5'] * 1.005).astype(int)
    
    # ============================================
    # 9. INTERACTION FEATURES (ML will love these)
    # ============================================
    
    # EMA trap context features
    df['EMA_Trap_Context'] = df['Distance_From_EMA21_Pct'] * df['ADX']
    df['Volume_EMA_Signal'] = df['Volume_Ratio'] * df['Distance_From_EMA21_Pct']
    df['RSI_EMA_Divergence'] = (df['RSI'] - 50) * df['Distance_From_EMA21_Pct']
    
    # Remove NaN rows
    initial_rows = len(df)
    df.dropna(inplace=True)
    final_rows = len(df)
    
    print(f"✅ Created {len(df.columns)} features")
    print(f"✅ Data: {initial_rows} → {final_rows} rows ({initial_rows - final_rows} NaN removed)")
    
    return df

def generate_pure_ml_targets(df):
    """
    Generate targets for EVERY candle based on future returns
    NO rule-based filtering - let ML learn what works
    """
    
    print("\nGenerating ML targets for ALL candles...")
    
    # Strategy parameters
    PROFIT_THRESHOLD = 0.002  # 0.2% profit target
    LOOKAHEAD_PERIODS = 2     # 2 candles = 10 minutes
    
    # Calculate future returns
    df['future_return'] = df['close'].shift(-LOOKAHEAD_PERIODS) / df['close'] - 1
    
    # Also calculate max favorable and adverse moves
    df['future_max_return'] = df['high'].shift(-LOOKAHEAD_PERIODS).rolling(LOOKAHEAD_PERIODS).max() / df['close'] - 1
    df['future_min_return'] = df['low'].shift(-LOOKAHEAD_PERIODS).rolling(LOOKAHEAD_PERIODS).min() / df['close'] - 1
    
    # TARGET GENERATION - Pure ML approach
    # Label as 1 if we can make profit in either direction
    df['target'] = 0
    
    # Profitable if price moves enough in ANY direction
    profitable_long = (df['future_max_return'] >= PROFIT_THRESHOLD)
    profitable_short = (df['future_min_return'] <= -PROFIT_THRESHOLD)
    
    df.loc[profitable_long | profitable_short, 'target'] = 1
    
    # Additional target: Direction (for analysis)
    df['target_direction'] = 0  # 0 = no clear direction, 1 = long, -1 = short
    df.loc[profitable_long & ~profitable_short, 'target_direction'] = 1
    df.loc[profitable_short & ~profitable_long, 'target_direction'] = -1
    
    # Remove rows without future data
    df = df[df['future_return'].notna()].copy()
    
    # Analysis
    total_samples = len(df)
    positive_samples = df['target'].sum()
    positive_pct = positive_samples / total_samples * 100
    
    long_opportunities = (df['target_direction'] == 1).sum()
    short_opportunities = (df['target_direction'] == -1).sum()
    both_directions = ((df['target'] == 1) & (df['target_direction'] == 0)).sum()
    
    print(f"\n📊 TARGET ANALYSIS:")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Profitable opportunities: {positive_samples:,} ({positive_pct:.2f}%)")
    print(f"   Long opportunities: {long_opportunities:,}")
    print(f"   Short opportunities: {short_opportunities:,}")
    print(f"   Both directions: {both_directions:,}")
    
    return df

def train_pure_ml_model(X, y, feature_names):
    """
    Train ML model on ALL data without rule-based filtering
    """
    
    print("\n🤖 Training Pure ML Model...")
    print("=" * 60)
    
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                 f1_score, roc_auc_score, classification_report, confusion_matrix)
    
    # Time-series split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"📊 Data Split:")
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test: {len(X_test):,} samples")
    print(f"   Train positive: {y_train.sum():,} ({y_train.sum()/len(y_train)*100:.2f}%)")
    print(f"   Test positive: {y_test.sum():,} ({y_test.sum()/len(y_test)*100:.2f}%)")
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")
    
    # Hyperparameter grid (optimized for this problem)
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'n_estimators': [200, 300, 500],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1],
        'scale_pos_weight': [scale_pos_weight]
    }
    
    # Base XGBoost model
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        tree_method='hist'  # Faster training
    )
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Grid search
    print(f"\n🔍 Hyperparameter Optimization...")
    print(f"   Testing {len(list(GridSearchCV(base_model, param_grid, cv=tscv).get_params()['param_grid']))} combinations")
    
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
    
    # Evaluate on test set
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
    
    print(f"\n🔝 Top 15 Most Important Features:")
    for i, (feature, importance) in enumerate(sorted_features[:15], 1):
        print(f"   {i:2d}. {feature:<30} {importance:.4f}")
    
    # Classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return best_model, metrics, feature_importance, (X_train, X_test, y_train, y_test)

def save_model_and_analysis(model, metrics, feature_importance, feature_names):
    """Save trained model and analysis"""
    
    print(f"\n💾 Saving Model and Results...")
    
    import joblib
    import json
    
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/ema_trap_pure_ml.pkl'
    joblib.dump(model, model_path)
    print(f"   ✅ Model saved: {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'Pure ML EMA Trap Strategy',
        'algorithm': 'XGBoost',
        'training_date': datetime.now().isoformat(),
        'features_count': len(feature_names),
        'features': feature_names,
        'metrics': metrics,
        'top_features': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20],
        'strategy_description': 'ML learns EMA trap patterns directly from all candles without rule-based filtering'
    }
    
    metadata_path = 'models/ema_trap_pure_ml_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"   ✅ Metadata saved: {metadata_path}")
    
    return model_path

def main():
    """Main pure ML training pipeline"""
    
    print("=" * 60)
    print("EMA TRAP STRATEGY - PURE ML APPROACH")
    print("=" * 60)
    print("Let ML learn patterns directly from ALL data")
    print("No rule-based filtering - ML decides what works!")
    print("=" * 60)
    
    try:
        # 1. Load data
        print("\n📂 STEP 1: Loading Data")
        df = load_data()
        
        # 2. Create comprehensive features
        print("\n🔧 STEP 2: Feature Engineering")
        features_df = create_comprehensive_features(df)
        
        # 3. Generate ML targets
        print("\n🎯 STEP 3: Target Generation")
        targets_df = generate_pure_ml_targets(features_df)
        
        # 4. Prepare training data
        print("\n📊 STEP 4: Preparing Training Data")
        
        # Select all relevant features (exclude target and future columns)
        exclude_cols = ['target', 'target_direction', 'future_return', 'future_max_return', 'future_min_return',
                       'open', 'high', 'low', 'close', 'volume', 'Returns']
        feature_cols = [col for col in targets_df.columns if col not in exclude_cols]
        
        print(f"   Using {len(feature_cols)} features for training")
        
        X = targets_df[feature_cols].values
        y = targets_df['target'].values
        
        # 5. Train model
        print("\n🚀 STEP 5: Training ML Model")
        model, metrics, feature_importance, data_split = train_pure_ml_model(X, y, feature_cols)
        
        # 6. Save everything
        print("\n💾 STEP 6: Saving Results")
        model_path = save_model_and_analysis(model, metrics, feature_importance, feature_cols)
        
        # 7. Final summary
        print("\n" + "=" * 60)
        print("✅ PURE ML TRAINING COMPLETE!")
        print("=" * 60)
        print(f"📊 Dataset: {len(targets_df):,} samples")
        print(f"🎯 Profitable opportunities: {y.sum():,} ({y.sum()/len(y)*100:.2f}%)")
        print(f"🔧 Features used: {len(feature_cols)}")
        print(f"📈 Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"📈 Test F1-Score: {metrics['f1_score']:.4f}")
        print(f"📈 Test ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"💾 Model saved: {model_path}")
        print("=" * 60)
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Use model.predict_proba() to get trade probabilities")
        print(f"   2. Set threshold (e.g., > 0.6) for entering trades")
        print(f"   3. Backtest the strategy")
        print(f"   4. Deploy for live trading")
        
        return model, metrics, feature_importance
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    model, metrics, feature_importance = main()