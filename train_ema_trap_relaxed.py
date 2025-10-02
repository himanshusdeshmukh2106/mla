#!/usr/bin/env python3
"""
EMA Trap Strategy Model Training Script - Relaxed Conditions
Train XGBoost model with more lenient conditions to get sufficient training samples
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

def load_reliance_data():
    """Load and prepare Reliance 5-minute data"""
    
    # Try multiple possible data paths
    possible_paths = [
        "data/reliance_data_5min_full_year.csv",
        "/content/reliance_data_5min_full_year.csv",
        "/content/data/reliance_data_5min_full_year.csv",
        "./reliance_data_5min_full_year.csv",
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(f"Data file not found in any of these locations: {possible_paths}")
    
    print(f"Loading data from: {data_path}")
    
    # Load the data
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
    
    return df

def create_relaxed_ema_features(data):
    """Create EMA trap features with relaxed conditions"""
    
    print("Creating EMA trap features with relaxed conditions...")
    
    import pandas_ta as ta
    
    df = data.copy()
    
    # Core indicators
    df['EMA_21'] = ta.ema(df['close'], length=21)
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    df['RSI'] = ta.rsi(df['close'], length=14)
    
    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    
    # Bollinger Bands (simplified)
    bb = ta.bbands(df['close'], length=20, std=2)
    if bb is not None and len(bb.columns) >= 3:
        bb_cols = bb.columns.tolist()
        df['BB_Upper'] = bb.iloc[:, 0]  # First column is usually upper
        df['BB_Middle'] = bb.iloc[:, 1]  # Middle
        df['BB_Lower'] = bb.iloc[:, 2]   # Lower
        df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    else:
        # Fallback: simple moving average bands
        sma_20 = ta.sma(df['close'], length=20)
        std_20 = df['close'].rolling(20).std()
        df['BB_Upper'] = sma_20 + (2 * std_20)
        df['BB_Middle'] = sma_20
        df['BB_Lower'] = sma_20 - (2 * std_20)
        df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # ATR and Volume
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['Volume_SMA_20'] = ta.sma(df['volume'], length=20)
    df['Volume_Ratio_20'] = df['volume'] / df['Volume_SMA_20']
    
    # EMA relationship features
    df['Price_Above_EMA21'] = (df['close'] > df['EMA_21']).astype(int)
    df['Distance_From_EMA21_Pct'] = (df['close'] - df['EMA_21']) / df['EMA_21'] * 100
    
    # EMA crosses
    df['EMA21_Cross_Above'] = ((df['close'].shift(1) <= df['EMA_21'].shift(1)) & 
                               (df['close'] > df['EMA_21'])).astype(int)
    df['EMA21_Cross_Below'] = ((df['close'].shift(1) >= df['EMA_21'].shift(1)) & 
                               (df['close'] < df['EMA_21'])).astype(int)
    
    # RELAXED trap detection - allow up to 3 candles lookback
    df['Bearish_Trap_Confirmed'] = 0
    df['Bullish_Trap_Confirmed'] = 0
    
    for i in range(3, len(df)):
        # Bearish trap: look for cross below after recent cross above
        if df['EMA21_Cross_Below'].iloc[i] == 1:
            for j in range(1, 4):  # Look back 3 candles
                if i-j >= 0 and df['EMA21_Cross_Above'].iloc[i-j] == 1:
                    df['Bearish_Trap_Confirmed'].iloc[i] = 1
                    break
        
        # Bullish trap: look for cross above after recent cross below
        if df['EMA21_Cross_Above'].iloc[i] == 1:
            for j in range(1, 4):  # Look back 3 candles
                if i-j >= 0 and df['EMA21_Cross_Below'].iloc[i-j] == 1:
                    df['Bullish_Trap_Confirmed'].iloc[i] = 1
                    break
    
    # RELAXED entry conditions
    # 1. ADX range: 15-40 (instead of 20-36)
    df['ADX_In_Range'] = ((df['ADX'] >= 15) & (df['ADX'] <= 40)).astype(int)
    
    # 2. Time windows: Expanded
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    df['Entry_Window_1'] = ((df['Hour'] == 9) & (df['Minute'].between(15, 60))).astype(int)  # 9:15-10:00
    df['Entry_Window_2'] = ((df['Hour'] == 10) | (df['Hour'] == 11)).astype(int)  # 10:00-12:00
    df['In_Entry_Window'] = (df['Entry_Window_1'] | df['Entry_Window_2']).astype(int)
    
    # 3. Candle size: <= 0.5% (instead of 0.20%)
    df['Candle_Body_Size_Pct'] = abs(df['close'] - df['open']) / df['open'] * 100
    df['Small_Candle'] = (df['Candle_Body_Size_Pct'] <= 0.5).astype(int)
    
    # Additional features
    df['Green_Candle'] = (df['close'] > df['open']).astype(int)
    df['Red_Candle'] = (df['close'] < df['open']).astype(int)
    df['Upper_Shadow_Pct'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open'] * 100
    df['Lower_Shadow_Pct'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open'] * 100
    
    # Remove NaN rows
    initial_rows = len(df)
    df.dropna(inplace=True)
    final_rows = len(df)
    
    print(f"Features created: {initial_rows} -> {final_rows} rows ({initial_rows - final_rows} NaN removed)")
    
    return df

def generate_relaxed_targets(df):
    """Generate targets with relaxed profit/loss thresholds"""
    
    print("Generating targets with relaxed thresholds...")
    
    # RELAXED thresholds: 0.2% instead of 0.4%
    profit_threshold = 0.002  # 0.2%
    loss_threshold = -0.002   # -0.2%
    lookahead = 2
    
    # Calculate future returns
    df['future_return'] = df['close'].shift(-lookahead) / df['close'] - 1
    
    # Entry conditions (relaxed)
    bearish_entry = (
        (df['Bearish_Trap_Confirmed'] == 1) &
        (df['In_Entry_Window'] == 1) &
        (df['ADX_In_Range'] == 1) &
        (df['Small_Candle'] == 1)
    )
    
    bullish_entry = (
        (df['Bullish_Trap_Confirmed'] == 1) &
        (df['In_Entry_Window'] == 1) &
        (df['ADX_In_Range'] == 1) &
        (df['Small_Candle'] == 1)
    )
    
    # Target generation
    df['target'] = 0
    
    # Profitable entries
    bearish_profitable = bearish_entry & (df['future_return'] <= loss_threshold)
    bullish_profitable = bullish_entry & (df['future_return'] >= profit_threshold)
    
    df.loc[bearish_profitable, 'target'] = 1
    df.loc[bullish_profitable, 'target'] = 1
    
    # Analysis
    total_bearish = bearish_entry.sum()
    total_bullish = bullish_entry.sum()
    profitable_bearish = bearish_profitable.sum()
    profitable_bullish = bullish_profitable.sum()
    
    print(f"Entry Analysis (Relaxed Conditions):")
    print(f"  Bearish entries: {total_bearish}, profitable: {profitable_bearish}")
    print(f"  Bullish entries: {total_bullish}, profitable: {profitable_bullish}")
    print(f"  Total signals: {total_bearish + total_bullish}")
    print(f"  Total targets: {df['target'].sum()}")
    print(f"  Success rate: {(profitable_bearish + profitable_bullish) / (total_bearish + total_bullish) * 100:.1f}%" if (total_bearish + total_bullish) > 0 else "N/A")
    
    return df

def train_xgboost_model(X, y, feature_names):
    """Train XGBoost model with class balancing"""
    
    print("Training XGBoost model with class balancing...")
    
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.utils.class_weight import compute_class_weight
    
    # Calculate class weights to handle imbalance
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, class_weights))
    
    print(f"Class weights: {class_weight_dict}")
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train target distribution: {np.bincount(y_train)}")
    print(f"Test target distribution: {np.bincount(y_test)}")
    
    # Simplified hyperparameter grid for faster training
    param_grid = {
        'max_depth': [4, 6],
        'learning_rate': [0.1, 0.15],
        'n_estimators': [200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [class_weight_dict.get(1, 1)]  # Handle class imbalance
    }
    
    # Base model
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Grid search
    print("Performing hyperparameter optimization...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1-score: {grid_search.best_score_:.4f}")
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
    }
    
    print("Model Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Feature importance
    feature_importance = dict(zip(feature_names, best_model.feature_importances_))
    
    return best_model, metrics, feature_importance, (X_train, X_test, y_train, y_test)

def main():
    """Main training pipeline with relaxed conditions"""
    
    print("EMA Trap Strategy - Relaxed Conditions Training")
    print("=" * 60)
    
    try:
        # 1. Load data
        print("\n1. Loading data...")
        raw_data = load_reliance_data()
        
        # 2. Create features
        print("\n2. Creating features...")
        features_df = create_relaxed_ema_features(raw_data)
        
        # 3. Generate targets
        print("\n3. Generating targets...")
        targets_df = generate_relaxed_targets(features_df)
        
        # 4. Prepare training data
        print("\n4. Preparing training data...")
        
        selected_features = [
            'EMA_21', 'ADX', 'Distance_From_EMA21_Pct', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Position', 'ATR', 'Volume_Ratio_20', 'Bearish_Trap_Confirmed',
            'Bullish_Trap_Confirmed', 'In_Entry_Window', 'ADX_In_Range',
            'Candle_Body_Size_Pct', 'Hour', 'Green_Candle', 'Red_Candle',
            'Upper_Shadow_Pct', 'Lower_Shadow_Pct'
        ]
        
        # Filter available features
        available_features = [f for f in selected_features if f in targets_df.columns]
        print(f"Using {len(available_features)} features")
        
        X = targets_df[available_features].values
        y = targets_df['target'].values
        
        print(f"Training data: {len(X)} samples")
        print(f"Target distribution: {np.bincount(y)}")
        
        positive_samples = y.sum()
        print(f"Positive samples: {positive_samples} ({positive_samples/len(y)*100:.2f}%)")
        
        if positive_samples < 10:
            print("⚠️ Still very few positive samples. Consider further relaxing conditions.")
        
        # 5. Train model
        print("\n5. Training model...")
        model, metrics, feature_importance, data_split = train_xgboost_model(X, y, available_features)
        
        # 6. Show results
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print(f"📊 Dataset: {len(targets_df)} samples")
        print(f"🎯 Positive targets: {positive_samples}")
        print(f"📈 Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"📈 Test F1-Score: {metrics['f1_score']:.4f}")
        print(f"📈 Test ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Top features
        print(f"\n🔝 Top 10 Features:")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"  {i:2d}. {feature:<25} {importance:.4f}")
        
        # Save model
        print(f"\n💾 Saving model...")
        import joblib
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/ema_trap_relaxed_model.pkl')
        print(f"Model saved to: models/ema_trap_relaxed_model.pkl")
        
        return model, metrics, feature_importance
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    model, metrics, feature_importance = main()