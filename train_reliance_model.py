#!/usr/bin/env python3
"""
Training script for Reliance 5-minute intraday XGBoost model
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def load_reliance_training_data():
    """Load and preprocess Reliance 5-minute training data"""
    
    print("Loading Reliance 5-minute training data...")
    
    # Load data
    data_file = Path("data/reliance_data_5min_full_year.csv")
    df = pd.read_csv(data_file)
    
    # Convert datetime and set as index
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    
    print(f"Loaded {len(df)} rows from {df.index.min().date()} to {df.index.max().date()}")
    
    return df

def load_reliance_testing_data():
    """Load and preprocess Reliance 5-minute testing data"""
    
    print("Loading Reliance 5-minute testing data...")
    
    # Load data
    data_file = Path("testing data/reliance_data_5min_full_year_testing.csv")
    df = pd.read_csv(data_file)
    
    # Convert datetime and set as index
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    
    print(f"Loaded {len(df)} rows from {df.index.min().date()} to {df.index.max().date()}")
    
    return df

def create_technical_indicators(df):
    """Create technical indicators for 5-minute data"""
    
    print("Creating technical indicators...")
    
    # Price-based indicators
    df['SMA_20'] = df['close'].rolling(window=20).mean()  # 100-minute SMA
    df['SMA_50'] = df['close'].rolling(window=50).mean()  # 250-minute SMA
    df['EMA_12'] = df['close'].ewm(span=12).mean()        # 60-minute EMA
    df['EMA_26'] = df['close'].ewm(span=26).mean()        # 130-minute EMA
    
    # RSI (14 periods = 70 minutes)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands (20 periods = 100 minutes)
    bb_period = 20
    bb_std = 2
    df['BB_Middle'] = df['close'].rolling(window=bb_period).mean()
    bb_std_dev = df['close'].rolling(window=bb_period).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # ATR (Average True Range - 14 periods)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # Volume indicators
    df['Volume_SMA_20'] = df['volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_SMA_20']
    
    # Price momentum
    df['Price_Change'] = df['close'].pct_change()
    df['Price_Change_5'] = df['close'].pct_change(periods=5)  # 25-minute change
    df['Volatility_20'] = df['Price_Change'].rolling(window=20).std()
    
    # Trend indicators
    df['Price_SMA_20_Ratio'] = df['close'] / df['SMA_20']
    df['EMA_Cross'] = (df['EMA_12'] > df['EMA_26']).astype(int)

    # Stochastic Oscillator (14 periods)
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    # Williams %R (14 periods)
    df['Williams_R'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))

    # --- ADVANCED FEATURES ---

    # 1. Volatility-based feature
    df['ATR_Percentage'] = (df['ATR'] / df['close']) * 100

    # 2. Time-based features
    df['Day_of_Week'] = df.index.dayofweek # Monday=0, Sunday=6
    # Market open in India is 9:15 AM. Total minutes from midnight = 9*60 + 15 = 555
    df['Time_Since_Open'] = (df.index.hour * 60 + df.index.minute) - 555
    
    # 3. Interaction features
    df['RSI_Volume_Ratio'] = df['RSI'] * df['Volume_Ratio']
    df['RSI_x_Volatility'] = df['RSI'] * df['Volatility_20']
    
    print(f"Created technical indicators, including advanced and time-based features")
    
    return df

def generate_targets_triple_barrier(df, atr_multiplier_profit=1.5, atr_multiplier_loss=1.5, time_horizon_candles=10):
    """
    Generate binary classification targets using the Triple Barrier Method.
    - 1 (Buy): Price hits the upper barrier (take profit) first.
    - 0 (Hold): Price hits the lower barrier (stop loss) or time barrier first.
    """
    print(f"Generating targets with Triple Barrier Method...")
    print(f"Params: TP multiplier={atr_multiplier_profit}, SL multiplier={atr_multiplier_loss}, Horizon={time_horizon_candles} candles")

    n = len(df)
    targets = np.zeros(n)  # Default to 0 (Hold/Loss)

    for i in range(n - time_horizon_candles):
        entry_price = df['close'].iloc[i]
        atr = df['ATR'].iloc[i]

        # Define barriers
        take_profit_level = entry_price + (atr * atr_multiplier_profit)
        stop_loss_level = entry_price - (atr * atr_multiplier_loss)

        # Look into the future path
        future_path = df.iloc[i+1 : i+1+time_horizon_candles]

        # Check when barriers are hit
        hit_tp_mask = future_path['high'] >= take_profit_level
        hit_sl_mask = future_path['low'] <= stop_loss_level

        tp_hit_time = future_path.index[hit_tp_mask].min() if hit_tp_mask.any() else pd.NaT
        sl_hit_time = future_path.index[hit_sl_mask].min() if hit_sl_mask.any() else pd.NaT

        # Determine which barrier was hit first
        if pd.notna(tp_hit_time) and pd.notna(sl_hit_time):
            if tp_hit_time <= sl_hit_time:
                targets[i] = 1  # Profit hit first
        elif pd.notna(tp_hit_time):
            targets[i] = 1  # Only profit was hit

    df['target_binary'] = targets
    df = df.iloc[:-time_horizon_candles].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Target distribution
    target_dist = df['target_binary'].value_counts().sort_index()
    print(f"Target distribution:")
    for target_val, count in target_dist.items():
        pct = count / len(df) * 100
        label = {0: 'Hold/Loss', 1: 'Buy'}[int(target_val)]
        print(f"   {label} ({int(target_val)}): {count:,} samples ({pct:.1f}%)")

    return df

def prepare_training_data(train_df, test_df):
    """Prepare data for training"""
    
    print("Preparing training data...")
    
    # Remove rows with NaN values (from technical indicators)
    initial_rows_train = len(train_df)
    train_df = train_df.dropna()
    final_rows_train = len(train_df)
    print(f"Removed {initial_rows_train - final_rows_train} rows with NaN values from training data")

    initial_rows_test = len(test_df)
    test_df = test_df.dropna()
    final_rows_test = len(test_df)
    print(f"Removed {initial_rows_test - final_rows_test} rows with NaN values from testing data")
    
    # Define feature columns (exclude OHLCV and target columns)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target_binary']
    # Remove original target columns if they exist from previous runs
    if 'target' in train_df.columns:
        exclude_cols.append('target')
    if 'future_return' in train_df.columns:
        exclude_cols.append('future_return')
        
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    print(f"Selected {len(feature_cols)} features: {feature_cols}")
    
    X_train = train_df[feature_cols]
    y_train = train_df['target_binary']
    X_test = test_df[feature_cols]
    y_test = test_df['target_binary']
    
    print(f"Training set: {len(X_train)} samples ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"Test set: {len(X_test)} samples ({test_df.index.min().date()} to {test_df.index.max().date()})")
    
    return X_train, y_train, X_test, y_test, feature_cols

def train_xgboost_model(X_train, y_train, X_test, y_test, feature_cols):
    """Train XGBoost model with scale_pos_weight, RandomizedSearchCV, and Early Stopping"""
    
    print("Training XGBoost model with scale_pos_weight, RandomizedSearchCV, and Early Stopping...")
    
    try:
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, train_test_split
    except ImportError:
        print("XGBoost or scikit-learn not installed. Please install with:")
        print("   pip install xgboost scikit-learn")
        return None, None

    # Calculate scale_pos_weight to handle class imbalance
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # Create a validation set for early stopping
    X_train_es, X_val_es, y_train_es, y_val_es = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Define the expanded parameter grid for RandomizedSearchCV
    param_grid = {
        'max_depth': [3, 6, 9, 12],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [1, 1.5, 2, 3]
    }

    # Initialize the XGBoost classifier with early stopping parameters
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        tree_method='hist',
        device='cuda',
        scale_pos_weight=scale_pos_weight,
        n_estimators=1000,
        early_stopping_rounds=50  # Add early stopping here
    )

    # Initialize TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=100,
        cv=tscv,
        scoring='f1',
        n_jobs=1,
        verbose=2,
        random_state=42
    )
    
    # Fit RandomizedSearchCV, passing the eval_set as a fit parameter
    random_search.fit(
        X_train_es, y_train_es,
        eval_set=[(X_val_es, y_val_es)],
        verbose=False
    )
    
    print(f"Best parameters found: {random_search.best_params_}")
    
    # Get the best model
    best_model = random_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model training completed")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    print(f"Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<20} {row['importance']:.4f}")
    
    return best_model, feature_importance

def save_model_and_results(model, feature_importance, feature_cols):
    """Save trained model and results"""
    
    print("Saving model and results...")
    
    try:
        import pickle
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = models_dir / "reliance_5min_xgboost.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save feature names
        features_path = models_dir / "reliance_5min_features.txt"
        with open(features_path, 'w') as f:
            for feature in feature_cols:
                f.write(f"{feature}\n")
        
        # Save feature importance
        importance_path = models_dir / "reliance_5min_feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        
        print(f"Model saved to: {model_path}")
        print(f"Features saved to: {features_path}")
        print(f"Feature importance saved to: {importance_path}")
        
    except Exception as e:
        print(f"Error saving model: {e}")

def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("RELIANCE 5-MIN INTRADAY XGBOOST TRAINING")
    print("=" * 60)
    
    try:
        # Step 1: Load data
        train_df = load_reliance_training_data()
        test_df = load_reliance_testing_data()
        
        # Step 2: Create technical indicators
        train_df = create_technical_indicators(train_df)
        test_df = create_technical_indicators(test_df)
        
        # Step 3: Generate targets using the Triple Barrier Method with a 2:1 Reward/Risk ratio
        train_df = generate_targets_triple_barrier(
            train_df, atr_multiplier_profit=2.0, atr_multiplier_loss=1.0, time_horizon_candles=8
        )
        test_df = generate_targets_triple_barrier(
            test_df, atr_multiplier_profit=2.0, atr_multiplier_loss=1.0, time_horizon_candles=8
        )
        
        # Step 4: Prepare training data
        X_train, y_train, X_test, y_test, feature_cols = prepare_training_data(train_df, test_df)
        
        # Step 5: Train model
        model, feature_importance = train_xgboost_model(X_train, y_train, X_test, y_test, feature_cols)
        
        if model is not None:
            # Step 6: Save results
            save_model_and_results(model, feature_importance, feature_cols)
            
            print(f"\n" + "=" * 60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("Your Reliance 5-min intraday model is ready!")
            print("=" * 60)
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()