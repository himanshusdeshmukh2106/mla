#!/usr/bin/env python3
"""
EMA Trap Strategy - Ultra Relaxed Version
Focus on core EMA reversal patterns with very lenient conditions
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
            print(f"Loaded {len(df)} data points")
            return df
    
    raise FileNotFoundError("Data file not found")

def create_ultra_relaxed_features(df):
    """Create features with ultra-relaxed conditions"""
    
    print("Creating ultra-relaxed EMA features...")
    
    import pandas_ta as ta
    
    # Basic indicators
    df['EMA_21'] = ta.ema(df['close'], length=21)
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    df['RSI'] = ta.rsi(df['close'], length=14)
    
    # Price-EMA relationship
    df['Price_Above_EMA'] = (df['close'] > df['EMA_21']).astype(int)
    df['Distance_From_EMA_Pct'] = (df['close'] - df['EMA_21']) / df['EMA_21'] * 100
    
    # EMA crosses (basic)
    df['Cross_Above'] = ((df['close'].shift(1) <= df['EMA_21'].shift(1)) & 
                        (df['close'] > df['EMA_21'])).astype(int)
    df['Cross_Below'] = ((df['close'].shift(1) >= df['EMA_21'].shift(1)) & 
                        (df['close'] < df['EMA_21'])).astype(int)
    
    # ULTRA RELAXED: Any reversal pattern
    # If price was above EMA and now crosses below = bearish signal
    # If price was below EMA and now crosses above = bullish signal
    df['Bearish_Signal'] = df['Cross_Below']
    df['Bullish_Signal'] = df['Cross_Above']
    
    # Time features (very broad)
    df['Hour'] = df.index.hour
    df['Trading_Hours'] = ((df['Hour'] >= 9) & (df['Hour'] <= 15)).astype(int)
    
    # ULTRA RELAXED conditions
    df['ADX_OK'] = (df['ADX'] >= 10).astype(int)  # Very low ADX threshold
    df['Candle_Size_Pct'] = abs(df['close'] - df['open']) / df['open'] * 100
    df['Reasonable_Candle'] = (df['Candle_Size_Pct'] <= 2.0).astype(int)  # 2% max
    
    # Additional simple features
    df['Volume_MA'] = df['volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_MA']
    df['Price_Change_Pct'] = df['close'].pct_change() * 100
    
    df.dropna(inplace=True)
    print(f"Features created: {len(df)} rows")
    
    return df

def generate_ultra_relaxed_targets(df):
    """Generate targets with very relaxed conditions"""
    
    print("Generating ultra-relaxed targets...")
    
    # VERY LOW thresholds
    profit_threshold = 0.001  # 0.1%
    loss_threshold = -0.001   # -0.1%
    
    # Multiple lookahead periods
    df['future_1'] = df['close'].shift(-1) / df['close'] - 1
    df['future_2'] = df['close'].shift(-2) / df['close'] - 1
    df['future_3'] = df['close'].shift(-3) / df['close'] - 1
    
    # ULTRA RELAXED entry conditions
    bearish_entries = (
        (df['Bearish_Signal'] == 1) &
        (df['Trading_Hours'] == 1) &
        (df['ADX_OK'] == 1) &
        (df['Reasonable_Candle'] == 1)
    )
    
    bullish_entries = (
        (df['Bullish_Signal'] == 1) &
        (df['Trading_Hours'] == 1) &
        (df['ADX_OK'] == 1) &
        (df['Reasonable_Candle'] == 1)
    )
    
    # Target: 1 if ANY of the future periods shows profit
    df['target'] = 0
    
    # Bearish: profit if price goes down
    bearish_profitable = (
        bearish_entries & (
            (df['future_1'] <= loss_threshold) |
            (df['future_2'] <= loss_threshold) |
            (df['future_3'] <= loss_threshold)
        )
    )
    
    # Bullish: profit if price goes up
    bullish_profitable = (
        bullish_entries & (
            (df['future_1'] >= profit_threshold) |
            (df['future_2'] >= profit_threshold) |
            (df['future_3'] >= profit_threshold)
        )
    )
    
    df.loc[bearish_profitable | bullish_profitable, 'target'] = 1
    
    # Analysis
    total_bearish = bearish_entries.sum()
    total_bullish = bullish_entries.sum()
    profitable_bearish = bearish_profitable.sum()
    profitable_bullish = bullish_profitable.sum()
    
    print(f"Ultra-Relaxed Analysis:")
    print(f"  Bearish entries: {total_bearish}, profitable: {profitable_bearish}")
    print(f"  Bullish entries: {total_bullish}, profitable: {profitable_bullish}")
    print(f"  Total entries: {total_bearish + total_bullish}")
    print(f"  Total profitable: {df['target'].sum()}")
    
    if (total_bearish + total_bullish) > 0:
        success_rate = (profitable_bearish + profitable_bullish) / (total_bearish + total_bullish) * 100
        print(f"  Success rate: {success_rate:.1f}%")
    
    return df

def train_simple_model(X, y, feature_names):
    """Train with simple approach"""
    
    print("Training with ultra-simple approach...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train distribution: {np.bincount(y_train)}")
    print(f"Test distribution: {np.bincount(y_test)}")
    
    # Use Random Forest (more robust than XGBoost for small datasets)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0)
    }
    
    print("Model Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Feature importance
    feature_importance = dict(zip(feature_names, model.feature_importances_))
    
    return model, metrics, feature_importance

def main():
    """Main ultra-relaxed training"""
    
    print("EMA Trap - Ultra Relaxed Training")
    print("=" * 50)
    
    try:
        # 1. Load data
        df = load_data()
        
        # 2. Create features
        features_df = create_ultra_relaxed_features(df)
        
        # 3. Generate targets
        targets_df = generate_ultra_relaxed_targets(features_df)
        
        # 4. Prepare data
        feature_cols = [
            'EMA_21', 'ADX', 'Distance_From_EMA_Pct', 'RSI',
            'Bearish_Signal', 'Bullish_Signal', 'Hour',
            'Volume_Ratio', 'Price_Change_Pct', 'Candle_Size_Pct'
        ]
        
        available_features = [f for f in feature_cols if f in targets_df.columns]
        print(f"Using {len(available_features)} features")
        
        X = targets_df[available_features].values
        y = targets_df['target'].values
        
        print(f"Dataset: {len(X)} samples")
        print(f"Target distribution: {np.bincount(y)}")
        
        positive_samples = y.sum()
        positive_pct = positive_samples / len(y) * 100
        print(f"Positive samples: {positive_samples} ({positive_pct:.2f}%)")
        
        if positive_samples < 10:
            print("❌ Still too few positive samples!")
            
            # Show some statistics
            print("\nDebugging info:")
            print(f"Bearish signals: {targets_df['Bearish_Signal'].sum()}")
            print(f"Bullish signals: {targets_df['Bullish_Signal'].sum()}")
            print(f"Trading hours: {targets_df['Trading_Hours'].sum()}")
            print(f"ADX OK: {targets_df['ADX_OK'].sum()}")
            print(f"Reasonable candles: {targets_df['Reasonable_Candle'].sum()}")
            
            return None, None, None
        
        # 5. Train model
        model, metrics, feature_importance = train_simple_model(X, y, available_features)
        
        # 6. Results
        print("\n" + "=" * 50)
        print("✅ Ultra-Relaxed Training Complete!")
        print(f"📊 Dataset: {len(targets_df)} samples")
        print(f"🎯 Positive targets: {positive_samples} ({positive_pct:.2f}%)")
        print(f"📈 Accuracy: {metrics['accuracy']:.4f}")
        print(f"📈 F1-Score: {metrics['f1_score']:.4f}")
        
        # Top features
        print(f"\n🔝 Top Features:")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:5], 1):
            print(f"  {i}. {feature}: {importance:.4f}")
        
        # Save model
        import joblib
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/ema_trap_ultra_relaxed.pkl')
        print(f"\nModel saved to: models/ema_trap_ultra_relaxed.pkl")
        
        return model, metrics, feature_importance
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    model, metrics, feature_importance = main()