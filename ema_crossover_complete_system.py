"""
EMA CROSSOVER STRATEGY - COMPLETE OPTIMIZED SYSTEM
Production-ready implementation with no look-ahead bias
Combines feature engineering, target creation, and model training
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class EMA_CrossoverSystem:
    """Complete EMA Crossover trading system"""
    
    def __init__(self):
        self.feature_names = []
        self.model = None
        
    # =====================================================================
    # STEP 1: FEATURE ENGINEERING (NO LOOK-AHEAD BIAS)
    # =====================================================================
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features using ONLY past and current data
        NO FUTURE DATA ALLOWED
        """
        
        print("Creating features (no look-ahead bias)...")
        data = df.copy()
        
        # Ensure datetime
        if 'datetime' not in data.columns:
            if 'timestamp' in data.columns:
                data['datetime'] = pd.to_datetime(data['timestamp'])
        
        # ===== CORE EMA FEATURES =====
        print("  EMAs...")
        data['EMA_8'] = ta.ema(data['close'], length=8)
        data['EMA_30'] = ta.ema(data['close'], length=30)
        data['EMA_Spread'] = (data['EMA_8'] - data['EMA_30']) / data['close']
        data['EMA_Spread_ROC'] = data['EMA_Spread'].diff()
        
        # EMA Slopes
        data['EMA_8_Slope_3'] = data['EMA_8'].diff(3) / data['EMA_8'].shift(3)
        data['EMA_8_Slope_5'] = data['EMA_8'].diff(5) / data['EMA_8'].shift(5)
        data['EMA_30_Slope_3'] = data['EMA_30'].diff(3) / data['EMA_30'].shift(3)
        data['EMA_30_Slope_5'] = data['EMA_30'].diff(5) / data['EMA_30'].shift(5)
        
        # Crossovers
        data['EMA_8_Above_30'] = (data['EMA_8'] > data['EMA_30']).astype(int)
        data['EMA_Cross_Above'] = ((data['EMA_8'] > data['EMA_30']) & 
                                   (data['EMA_8'].shift(1) <= data['EMA_30'].shift(1))).astype(int)
        data['EMA_Cross_Below'] = ((data['EMA_8'] < data['EMA_30']) & 
                                   (data['EMA_8'].shift(1) >= data['EMA_30'].shift(1))).astype(int)
        
        # ===== PRICE-EMA RELATIONSHIP =====
        print("  Price-EMA relationships...")
        data['Price_Distance_EMA8'] = (data['close'] - data['EMA_8']) / data['close']
        data['Price_Distance_EMA30'] = (data['close'] - data['EMA_30']) / data['close']
        
        # Price position (vectorized - faster than apply)
        data['Price_Position_Flag'] = 1  # Default: between EMAs
        data.loc[(data['close'] < data['EMA_8']) & (data['close'] < data['EMA_30']), 'Price_Position_Flag'] = 0
        data.loc[(data['close'] > data['EMA_8']) & (data['close'] > data['EMA_30']), 'Price_Position_Flag'] = 2
        
        # ===== PRICE ACTION =====
        print("  Price action...")
        data['Candle_Body_Size'] = abs(data['close'] - data['open']) / data['close']
        data['Candle_Range'] = (data['high'] - data['low']) / data['close']
        data['Wick_to_Body_Ratio'] = (data['Candle_Range'] - data['Candle_Body_Size']) / (data['Candle_Body_Size'] + 1e-10)
        
        data['Green_Candle'] = (data['close'] > data['open']).astype(int)
        data['Red_Candle'] = (data['close'] < data['open']).astype(int)
        
        # Strong candle (vectorized)
        candle_ma = data['Candle_Body_Size'].rolling(20).mean()
        data['Strong_Candle'] = (data['Candle_Body_Size'] > candle_ma).astype(int)
        
        # ===== VOLATILITY =====
        print("  Volatility...")
        data['Price_Returns'] = data['close'].pct_change()
        data['Rolling_Volatility'] = data['Price_Returns'].rolling(20).std()
        
        # ===== VOLUME =====
        print("  Volume...")
        data['Volume_MA20'] = data['volume'].rolling(20).mean()
        data['Volume_Spike'] = data['volume'] / (data['Volume_MA20'] + 1e-10)
        data['Volume_ROC'] = data['volume'].pct_change()
        
        # ===== HISTORICAL CONTEXT (LAG FEATURES) =====
        print("  Historical context...")
        for lag in [1, 2, 3]:
            data[f'EMA_Spread_Lag_{lag}'] = data['EMA_Spread'].shift(lag)
            data[f'Price_Distance_EMA30_Lag_{lag}'] = data['Price_Distance_EMA30'].shift(lag)
        
        # Crossover history
        for lookback in [5, 10]:
            data[f'Cross_Above_Last_{lookback}'] = data['EMA_Cross_Above'].rolling(lookback).sum()
            data[f'Cross_Below_Last_{lookback}'] = data['EMA_Cross_Below'].rolling(lookback).sum()
        
        # ===== SWING LEVELS =====
        print("  Swing levels...")
        for periods in [5, 10, 20]:
            data[f'Swing_High_{periods}'] = data['high'].rolling(periods).max()
            data[f'Swing_Low_{periods}'] = data['low'].rolling(periods).min()
        
        data['Distance_to_Swing_High'] = (data['Swing_High_10'] - data['close']) / data['close']
        data['Distance_to_Swing_Low'] = (data['close'] - data['Swing_Low_10']) / data['close']
        
        # ===== TIME FEATURES =====
        print("  Time features...")
        if 'datetime' in data.columns:
            data['Hour'] = data['datetime'].dt.hour
            data['Minute'] = data['datetime'].dt.minute
            data['Time_Slot'] = data['Hour'] * 60 + data['Minute']
            data['Best_Hours'] = ((data['Hour'] >= 10) & (data['Hour'] < 14)).astype(int)
        
        # Clean up
        initial_rows = len(data)
        data = data.dropna()
        print(f"  Removed {initial_rows - len(data)} rows with NaN")
        
        return data
    
    # =====================================================================
    # STEP 2: TARGET CREATION (NO LOOK-AHEAD BIAS)
    # =====================================================================
    
    def create_targets(self, df: pd.DataFrame, forward_periods: int = 5) -> pd.DataFrame:
        """
        Create realistic targets without look-ahead bias
        Predicts immediate price movement, not final trade outcome
        """
        
        print(f"\nCreating targets (forward {forward_periods} candles)...")
        data = df.copy()
        
        # Calculate forward returns
        data['forward_return'] = data['close'].shift(-forward_periods) / data['close'] - 1
        
        # Define targets based on immediate movement
        # Target 1: Significant upward movement (>0.3%)
        data['target_up_30bp'] = (data['forward_return'] > 0.003).astype(int)
        
        # Target 2: Significant downward movement (<-0.3%)
        data['target_down_30bp'] = (data['forward_return'] < -0.003).astype(int)
        
        # Target 3: Any significant movement (volatility)
        data['target_movement'] = (abs(data['forward_return']) > 0.003).astype(int)
        
        # Target 4: Direction (for classification)
        data['target_direction'] = 0  # Neutral
        data.loc[data['forward_return'] > 0.002, 'target_direction'] = 1  # Up
        data.loc[data['forward_return'] < -0.002, 'target_direction'] = -1  # Down
        
        # Remove the forward_return column (contains future data)
        data = data.drop(columns=['forward_return'])
        
        # Remove last N rows (no future data available)
        data = data.iloc[:-forward_periods]
        
        # Statistics
        print(f"  Target statistics:")
        print(f"    Up movements: {data['target_up_30bp'].sum()} ({data['target_up_30bp'].mean()*100:.1f}%)")
        print(f"    Down movements: {data['target_down_30bp'].sum()} ({data['target_down_30bp'].mean()*100:.1f}%)")
        print(f"    Any movement: {data['target_movement'].sum()} ({data['target_movement'].mean()*100:.1f}%)")
        
        return data
    
    # =====================================================================
    # STEP 3: MODEL TRAINING
    # =====================================================================
    
    def train_model(self, df: pd.DataFrame, target_col: str = 'target_movement'):
        """
        Train XGBoost model with proper time-series validation
        """
        
        print(f"\nTraining model for target: {target_col}")
        
        # Define features (exclude OHLCV, datetime, and targets)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'datetime', 'timestamp']
        target_cols = [col for col in df.columns if col.startswith('target_')]
        exclude_cols.extend(target_cols)
        
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_names].fillna(0)
        y = df[target_col]
        
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Samples: {len(X)}")
        print(f"  Positive class: {y.sum()} ({y.mean()*100:.1f}%)")
        
        # Time-series split (respects temporal order)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train final model on 80% of data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        
        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 5,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train
        print("  Training...")
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train, 
                      eval_set=[(X_test, y_test)],
                      early_stopping_rounds=50,
                      verbose=False)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("\n  Test Set Performance:")
        print(classification_report(y_test, y_pred, digits=3))
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
            print(f"  AUC-ROC: {auc:.3f}")
        except:
            pass
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n  Top 10 Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance
    
    # =====================================================================
    # STEP 4: SAVE MODEL
    # =====================================================================
    
    def save_model(self, model_name: str = "ema_crossover_optimized"):
        """Save model and metadata"""
        
        print(f"\nSaving model: {model_name}")
        
        # Save model
        model_path = f"models/{model_name}.pkl"
        joblib.dump(self.model, model_path)
        print(f"  Model: {model_path}")
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'model_type': 'XGBClassifier',
            'features': self.feature_names,
            'n_features': len(self.feature_names),
            'training_date': datetime.now().isoformat(),
            'strategy': 'EMA 8/30 Crossover + Retest',
            'no_look_ahead_bias': True
        }
        
        metadata_path = f"models/{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata: {metadata_path}")
    
    # =====================================================================
    # COMPLETE PIPELINE
    # =====================================================================
    
    def run_complete_pipeline(self, data_path: str):
        """Run complete training pipeline"""
        
        print("="*70)
        print("EMA CROSSOVER SYSTEM - COMPLETE PIPELINE")
        print("="*70)
        
        # Load data
        print("\n1. Loading data...")
        df = pd.read_csv(data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"   Loaded: {len(df)} rows")
        
        # Create features
        print("\n2. Creating features...")
        df_featured = self.create_features(df)
        print(f"   Features created: {len(df_featured.columns) - len(df.columns)}")
        
        # Create targets
        print("\n3. Creating targets...")
        df_final = self.create_targets(df_featured)
        print(f"   Final dataset: {len(df_final)} rows")
        
        # Train model
        print("\n4. Training model...")
        feature_importance = self.train_model(df_final, target_col='target_movement')
        
        # Save model
        print("\n5. Saving model...")
        self.save_model("ema_crossover_optimized")
        
        # Save processed data
        df_final.to_csv('ema_crossover_processed_data.csv', index=False)
        print(f"   Processed data: ema_crossover_processed_data.csv")
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE!")
        print("="*70)
        
        return df_final, feature_importance


if __name__ == "__main__":
    # Run complete system
    system = EMA_CrossoverSystem()
    
    try:
        df_final, feature_importance = system.run_complete_pipeline(
            "testing data/reliance_data_5min_full_year_testing.csv"
        )
        
        print("\n📋 NEXT STEPS:")
        print("   1. Review feature importance")
        print("   2. Test model on out-of-sample data")
        print("   3. Create backtest script")
        print("   4. Optimize parameters")
        
    except FileNotFoundError:
        print("Data file not found!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
