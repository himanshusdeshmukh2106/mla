"""
Parameter Optimization Script
Tests different combinations of:
- Confidence thresholds
- ADX levels
- ATR multipliers (stop loss)
- Profit targets
- Trailing stops
"""

import pandas as pd
import numpy as np
import joblib
from datetime import time
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from src.logger import get_logger
logger = get_logger(__name__)


class ParameterOptimizer:
    """Optimize trading parameters through grid search"""
    
    def __init__(self, model_path, test_data_path):
        """Initialize optimizer"""
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load metadata
        import json
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_names = self.metadata['features']
        
        # Load and prepare test data
        self.df = self.load_test_data(test_data_path)
        
        # Results storage
        self.results = []
    
    def load_test_data(self, test_file):
        """Load and prepare test data"""
        logger.info(f"Loading test data from {test_file}")
        
        df = pd.read_csv(test_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        
        # Create features (same as training)
        df = self.create_features(df)
        
        # Calculate ATR
        import pandas_ta as ta
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['ATR_pct'] = (df['ATR'] / df['close']) * 100
        
        df = df.dropna().reset_index()
        
        logger.info(f"Prepared {len(df)} candles for optimization")
        return df
    
    def create_features(self, df):
        """Create features (same as training)"""
        import pandas_ta as ta
        
        # EMA features
        df['EMA_21'] = ta.ema(df['close'], length=21)
        df['Distance_From_EMA21_Pct'] = (df['close'] - df['EMA_21']) / df['EMA_21'] * 100
        
        df['EMA21_Cross_Above'] = ((df['close'].shift(1) <= df['EMA_21'].shift(1)) & 
                                   (df['close'] > df['EMA_21'])).astype(int)
        df['EMA21_Cross_Below'] = ((df['close'].shift(1) >= df['EMA_21'].shift(1)) & 
                                   (df['close'] < df['EMA_21'])).astype(int)
        
        for lookback in [2, 3, 5, 10]:
            df[f'Crosses_Above_Last_{lookback}'] = df['EMA21_Cross_Above'].rolling(lookback).sum()
            df[f'Crosses_Below_Last_{lookback}'] = df['EMA21_Cross_Below'].rolling(lookback).sum()
        
        df['Distance_EMA_Change'] = df['Distance_From_EMA21_Pct'].diff()
        df['Distance_EMA_Trend'] = df['Distance_From_EMA21_Pct'].rolling(3).mean()
        
        # ADX
        adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['ADX'] = adx_result['ADX_14']
        df['ADX_Change'] = df['ADX'].diff()
        
        df['ADX_Very_Weak'] = (df['ADX'] < 15).astype(int)
        df['ADX_Weak'] = ((df['ADX'] >= 15) & (df['ADX'] < 20)).astype(int)
        df['ADX_Optimal'] = ((df['ADX'] >= 20) & (df['ADX'] <= 30)).astype(int)
        df['ADX_Strong'] = ((df['ADX'] > 30) & (df['ADX'] <= 40)).astype(int)
        df['ADX_Very_Strong'] = (df['ADX'] > 40).astype(int)
        
        # Time features
        df['Hour'] = df.index.hour
        df['Minute'] = df.index.minute
        df['Time_Slot'] = (df['Hour'] * 60 + df['Minute']) // 15
        
        df['Is_9_15_to_9_30'] = ((df['Hour'] == 9) & (df['Minute'].between(15, 30))).astype(int)
        df['Is_9_30_to_10_00'] = ((df['Hour'] == 9) & (df['Minute'] > 30)).astype(int)
        df['Is_10_00_to_10_30'] = ((df['Hour'] == 10) & (df['Minute'] <= 30)).astype(int)
        df['Is_10_30_to_11_00'] = ((df['Hour'] == 10) & (df['Minute'] > 30)).astype(int)
        df['Is_11_00_to_12_00'] = (df['Hour'] == 11).astype(int)
        
        # Candle features
        df['Candle_Body_Pct'] = abs(df['close'] - df['open']) / df['open'] * 100
        df['Candle_Range_Pct'] = (df['high'] - df['low']) / df['open'] * 100
        df['Candle_Efficiency'] = np.where(df['Candle_Range_Pct'] > 0,
                                           df['Candle_Body_Pct'] / df['Candle_Range_Pct'], 0)
        
        df['Micro_Candle'] = (df['Candle_Body_Pct'] <= 0.10).astype(int)
        df['Tiny_Candle'] = ((df['Candle_Body_Pct'] > 0.10) & (df['Candle_Body_Pct'] <= 0.15)).astype(int)
        df['Small_Candle'] = ((df['Candle_Body_Pct'] > 0.15) & (df['Candle_Body_Pct'] <= 0.25)).astype(int)
        df['Medium_Candle'] = ((df['Candle_Body_Pct'] > 0.25) & (df['Candle_Body_Pct'] <= 0.50)).astype(int)
        
        df['Green_Candle'] = (df['close'] > df['open']).astype(int)
        df['Red_Candle'] = (df['close'] < df['open']).astype(int)
        
        # Price momentum
        df['Price_Change_1'] = df['close'].pct_change(1) * 100
        df['Price_Change_3'] = df['close'].pct_change(3) * 100
        df['Price_Change_5'] = df['close'].pct_change(5) * 100
        df['Price_Momentum'] = df['Price_Change_1'].rolling(3).mean()
        
        # Volume features
        df['Volume_Ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['Volume_Change'] = df['volume'].pct_change() * 100
        
        df['Very_Low_Volume'] = (df['Volume_Ratio'] < 0.5).astype(int)
        df['Low_Volume'] = ((df['Volume_Ratio'] >= 0.5) & (df['Volume_Ratio'] < 0.8)).astype(int)
        df['Normal_Volume'] = ((df['Volume_Ratio'] >= 0.8) & (df['Volume_Ratio'] <= 1.2)).astype(int)
        df['High_Volume'] = (df['Volume_Ratio'] > 1.2).astype(int)
        
        # Interaction features
        df['EMA_ADX_Signal'] = df['Distance_From_EMA21_Pct'] * df['ADX'] / 100
        df['Volume_Candle_Signal'] = df['Volume_Ratio'] * df['Candle_Body_Pct']
        df['Time_EMA_Signal'] = df['Time_Slot'] * abs(df['Distance_From_EMA21_Pct'])
        
        return df
    
    def backtest_parameters(self, confidence_threshold, min_adx, atr_multiplier, 
                           profit_target_atr, trailing_stop_atr, max_holding):
        """Backtest with specific parameters"""
        
        # Generate signals
        X = self.df[self.feature_names].values
        probabilities = self.model.predict_proba(X)[:, 1]
        
        self.df['signal_prob'] = probabilities
        self.df['signal'] = 0
        
        # Apply filters
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            
            if row['signal_prob'] < confidence_threshold:
                continue
            if row['ADX'] < min_adx:
                continue
            if row['datetime'].time() < time(9, 30):
                continue
            if not (time(10, 0) <= row['datetime'].time() <= time(14, 0)):
                continue
            
            self.df.at[idx, 'signal'] = 1
        
        # Simulate trades
        trades = []
        in_trade = False
        entry_idx = None
        entry_price = None
        stop_loss_price = None
        profit_targets = []
        highest_price = None
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            
            if in_trade:
                current_price = row['close']
                
                # Update highest price and trailing stop
                if current_price > highest_price:
                    highest_price = current_price
                    # Trailing stop based on ATR
                    atr_value = row['ATR']
                    new_trailing_stop = highest_price - (atr_value * trailing_stop_atr)
                    if new_trailing_stop > stop_loss_price:
                        stop_loss_price = new_trailing_stop
                
                # Check exits
                exit_reason = None
                exit_price = current_price
                
                # Check profit targets (multiple levels)
                for i, target in enumerate(profit_targets):
                    if row['high'] >= target:
                        exit_price = target
                        exit_reason = f'profit_target_{i+1}'
                        break
                
                # Stop loss
                if exit_reason is None and row['low'] <= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_reason = 'stop_loss'
                
                # Max holding
                if exit_reason is None and (idx - entry_idx) >= max_holding:
                    exit_reason = 'max_holding'
                
                # EOD
                if exit_reason is None and row['datetime'].time() >= time(15, 15):
                    exit_reason = 'eod'
                
                if exit_reason:
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'win': 1 if pnl_pct > 0 else 0,
                        'exit_reason': exit_reason
                    })
                    
                    in_trade = False
                
                continue
            
            # Check for entry
            if row['signal'] == 1:
                # Confirmation
                if idx > 0 and self.df.iloc[idx-1]['signal'] == 0:
                    continue
                
                in_trade = True
                entry_idx = idx
                entry_price = row['close']
                highest_price = entry_price
                
                # Calculate stop loss and profit targets based on ATR
                atr_value = row['ATR']
                stop_loss_price = entry_price - (atr_value * atr_multiplier)
                
                # Multiple profit targets (1x, 2x, 3x ATR)
                profit_targets = [
                    entry_price + (atr_value * profit_target_atr),
                    entry_price + (atr_value * profit_target_atr * 2),
                    entry_price + (atr_value * profit_target_atr * 3)
                ]
        
        # Calculate metrics
        if not trades:
            return None
        
        trades_df = pd.DataFrame(trades)
        
        total_trades = len(trades_df)
        winning_trades = trades_df['win'].sum()
        win_rate = (winning_trades / total_trades) * 100
        
        total_pnl = trades_df['pnl_pct'].sum()
        avg_pnl = trades_df['pnl_pct'].mean()
        
        winning_pnl = trades_df[trades_df['win'] == 1]['pnl_pct'].sum()
        losing_pnl = abs(trades_df[trades_df['win'] == 0]['pnl_pct'].sum())
        
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')
        
        return {
            'confidence_threshold': confidence_threshold,
            'min_adx': min_adx,
            'atr_multiplier': atr_multiplier,
            'profit_target_atr': profit_target_atr,
            'trailing_stop_atr': trailing_stop_atr,
            'max_holding': max_holding,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'profit_factor': profit_factor
        }
    
    def optimize(self):
        """Run optimization grid search"""
        
        print("\n" + "="*70)
        print("PARAMETER OPTIMIZATION - GRID SEARCH")
        print("="*70)
        
        # Define parameter grid
        param_grid = {
            'confidence_threshold': [0.70, 0.75, 0.80, 0.85],
            'min_adx': [25, 28, 30, 32],
            'atr_multiplier': [1.0, 1.5, 2.0],  # Stop loss
            'profit_target_atr': [1.0, 1.5, 2.0],  # First profit target
            'trailing_stop_atr': [0.5, 0.75, 1.0],  # Trailing stop
            'max_holding': [15, 20, 25]
        }
        
        # Calculate total combinations
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"\nTotal combinations to test: {total_combinations}")
        print(f"This will take approximately {total_combinations * 2 / 60:.1f} minutes\n")
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(product(*values))
        
        print("Starting optimization...")
        
        for i, combo in enumerate(combinations, 1):
            params = dict(zip(keys, combo))
            
            if i % 10 == 0:
                print(f"Progress: {i}/{total_combinations} ({i/total_combinations*100:.1f}%)")
            
            result = self.backtest_parameters(**params)
            
            if result:
                self.results.append(result)
        
        print(f"\n✅ Optimization complete! Tested {len(self.results)} valid combinations")
    
    def analyze_results(self):
        """Analyze and display results"""
        
        if not self.results:
            print("No results to analyze!")
            return
        
        results_df = pd.DataFrame(self.results)
        
        print("\n" + "="*70)
        print("OPTIMIZATION RESULTS")
        print("="*70)
        
        # Sort by different metrics
        print("\n🏆 TOP 10 BY TOTAL P&L:")
        top_pnl = results_df.nlargest(10, 'total_pnl')
        for idx, row in top_pnl.iterrows():
            print(f"   Conf:{row['confidence_threshold']:.2f} ADX:{row['min_adx']:.0f} "
                  f"SL:{row['atr_multiplier']:.1f}x PT:{row['profit_target_atr']:.1f}x "
                  f"Trail:{row['trailing_stop_atr']:.2f}x → "
                  f"P&L:{row['total_pnl']:+.2f}% ({row['total_trades']:.0f} trades, "
                  f"{row['win_rate']:.1f}% WR, PF:{row['profit_factor']:.2f})")
        
        print("\n🎯 TOP 10 BY PROFIT FACTOR:")
        top_pf = results_df.nlargest(10, 'profit_factor')
        for idx, row in top_pf.iterrows():
            print(f"   Conf:{row['confidence_threshold']:.2f} ADX:{row['min_adx']:.0f} "
                  f"SL:{row['atr_multiplier']:.1f}x PT:{row['profit_target_atr']:.1f}x "
                  f"Trail:{row['trailing_stop_atr']:.2f}x → "
                  f"PF:{row['profit_factor']:.2f} (P&L:{row['total_pnl']:+.2f}%, "
                  f"{row['total_trades']:.0f} trades, {row['win_rate']:.1f}% WR)")
        
        print("\n📊 TOP 10 BY WIN RATE:")
        top_wr = results_df.nlargest(10, 'win_rate')
        for idx, row in top_wr.iterrows():
            print(f"   Conf:{row['confidence_threshold']:.2f} ADX:{row['min_adx']:.0f} "
                  f"SL:{row['atr_multiplier']:.1f}x PT:{row['profit_target_atr']:.1f}x "
                  f"Trail:{row['trailing_stop_atr']:.2f}x → "
                  f"WR:{row['win_rate']:.1f}% (P&L:{row['total_pnl']:+.2f}%, "
                  f"{row['total_trades']:.0f} trades, PF:{row['profit_factor']:.2f})")
        
        # Save results
        results_df.to_csv('optimization_results.csv', index=False)
        print(f"\n💾 Full results saved to: optimization_results.csv")
        
        # Best overall (balanced)
        results_df['score'] = (results_df['total_pnl'] * 0.4 + 
                               results_df['profit_factor'] * 10 * 0.3 +
                               results_df['win_rate'] * 0.3)
        
        best = results_df.nlargest(1, 'score').iloc[0]
        
        print("\n" + "="*70)
        print("🏆 BEST OVERALL PARAMETERS (Balanced Score)")
        print("="*70)
        print(f"Confidence Threshold: {best['confidence_threshold']:.2f}")
        print(f"Minimum ADX: {best['min_adx']:.0f}")
        print(f"Stop Loss: {best['atr_multiplier']:.1f}x ATR")
        print(f"Profit Target: {best['profit_target_atr']:.1f}x ATR")
        print(f"Trailing Stop: {best['trailing_stop_atr']:.2f}x ATR")
        print(f"Max Holding: {best['max_holding']:.0f} candles")
        print(f"\nResults:")
        print(f"  Total P&L: {best['total_pnl']:+.2f}%")
        print(f"  Trades: {best['total_trades']:.0f}")
        print(f"  Win Rate: {best['win_rate']:.1f}%")
        print(f"  Profit Factor: {best['profit_factor']:.2f}")
        print("="*70)


if __name__ == "__main__":
    MODEL_PATH = "models/ema_trap_balanced_ml.pkl"
    TEST_DATA = "testing data/reliance_data_5min_full_year_testing.csv"
    
    print("🚀 Starting Parameter Optimization...")
    
    optimizer = ParameterOptimizer(MODEL_PATH, TEST_DATA)
    optimizer.optimize()
    optimizer.analyze_results()
    
    print("\n✅ Optimization Complete!")
