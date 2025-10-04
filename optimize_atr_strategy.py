"""
ATR-Based Strategy Optimizer with Breakeven & Trailing Stops
Tests different combinations of:
- Stop loss ATR multipliers
- Profit target ATR multipliers
- Breakeven trigger points
- Trailing stop strategies
- EMA trend filters
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import time
import joblib
import json
from itertools import product
import warnings
warnings.filterwarnings('ignore')


class ATRStrategyOptimizer:
    """Optimize ATR-based strategy with advanced trailing stops"""
    
    def __init__(self, model_path):
        print(f"📦 Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['features']
        print(f"✅ Model loaded with {len(self.feature_names)} features")
        
        self.trades = []
        self.results = []
    
    def load_data(self, data_path):
        """Load and prepare test data with features"""
        print(f"\n📊 Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Check if features already exist
        if not all(feat in df.columns for feat in self.feature_names):
            print("⚙️  Generating features...")
            df = self.create_features(df)
            df = df.dropna().reset_index(drop=True)
        
        print(f"✅ Loaded {len(df)} rows with features")
        return df
    
    def create_features(self, df):
        """Create features exactly as in training"""
        import pandas_ta as ta
        
        # Core EMA features
        df['EMA_21'] = ta.ema(df['close'], length=21)
        df['Distance_From_EMA21_Pct'] = (df['close'] - df['EMA_21']) / df['EMA_21'] * 100
        
        # EMA crosses
        df['EMA21_Cross_Above'] = ((df['close'].shift(1) <= df['EMA_21'].shift(1)) & 
                                   (df['close'] > df['EMA_21'])).astype(int)
        df['EMA21_Cross_Below'] = ((df['close'].shift(1) >= df['EMA_21'].shift(1)) & 
                                   (df['close'] < df['EMA_21'])).astype(int)
        
        # Cross history
        for lookback in [2, 3, 5, 10]:
            df[f'Crosses_Above_Last_{lookback}'] = df['EMA21_Cross_Above'].rolling(lookback).sum()
            df[f'Crosses_Below_Last_{lookback}'] = df['EMA21_Cross_Below'].rolling(lookback).sum()
        
        # Distance trends
        df['Distance_EMA_Change'] = df['Distance_From_EMA21_Pct'].diff()
        df['Distance_EMA_Trend'] = df['Distance_EMA_Change'].rolling(3).mean()
        
        # ADX
        adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['ADX'] = adx_data['ADX_14']
        df['ADX_Change'] = df['ADX'].diff()
        
        # ADX categories
        df['ADX_Very_Weak'] = (df['ADX'] < 20).astype(int)
        df['ADX_Weak'] = ((df['ADX'] >= 20) & (df['ADX'] < 25)).astype(int)
        df['ADX_Optimal'] = ((df['ADX'] >= 25) & (df['ADX'] < 35)).astype(int)
        df['ADX_Strong'] = ((df['ADX'] >= 35) & (df['ADX'] < 50)).astype(int)
        df['ADX_Very_Strong'] = (df['ADX'] >= 50).astype(int)
        
        # Time features
        df['Hour'] = df['datetime'].dt.hour
        df['Minute'] = df['datetime'].dt.minute
        df['Time_Slot'] = df['Hour'] * 60 + df['Minute']
        df['Is_9_15_to_9_30'] = ((df['Hour'] == 9) & (df['Minute'] < 30)).astype(int)
        df['Is_9_30_to_10_00'] = ((df['Hour'] == 9) & (df['Minute'] >= 30)).astype(int)
        df['Is_10_00_to_10_30'] = ((df['Hour'] == 10) & (df['Minute'] < 30)).astype(int)
        df['Is_10_30_to_11_00'] = ((df['Hour'] == 10) & (df['Minute'] >= 30)).astype(int)
        df['Is_11_00_to_12_00'] = (df['Hour'] == 11).astype(int)
        
        # Candle features
        df['Candle_Body_Pct'] = abs(df['close'] - df['open']) / df['open'] * 100
        df['Candle_Range_Pct'] = (df['high'] - df['low']) / df['low'] * 100
        df['Candle_Efficiency'] = df['Candle_Body_Pct'] / (df['Candle_Range_Pct'] + 1e-10)
        
        # Candle size categories
        df['Micro_Candle'] = (df['Candle_Body_Pct'] < 0.05).astype(int)
        df['Tiny_Candle'] = ((df['Candle_Body_Pct'] >= 0.05) & (df['Candle_Body_Pct'] < 0.1)).astype(int)
        df['Small_Candle'] = ((df['Candle_Body_Pct'] >= 0.1) & (df['Candle_Body_Pct'] < 0.2)).astype(int)
        df['Medium_Candle'] = (df['Candle_Body_Pct'] >= 0.2).astype(int)
        
        # Candle direction
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
        
        # Volume categories
        df['Very_Low_Volume'] = (df['Volume_Ratio'] < 0.5).astype(int)
        df['Low_Volume'] = ((df['Volume_Ratio'] >= 0.5) & (df['Volume_Ratio'] < 0.8)).astype(int)
        df['Normal_Volume'] = ((df['Volume_Ratio'] >= 0.8) & (df['Volume_Ratio'] < 1.2)).astype(int)
        df['High_Volume'] = (df['Volume_Ratio'] >= 1.2).astype(int)
        
        # Composite signals
        df['EMA_ADX_Signal'] = ((df['Distance_From_EMA21_Pct'] > 0) & (df['ADX'] > 25)).astype(int)
        df['Volume_Candle_Signal'] = ((df['Volume_Ratio'] > 1.0) & (df['Green_Candle'] == 1)).astype(int)
        df['Time_EMA_Signal'] = ((df['Hour'] >= 10) & (df['Hour'] < 14) & (df['Distance_From_EMA21_Pct'] > 0)).astype(int)
        
        # ATR for stop loss
        atr_data = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['ATR'] = atr_data
        
        return df

    
    def generate_signals(self, df, confidence_threshold=0.80, min_adx=30, 
                        ema_trend_filter=False, ema_period=50):
        """Generate trading signals with filters"""
        
        # Get predictions
        X = df[self.feature_names].values
        probabilities = self.model.predict_proba(X)[:, 1]
        
        df['signal_prob'] = probabilities
        df['signal'] = 0
        
        # Calculate EMA for trend filter if needed
        if ema_trend_filter:
            df[f'EMA_{ema_period}'] = df['close'].ewm(span=ema_period, adjust=False).mean()
            df['ema_slope'] = df[f'EMA_{ema_period}'].diff(5)  # 5-candle slope
        
        # Apply filters
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # Filter 1: Confidence
            if row['signal_prob'] < confidence_threshold:
                continue
            
            # Filter 2: ADX
            if row['ADX'] < min_adx:
                continue
            
            # Filter 3: Trading hours (10 AM - 2 PM)
            if not (time(10, 0) <= row['datetime'].time() <= time(14, 0)):
                continue
            
            # Filter 4: EMA trend (optional)
            if ema_trend_filter and idx >= 5:
                if row['ema_slope'] <= 0:  # Only trade in uptrend
                    continue
            
            df.at[idx, 'signal'] = 1
        
        return df
    
    def simulate_trades_advanced(self, df, stop_loss_atr=1.5, profit_target_atr=2.0,
                                 breakeven_rr=1.0, trailing_start_rr=1.5, 
                                 trailing_atr=1.0, max_holding_candles=15):
        """
        Advanced trade simulation with breakeven and trailing stops
        
        Parameters:
        - stop_loss_atr: Initial stop loss (e.g., 1.5 ATR below entry)
        - profit_target_atr: Profit target (e.g., 2.0 ATR above entry)
        - breakeven_rr: Move stop to breakeven when price reaches this R:R (e.g., 1.0 = 1:1)
        - trailing_start_rr: Start trailing when price reaches this R:R (e.g., 1.5 = 1.5:1)
        - trailing_atr: Trailing stop distance (e.g., 1.0 ATR below highest price)
        - max_holding_candles: Maximum candles to hold
        """
        
        trades = []
        in_trade = False
        entry_idx = None
        entry_price = None
        entry_time = None
        stop_loss_price = None
        profit_target_price = None
        highest_price = None
        initial_risk = None
        breakeven_triggered = False
        trailing_active = False
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            if in_trade:
                current_price = row['close']
                candles_held = idx - entry_idx
                
                # Update highest price
                if current_price > highest_price:
                    highest_price = current_price
                
                # Calculate current R:R
                current_gain = highest_price - entry_price
                current_rr = current_gain / initial_risk if initial_risk > 0 else 0
                
                # Breakeven logic: Move stop to entry when target R:R reached
                if not breakeven_triggered and current_rr >= breakeven_rr:
                    stop_loss_price = entry_price  # Move to breakeven
                    breakeven_triggered = True
                
                # Trailing stop logic: Start trailing when target R:R reached
                if not trailing_active and current_rr >= trailing_start_rr:
                    trailing_active = True
                
                # Update trailing stop if active
                if trailing_active:
                    atr_value = row['ATR']
                    new_trailing_stop = highest_price - (atr_value * trailing_atr)
                    if new_trailing_stop > stop_loss_price:
                        stop_loss_price = new_trailing_stop
                
                # Exit conditions
                exit_reason = None
                exit_price = current_price
                
                # 1. Profit target hit
                if row['high'] >= profit_target_price:
                    exit_price = profit_target_price
                    exit_reason = 'profit_target'
                
                # 2. Stop loss hit
                elif row['low'] <= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_reason = 'stop_loss'
                
                # 3. Max holding period
                elif candles_held >= max_holding_candles:
                    exit_reason = 'max_holding'
                
                # 4. End of day
                elif row['datetime'].time() >= time(15, 15):
                    exit_reason = 'eod'
                
                # Exit trade
                if exit_reason:
                    pnl_pct = (exit_price - entry_price) / entry_price
                    pnl_points = exit_price - entry_price
                    
                    trade = {
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_time': row['datetime'],
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl_pct': pnl_pct * 100,
                        'pnl_points': pnl_points,
                        'pnl_r': pnl_points / initial_risk if initial_risk > 0 else 0,
                        'candles_held': candles_held,
                        'win': 1 if pnl_pct > 0 else 0,
                        'highest_price': highest_price,
                        'max_rr_reached': current_rr,
                        'breakeven_triggered': breakeven_triggered,
                        'trailing_triggered': trailing_active
                    }
                    
                    trades.append(trade)
                    in_trade = False
                    breakeven_triggered = False
                    trailing_active = False
                    
                continue
            
            # Check for entry signal
            if row['signal'] == 1:
                # Enter trade
                in_trade = True
                entry_idx = idx
                entry_price = row['close']
                entry_time = row['datetime']
                highest_price = entry_price
                
                # ATR-based stop loss and profit target
                atr_value = row['ATR']
                stop_loss_price = entry_price - (atr_value * stop_loss_atr)
                profit_target_price = entry_price + (atr_value * profit_target_atr)
                
                initial_risk = entry_price - stop_loss_price
        
        # Close any open trade at end
        if in_trade:
            row = df.iloc[-1]
            pnl_pct = (row['close'] - entry_price) / entry_price
            pnl_points = row['close'] - entry_price
            
            trade = {
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': row['datetime'],
                'exit_price': row['close'],
                'exit_reason': 'end_of_data',
                'pnl_pct': pnl_pct * 100,
                'pnl_points': pnl_points,
                'pnl_r': pnl_points / initial_risk if initial_risk > 0 else 0,
                'candles_held': len(df) - entry_idx,
                'win': 1 if pnl_pct > 0 else 0,
                'highest_price': highest_price,
                'max_rr_reached': (highest_price - entry_price) / initial_risk if initial_risk > 0 else 0,
                'breakeven_triggered': breakeven_triggered,
                'trailing_triggered': trailing_active
            }
            trades.append(trade)
        
        return trades

    
    def calculate_metrics(self, trades):
        """Calculate performance metrics"""
        if len(trades) == 0:
            return None
        
        trades_df = pd.DataFrame(trades)
        
        total_trades = len(trades_df)
        winning_trades = trades_df['win'].sum()
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = trades_df[trades_df['win'] == 1]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['win'] == 0]['pnl_pct'].mean() if losing_trades > 0 else 0
        
        total_return = trades_df['pnl_pct'].sum()
        avg_return = trades_df['pnl_pct'].mean()
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Profit factor
        gross_profit = trades_df[trades_df['win'] == 1]['pnl_pct'].sum()
        gross_loss = abs(trades_df[trades_df['win'] == 0]['pnl_pct'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Max drawdown
        cumulative = trades_df['pnl_pct'].cumsum()
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        # R-multiples
        avg_r_multiple = trades_df['pnl_r'].mean()
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_return': total_return,
            'avg_return': avg_return,
            'expectancy': expectancy,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'avg_r_multiple': avg_r_multiple
        }
    
    def run_optimization(self, data_path):
        """Run grid search optimization"""
        
        # Load data once
        df = self.load_data(data_path)
        
        # Define parameter grid
        param_grid = {
            'stop_loss_atr': [1.0, 1.5, 2.0],
            'profit_target_atr': [1.5, 2.0, 2.5, 3.0],
            'breakeven_rr': [0.5, 1.0, 1.5],
            'trailing_start_rr': [1.0, 1.5, 2.0],
            'trailing_atr': [0.5, 1.0, 1.5],
            'ema_trend_filter': [False, True]
        }
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        print(f"\n🔍 Testing {len(combinations)} parameter combinations...")
        print("="*80)
        
        results = []
        
        for i, params in enumerate(combinations, 1):
            # Generate signals
            df_test = self.generate_signals(
                df.copy(),
                confidence_threshold=0.80,
                min_adx=30,
                ema_trend_filter=params['ema_trend_filter']
            )
            
            # Simulate trades
            trades = self.simulate_trades_advanced(
                df_test,
                stop_loss_atr=params['stop_loss_atr'],
                profit_target_atr=params['profit_target_atr'],
                breakeven_rr=params['breakeven_rr'],
                trailing_start_rr=params['trailing_start_rr'],
                trailing_atr=params['trailing_atr'],
                max_holding_candles=15
            )
            
            # Calculate metrics
            metrics = self.calculate_metrics(trades)
            
            if metrics:
                result = {**params, **metrics}
                results.append(result)
                
                if i % 10 == 0:
                    print(f"Progress: {i}/{len(combinations)} | "
                          f"Best so far: {max([r['total_return'] for r in results]):.2f}%")
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('total_return', ascending=False)
        
        return results_df


if __name__ == "__main__":
    MODEL_PATH = "models/ema_trap_balanced_ml.pkl"
    TEST_DATA = "testing data/reliance_data_5min_full_year_testing.csv"
    
    print("\n" + "="*80)
    print("🎯 ATR STRATEGY OPTIMIZER - ADVANCED TRAILING STOPS")
    print("="*80)
    
    optimizer = ATRStrategyOptimizer(MODEL_PATH)
    results_df = optimizer.run_optimization(TEST_DATA)
    
    # Save all results
    results_df.to_csv('atr_optimization_results.csv', index=False)
    print(f"\n💾 Saved all results to atr_optimization_results.csv")
    
    # Display top 10 results
    print("\n" + "="*80)
    print("🏆 TOP 10 PARAMETER COMBINATIONS")
    print("="*80)
    
    top_10 = results_df.head(10)
    
    for i, row in top_10.iterrows():
        print(f"\n#{list(top_10.index).index(i) + 1}")
        print(f"  Stop Loss: {row['stop_loss_atr']:.1f} ATR")
        print(f"  Profit Target: {row['profit_target_atr']:.1f} ATR")
        print(f"  Breakeven at: {row['breakeven_rr']:.1f}R")
        print(f"  Trailing starts at: {row['trailing_start_rr']:.1f}R")
        print(f"  Trailing distance: {row['trailing_atr']:.1f} ATR")
        print(f"  EMA Filter: {'Yes' if row['ema_trend_filter'] else 'No'}")
        print(f"  ---")
        print(f"  Total Return: {row['total_return']:.2f}%")
        print(f"  Win Rate: {row['win_rate']:.1f}%")
        print(f"  Avg Win: {row['avg_win']:.2f}% | Avg Loss: {row['avg_loss']:.2f}%")
        print(f"  Profit Factor: {row['profit_factor']:.2f}")
        print(f"  Expectancy: {row['expectancy']:.3f}%")
        print(f"  Avg R-Multiple: {row['avg_r_multiple']:.2f}R")
        print(f"  Total Trades: {row['total_trades']}")
    
    print("\n" + "="*80)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*80)
