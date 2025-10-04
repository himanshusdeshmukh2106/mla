"""
Comprehensive Test: ALL Filter Combinations
Find why 66% win rate gives low returns
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import time
import joblib
import json
import warnings
from itertools import product
warnings.filterwarnings('ignore')


def load_model(model_path):
    """Load model and metadata"""
    model = joblib.load(model_path)
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return model, metadata['features']


def load_and_prepare_data(data_path):
    """Load data with all features"""
    import pandas_ta as ta
    
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Core EMA features
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
    df['Distance_EMA_Trend'] = df['Distance_EMA_Change'].rolling(3).mean()
    
    # ADX
    adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx_data['ADX_14']
    df['ADX_Change'] = df['ADX'].diff()
    
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
    
    df['Micro_Candle'] = (df['Candle_Body_Pct'] < 0.05).astype(int)
    df['Tiny_Candle'] = ((df['Candle_Body_Pct'] >= 0.05) & (df['Candle_Body_Pct'] < 0.1)).astype(int)
    df['Small_Candle'] = ((df['Candle_Body_Pct'] >= 0.1) & (df['Candle_Body_Pct'] < 0.2)).astype(int)
    df['Medium_Candle'] = (df['Candle_Body_Pct'] >= 0.2).astype(int)
    
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
    df['Normal_Volume'] = ((df['Volume_Ratio'] >= 0.8) & (df['Volume_Ratio'] < 1.2)).astype(int)
    df['High_Volume'] = (df['Volume_Ratio'] >= 1.2).astype(int)
    
    # Composite signals
    df['EMA_ADX_Signal'] = ((df['Distance_From_EMA21_Pct'] > 0) & (df['ADX'] > 25)).astype(int)
    df['Volume_Candle_Signal'] = ((df['Volume_Ratio'] > 1.0) & (df['Green_Candle'] == 1)).astype(int)
    df['Time_EMA_Signal'] = ((df['Hour'] >= 10) & (df['Hour'] < 14) & (df['Distance_From_EMA21_Pct'] > 0)).astype(int)
    
    # ATR
    atr_data = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ATR'] = atr_data
    
    # Swing lows
    df['Swing_Low_10'] = df['low'].rolling(10).min()
    
    # Market structure
    df['Recent_Range_20'] = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close'] * 100
    
    df = df.dropna().reset_index(drop=True)
    
    return df


def test_strategy(df, model, feature_names, config):
    """Test a strategy configuration"""
    
    # Get predictions
    X = df[feature_names].values
    probabilities = model.predict_proba(X)[:, 1]
    df['signal_prob'] = probabilities
    df['signal'] = 0
    
    # EMA trend filter
    if config.get('ema_filter'):
        df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_slope'] = df['EMA_50'].diff(5)
    
    # Generate signals
    for idx in range(len(df)):
        row = df.iloc[idx]
        
        # Base filters
        if row['signal_prob'] < config['confidence']:
            continue
        if row['ADX'] < config['min_adx']:
            continue
        if not (time(10, 0) <= row['datetime'].time() <= time(14, 0)):
            continue
        
        # Volume filter
        if config.get('volume_filter') and row['Volume_Ratio'] < config.get('min_volume', 1.2):
            continue
        
        # Strong candle filter
        if config.get('candle_filter'):
            if row['close'] <= row['open']:
                continue
            if row['Candle_Body_Pct'] < config.get('min_body', 0.15):
                continue
            candle_pos = (row['close'] - row['low']) / (row['high'] - row['low'] + 1e-10)
            if candle_pos < 0.5:
                continue
        
        # Market structure filter
        if config.get('structure_filter') and row['Recent_Range_20'] < config.get('min_range', 2.0):
            continue
        
        # EMA trend filter
        if config.get('ema_filter') and idx >= 5:
            if row['ema_slope'] <= 0:
                continue
        
        df.at[idx, 'signal'] = 1
    
    # Simulate trades
    trades = []
    in_trade = False
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        
        if in_trade:
            current_price = row['close']
            candles_held = idx - entry_idx
            
            if current_price > highest_price:
                highest_price = current_price
            
            current_rr = (highest_price - entry_price) / initial_risk if initial_risk > 0 else 0
            
            # Breakeven
            if not breakeven_triggered and current_rr >= config.get('breakeven_rr', 1.0):
                stop_loss_price = entry_price
                breakeven_triggered = True
            
            # Trailing
            if not trailing_active and current_rr >= config.get('trailing_start', 1.5):
                trailing_active = True
            
            if trailing_active:
                new_stop = highest_price - (row['ATR'] * config.get('trailing_atr', 1.0))
                if new_stop > stop_loss_price:
                    stop_loss_price = new_stop
            
            # Exit conditions
            exit_reason = None
            exit_price = current_price
            
            if row['high'] >= profit_target_price:
                exit_price = profit_target_price
                exit_reason = 'profit_target'
            elif row['low'] <= stop_loss_price:
                exit_price = stop_loss_price
                exit_reason = 'stop_loss'
            elif candles_held >= config.get('max_holding', 20):
                exit_reason = 'max_holding'
            elif row['datetime'].time() >= time(15, 15):
                exit_reason = 'eod'
            
            if exit_reason:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                pnl_r = (exit_price - entry_price) / initial_risk if initial_risk > 0 else 0
                
                trades.append({
                    'pnl_pct': pnl_pct,
                    'pnl_r': pnl_r,
                    'exit_reason': exit_reason,
                    'win': 1 if pnl_pct > 0 else 0
                })
                
                in_trade = False
                breakeven_triggered = False
                trailing_active = False
            
            continue
        
        # Entry signal
        if row['signal'] == 1:
            in_trade = True
            entry_idx = idx
            entry_price = row['close']
            highest_price = entry_price
            breakeven_triggered = False
            trailing_active = False
            
            atr_value = row['ATR']
            
            # Stop loss
            stop_loss_price = entry_price - (atr_value * config['stop_atr'])
            
            # Profit target
            initial_risk = entry_price - stop_loss_price
            profit_target_price = entry_price + (initial_risk * config['rr_ratio'])
    
    # Calculate metrics
    if len(trades) == 0:
        return None
    
    trades_df = pd.DataFrame(trades)
    win_rate = trades_df['win'].mean() * 100
    total_return = trades_df['pnl_pct'].sum()
    avg_win = trades_df[trades_df['win'] == 1]['pnl_pct'].mean() if trades_df['win'].sum() > 0 else 0
    avg_loss = trades_df[trades_df['win'] == 0]['pnl_pct'].mean() if (trades_df['win'] == 0).sum() > 0 else 0
    
    gross_profit = trades_df[trades_df['win'] == 1]['pnl_pct'].sum()
    gross_loss = abs(trades_df[trades_df['win'] == 0]['pnl_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
    avg_r = trades_df['pnl_r'].mean()
    
    return {
        'total_trades': len(trades_df),
        'win_rate': win_rate,
        'total_return': total_return,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'avg_r': avg_r
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPREHENSIVE FILTER COMBINATION TEST")
    print("="*80)
    
    # Load model and data
    print("\nLoading model and data...")
    model, features = load_model("models/ema_trap_balanced_ml.pkl")
    df = load_and_prepare_data("testing data/reliance_data_5min_full_year_testing.csv")
    print(f"Loaded {len(df)} candles")
    
    # Define parameter grid - TIGHTER STOPS + MORE FILTERS
    param_grid = {
        'stop_atr': [0.5, 0.75, 1.0, 1.25, 1.5],  # TIGHTER stops
        'rr_ratio': [1.5, 2.0, 2.5, 3.0, 4.0],  # AGGRESSIVE targets
        'ema_filter': [True],
        'volume_filter': [False, True],
        'candle_filter': [False, True],
        'structure_filter': [False, True],
        'min_volume': [1.2, 1.5],  # Test stricter volume
        'min_body': [0.15, 0.20],  # Test stricter candle size
        'min_range': [2.0, 2.5]  # Test stricter market structure
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    # Add base config to each
    for combo in combinations:
        combo.update({
            'confidence': 0.80,
            'min_adx': 30,
            'breakeven_rr': 1.0,
            'trailing_start': 1.5,
            'trailing_atr': 1.0,
            'max_holding': 20
        })
    
    print(f"\nTesting {len(combinations)} combinations...")
    print("="*80)
    
    results = []
    for i, config in enumerate(combinations, 1):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(combinations)}")
        
        metrics = test_strategy(df.copy(), model, features, config)
        
        if metrics and metrics['total_trades'] >= 5:  # Min 5 trades
            # Create filter description
            filters = []
            if config['ema_filter']:
                filters.append('EMA')
            if config['volume_filter']:
                filters.append('Vol')
            if config['candle_filter']:
                filters.append('Candle')
            if config['structure_filter']:
                filters.append('Structure')
            
            filter_desc = '+'.join(filters) if filters else 'None'
            
            result = {
                'filters': filter_desc,
                'stop_atr': config['stop_atr'],
                'rr_ratio': config['rr_ratio'],
                **metrics
            }
            results.append(result)
    
    # Display results
    results_df = pd.DataFrame(results)
    
    # Sort by different metrics
    print("\n" + "="*80)
    print("TOP 10 BY TOTAL RETURN")
    print("="*80)
    
    top_return = results_df.nlargest(10, 'total_return')
    for idx, row in top_return.iterrows():
        rank = list(top_return.index).index(idx) + 1
        print(f"\n#{rank}. Filters: {row['filters']} | Stop: {row['stop_atr']}ATR | R:R: 1:{row['rr_ratio']}")
        print(f"   Return: {row['total_return']:.2f}% | Win Rate: {row['win_rate']:.1f}% | Trades: {row['total_trades']}")
        print(f"   Avg Win: {row['avg_win']:.2f}% | Avg Loss: {row['avg_loss']:.2f}% | PF: {row['profit_factor']:.2f}")
        print(f"   Expectancy: {row['expectancy']:.3f}% | Avg R: {row['avg_r']:.2f}R")
    
    print("\n" + "="*80)
    print("TOP 10 BY WIN RATE")
    print("="*80)
    
    top_winrate = results_df.nlargest(10, 'win_rate')
    for idx, row in top_winrate.iterrows():
        rank = list(top_winrate.index).index(idx) + 1
        print(f"\n#{rank}. Filters: {row['filters']} | Stop: {row['stop_atr']}ATR | R:R: 1:{row['rr_ratio']}")
        print(f"   Win Rate: {row['win_rate']:.1f}% | Return: {row['total_return']:.2f}% | Trades: {row['total_trades']}")
        print(f"   Avg Win: {row['avg_win']:.2f}% | Avg Loss: {row['avg_loss']:.2f}% | PF: {row['profit_factor']:.2f}")
    
    print("\n" + "="*80)
    print("TOP 10 BY PROFIT FACTOR")
    print("="*80)
    
    top_pf = results_df.nlargest(10, 'profit_factor')
    for idx, row in top_pf.iterrows():
        rank = list(top_pf.index).index(idx) + 1
        print(f"\n#{rank}. Filters: {row['filters']} | Stop: {row['stop_atr']}ATR | R:R: 1:{row['rr_ratio']}")
        print(f"   PF: {row['profit_factor']:.2f} | Return: {row['total_return']:.2f}% | Win Rate: {row['win_rate']:.1f}%")
        print(f"   Avg Win: {row['avg_win']:.2f}% | Avg Loss: {row['avg_loss']:.2f}% | Trades: {row['total_trades']}")
    
    # Save results
    results_df.to_csv('all_combinations_results.csv', index=False)
    print("\n" + "="*80)
    print("Results saved to all_combinations_results.csv")
    print("="*80)
    
    # Analysis: Why low returns with high win rate?
    print("\n" + "="*80)
    print("ANALYSIS: Why Low Returns with High Win Rate?")
    print("="*80)
    
    high_winrate = results_df[results_df['win_rate'] >= 65]
    if len(high_winrate) > 0:
        print(f"\nStrategies with 65%+ win rate:")
        print(f"Average Return: {high_winrate['total_return'].mean():.2f}%")
        print(f"Average Win: {high_winrate['avg_win'].mean():.2f}%")
        print(f"Average Loss: {high_winrate['avg_loss'].mean():.2f}%")
        print(f"Average Trades: {high_winrate['total_trades'].mean():.0f}")
        print(f"\nPROBLEM: Avg Win ({high_winrate['avg_win'].mean():.2f}%) is TOO CLOSE to Avg Loss ({abs(high_winrate['avg_loss'].mean()):.2f}%)")
        print(f"SOLUTION: Need WIDER profit targets or TIGHTER stops!")
