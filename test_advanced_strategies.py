"""
Test Advanced Strategies:
1. Volume + Candle + Market Structure Filters
2. Aggressive R:R (1:2, 1:3)
3. Swing Low Stop Loss
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import time
import joblib
import json
import warnings
warnings.filterwarnings('ignore')


def load_model(model_path):
    """Load model and metadata"""
    model = joblib.load(model_path)
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return model, metadata['features']


def load_and_prepare_data(data_path):
    """Load data with features"""
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
    df['Swing_Low_5'] = df['low'].rolling(5).min()
    df['Swing_Low_10'] = df['low'].rolling(10).min()
    df['Swing_Low_20'] = df['low'].rolling(20).min()
    
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
        if config.get('volume_filter') and row['Volume_Ratio'] < 1.2:
            continue
        
        # Strong candle filter
        if config.get('candle_filter'):
            if row['close'] <= row['open']:  # Must be green
                continue
            if row['Candle_Body_Pct'] < 0.15:  # Min body size
                continue
            candle_pos = (row['close'] - row['low']) / (row['high'] - row['low'] + 1e-10)
            if candle_pos < 0.5:  # Must close in upper 50%
                continue
        
        # Market structure filter
        if config.get('structure_filter') and row['Recent_Range_20'] < 2.0:
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
            
            # Breakeven at 1R
            if not breakeven_triggered and current_rr >= 1.0:
                stop_loss_price = entry_price
                breakeven_triggered = True
            
            # Trailing at 1.5R
            if not trailing_active and current_rr >= 1.5:
                trailing_active = True
            
            if trailing_active:
                new_stop = highest_price - (row['ATR'] * 1.0)
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
            elif candles_held >= 20:
                exit_reason = 'max_holding'
            elif row['datetime'].time() >= time(15, 15):
                exit_reason = 'eod'
            
            if exit_reason:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                pnl_r = (exit_price - entry_price) / initial_risk if initial_risk > 0 else 0
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': row['datetime'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
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
            entry_time = row['datetime']
            highest_price = entry_price
            breakeven_triggered = False
            trailing_active = False
            
            atr_value = row['ATR']
            
            # Stop loss calculation
            if config['stop_type'] == 'atr':
                stop_loss_price = entry_price - (atr_value * config['stop_atr'])
            else:  # swing_low
                swing_low = df['Swing_Low_10'].iloc[idx]
                buffer = atr_value * 0.2
                stop_loss_price = swing_low - buffer
                # Max 3 ATR
                max_stop = entry_price - (atr_value * 3.0)
                if stop_loss_price < max_stop:
                    stop_loss_price = max_stop
            
            # Profit target based on risk
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
    print("ADVANCED STRATEGY TESTER")
    print("="*80)
    
    # Load model and data
    print("\nLoading model and data...")
    model, features = load_model("models/ema_trap_balanced_ml.pkl")
    df = load_and_prepare_data("testing data/reliance_data_5min_full_year_testing.csv")
    print(f"Loaded {len(df)} candles")
    
    # Define test scenarios
    scenarios = [
        {
            'name': 'Baseline (2 ATR stop, 1:1.5 R:R)',
            'confidence': 0.80,
            'min_adx': 30,
            'stop_type': 'atr',
            'stop_atr': 2.0,
            'rr_ratio': 1.5,
            'ema_filter': True,
            'volume_filter': False,
            'candle_filter': False,
            'structure_filter': False
        },
        {
            'name': 'Baseline + Volume Filter',
            'confidence': 0.80,
            'min_adx': 30,
            'stop_type': 'atr',
            'stop_atr': 2.0,
            'rr_ratio': 1.5,
            'ema_filter': True,
            'volume_filter': True,
            'candle_filter': False,
            'structure_filter': False
        },
        {
            'name': 'Baseline + Strong Candle Filter',
            'confidence': 0.80,
            'min_adx': 30,
            'stop_type': 'atr',
            'stop_atr': 2.0,
            'rr_ratio': 1.5,
            'ema_filter': True,
            'volume_filter': False,
            'candle_filter': True,
            'structure_filter': False
        },
        {
            'name': 'Baseline + Market Structure Filter',
            'confidence': 0.80,
            'min_adx': 30,
            'stop_type': 'atr',
            'stop_atr': 2.0,
            'rr_ratio': 1.5,
            'ema_filter': True,
            'volume_filter': False,
            'candle_filter': False,
            'structure_filter': True
        },
        {
            'name': 'ALL Filters Combined',
            'confidence': 0.80,
            'min_adx': 30,
            'stop_type': 'atr',
            'stop_atr': 2.0,
            'rr_ratio': 1.5,
            'ema_filter': True,
            'volume_filter': True,
            'candle_filter': True,
            'structure_filter': True
        },
        {
            'name': 'ALL Filters + 1:2 R:R',
            'confidence': 0.80,
            'min_adx': 30,
            'stop_type': 'atr',
            'stop_atr': 2.0,
            'rr_ratio': 2.0,
            'ema_filter': True,
            'volume_filter': True,
            'candle_filter': True,
            'structure_filter': True
        },
        {
            'name': 'ALL Filters + 1:3 R:R',
            'confidence': 0.80,
            'min_adx': 30,
            'stop_type': 'atr',
            'stop_atr': 2.0,
            'rr_ratio': 3.0,
            'ema_filter': True,
            'volume_filter': True,
            'candle_filter': True,
            'structure_filter': True
        },
        {
            'name': 'Swing Low Stop + 1:2 R:R + ALL Filters',
            'confidence': 0.80,
            'min_adx': 30,
            'stop_type': 'swing_low',
            'stop_atr': 0,
            'rr_ratio': 2.0,
            'ema_filter': True,
            'volume_filter': True,
            'candle_filter': True,
            'structure_filter': True
        },
        {
            'name': 'Swing Low Stop + 1:3 R:R + ALL Filters',
            'confidence': 0.80,
            'min_adx': 30,
            'stop_type': 'swing_low',
            'stop_atr': 0,
            'rr_ratio': 3.0,
            'ema_filter': True,
            'volume_filter': True,
            'candle_filter': True,
            'structure_filter': True
        }
    ]
    
    # Test all scenarios
    print("\nTesting scenarios...")
    print("="*80)
    
    results = []
    for i, config in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] {config['name']}")
        metrics = test_strategy(df.copy(), model, features, config)
        
        if metrics:
            result = {'scenario': config['name'], **metrics}
            results.append(result)
            print(f"  Trades: {metrics['total_trades']} | Win Rate: {metrics['win_rate']:.1f}% | "
                  f"Return: {metrics['total_return']:.2f}% | PF: {metrics['profit_factor']:.2f}")
        else:
            print("  No trades generated")
    
    # Display results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_return', ascending=False)
    
    print("\n" + "="*80)
    print("RESULTS RANKED BY TOTAL RETURN")
    print("="*80)
    
    for idx, row in results_df.iterrows():
        rank = list(results_df.index).index(idx) + 1
        print(f"\n#{rank}. {row['scenario']}")
        print(f"   Total Return: {row['total_return']:.2f}%")
        print(f"   Win Rate: {row['win_rate']:.1f}%")
        print(f"   Avg Win: {row['avg_win']:.2f}% | Avg Loss: {row['avg_loss']:.2f}%")
        print(f"   Profit Factor: {row['profit_factor']:.2f}")
        print(f"   Expectancy: {row['expectancy']:.3f}%")
        print(f"   Avg R-Multiple: {row['avg_r']:.2f}R")
        print(f"   Total Trades: {row['total_trades']}")
    
    # Save results
    results_df.to_csv('advanced_strategy_results.csv', index=False)
    print("\n" + "="*80)
    print("Results saved to advanced_strategy_results.csv")
    print("="*80)
