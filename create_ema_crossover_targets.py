"""
EMA Crossover Strategy - Target Generation
Creates targets based on swing high/low stops and 1:1, 1:2 R:R
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def create_ema_crossover_targets(df: pd.DataFrame, 
                                 swing_periods: int = 10,
                                 rr_ratios: list = [1.0, 2.0],
                                 max_holding_periods: int = 20) -> pd.DataFrame:
    """
    Create targets for EMA crossover strategy
    
    Args:
        df: DataFrame with OHLCV and features
        swing_periods: Periods to look back for swing high/low
        rr_ratios: Risk-reward ratios to test
        max_holding_periods: Maximum candles to hold position
        
    Returns:
        DataFrame with target columns
    """
    
    print("Creating EMA Crossover targets...")
    
    data = df.copy()
    
    # Initialize target columns
    for rr in rr_ratios:
        data[f'target_long_{rr}R'] = 0
        data[f'target_short_{rr}R'] = 0
    
    print(f"  Processing {len(data)} rows...")
    
    # Process each row
    for idx in range(swing_periods, len(data) - max_holding_periods):
        if idx % 1000 == 0:
            print(f"    Progress: {idx}/{len(data)}")
        
        current_row = data.iloc[idx]
        entry_price = current_row['close']
        
        # Calculate swing levels for stops
        swing_high = data['high'].iloc[idx-swing_periods:idx].max()
        swing_low = data['low'].iloc[idx-swing_periods:idx].min()
        
        # LONG TRADE ANALYSIS
        # Stop loss at swing low
        long_stop = swing_low
        long_risk = entry_price - long_stop
        
        if long_risk > 0:  # Valid long setup
            # Calculate targets for different R:R ratios
            long_targets = {}
            for rr in rr_ratios:
                long_targets[rr] = entry_price + (long_risk * rr)
            
            # Look forward to see if targets are hit
            future_data = data.iloc[idx+1:idx+1+max_holding_periods]
            
            for rr in rr_ratios:
                target_price = long_targets[rr]
                target_hit = False
                stop_hit = False
                
                for future_idx, future_row in future_data.iterrows():
                    # Check if stop loss hit first
                    if future_row['low'] <= long_stop:
                        stop_hit = True
                        break
                    
                    # Check if target hit
                    if future_row['high'] >= target_price:
                        target_hit = True
                        break
                
                # Set target: 1 if target hit before stop, 0 otherwise
                data.at[idx, f'target_long_{rr}R'] = 1 if target_hit and not stop_hit else 0
        
        # SHORT TRADE ANALYSIS
        # Stop loss at swing high
        short_stop = swing_high
        short_risk = short_stop - entry_price
        
        if short_risk > 0:  # Valid short setup
            # Calculate targets for different R:R ratios
            short_targets = {}
            for rr in rr_ratios:
                short_targets[rr] = entry_price - (short_risk * rr)
            
            # Look forward to see if targets are hit
            future_data = data.iloc[idx+1:idx+1+max_holding_periods]
            
            for rr in rr_ratios:
                target_price = short_targets[rr]
                target_hit = False
                stop_hit = False
                
                for future_idx, future_row in future_data.iterrows():
                    # Check if stop loss hit first
                    if future_row['high'] >= short_stop:
                        stop_hit = True
                        break
                    
                    # Check if target hit
                    if future_row['low'] <= target_price:
                        target_hit = True
                        break
                
                # Set target: 1 if target hit before stop, 0 otherwise
                data.at[idx, f'target_short_{rr}R'] = 1 if target_hit and not stop_hit else 0
    
    # Calculate target statistics
    print("\n  Target Statistics:")
    for rr in rr_ratios:
        long_success_rate = data[f'target_long_{rr}R'].mean() * 100
        short_success_rate = data[f'target_short_{rr}R'].mean() * 100
        total_long_signals = data[f'target_long_{rr}R'].sum()
        total_short_signals = data[f'target_short_{rr}R'].sum()
        
        print(f"    {rr}R Long: {long_success_rate:.1f}% success ({total_long_signals} signals)")
        print(f"    {rr}R Short: {short_success_rate:.1f}% success ({total_short_signals} signals)")
    
    return data


def create_combined_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined long/short targets for unified model training
    """
    
    print("Creating combined targets...")
    
    data = df.copy()
    
    # Create combined targets (both long and short opportunities)
    for rr in [1.0, 2.0]:
        # Combined target: 1 if either long or short target is successful
        data[f'target_combined_{rr}R'] = (
            (data[f'target_long_{rr}R'] == 1) | 
            (data[f'target_short_{rr}R'] == 1)
        ).astype(int)
        
        # Direction target: 1 for long, -1 for short, 0 for no signal
        data[f'direction_{rr}R'] = 0
        data.loc[data[f'target_long_{rr}R'] == 1, f'direction_{rr}R'] = 1
        data.loc[data[f'target_short_{rr}R'] == 1, f'direction_{rr}R'] = -1
    
    # Statistics
    for rr in [1.0, 2.0]:
        combined_success = data[f'target_combined_{rr}R'].mean() * 100
        total_signals = data[f'target_combined_{rr}R'].sum()
        long_signals = (data[f'direction_{rr}R'] == 1).sum()
        short_signals = (data[f'direction_{rr}R'] == -1).sum()
        
        print(f"  {rr}R Combined: {combined_success:.1f}% success ({total_signals} total signals)")
        print(f"    Long signals: {long_signals}, Short signals: {short_signals}")
    
    return data


def analyze_signal_quality(df: pd.DataFrame) -> None:
    """
    Analyze the quality of generated signals
    """
    
    print("\n" + "="*60)
    print("SIGNAL QUALITY ANALYSIS")
    print("="*60)
    
    # Check signal distribution by features
    feature_cols = ['EMA_Cross_Above', 'EMA_Cross_Below', 'Bullish_Cross_Signal', 
                   'Bearish_Cross_Signal', 'Bullish_Retest_Signal', 'Bearish_Retest_Signal']
    
    for feature in feature_cols:
        if feature in df.columns:
            feature_signals = df[feature].sum()
            if feature_signals > 0:
                success_1r = df[df[feature] == 1]['target_combined_1.0R'].mean() * 100
                success_2r = df[df[feature] == 1]['target_combined_2.0R'].mean() * 100
                print(f"{feature}: {feature_signals} signals | 1R: {success_1r:.1f}% | 2R: {success_2r:.1f}%")
    
    # Time-based analysis
    if 'Hour' in df.columns:
        print("\nHourly Success Rates (1R):")
        hourly_stats = df.groupby('Hour')['target_combined_1.0R'].agg(['count', 'mean'])
        for hour, stats in hourly_stats.iterrows():
            if stats['count'] > 10:  # Only show hours with sufficient data
                print(f"  {hour:02d}:00 - {stats['count']} signals, {stats['mean']*100:.1f}% success")


if __name__ == "__main__":
    print("Testing EMA Crossover Target Generation...")
    
    try:
        # Load featured data
        df = pd.read_csv("ema_crossover_features_sample.csv")
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"Loaded {len(df)} rows of featured data")
        
        # Create targets
        df_with_targets = create_ema_crossover_targets(df)
        
        # Create combined targets
        df_final = create_combined_targets(df_with_targets)
        
        # Analyze signal quality
        analyze_signal_quality(df_final)
        
        # Save results
        df_final.to_csv('ema_crossover_with_targets.csv', index=False)
        print(f"\nSaved complete dataset to ema_crossover_with_targets.csv")
        
        # Show sample
        target_cols = [col for col in df_final.columns if 'target_' in col or 'direction_' in col]
        print(f"\nTarget columns created: {target_cols}")
        print(f"\nSample of targets:")
        print(df_final[target_cols].head(10))
        
    except FileNotFoundError:
        print("Featured data file not found. Please run create_ema_crossover_features.py first.")
    except Exception as e:
        print(f"Error: {e}")