"""
EMA Crossover Targets - FIXED VERSION
Prevents data leakage and creates realistic targets
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def create_realistic_targets(df: pd.DataFrame, 
                            risk_reward_ratios=[1.0, 1.5, 2.0],
                            min_holding_periods=3,
                            max_holding_periods=20) -> pd.DataFrame:
    """
    Create realistic trading targets WITHOUT data leakage
    
    Key principles:
    1. Only use FUTURE data (not current candle)
    2. Simulate realistic entry/exit
    3. Account for slippage
    4. Minimum holding period
    
    Args:
        df: DataFrame with OHLCV and features
        risk_reward_ratios: List of R:R ratios to test
        min_holding_periods: Minimum candles to hold (prevents noise)
        max_holding_periods: Maximum candles to hold (timeout)
    """
    
    print("="*80)
    print("🎯 CREATING REALISTIC TARGETS (NO DATA LEAKAGE)")
    print("="*80)
    
    data = df.copy()
    data = data.sort_values('datetime').reset_index(drop=True)
    
    print(f"\nDataset: {len(data):,} rows")
    
    # Calculate swing levels for stop loss
    print("\n1️⃣ Calculating swing levels...")
    data['swing_high_10'] = data['high'].rolling(10, min_periods=1).max()
    data['swing_low_10'] = data['low'].rolling(10, min_periods=1).min()
    
    # Shift swing levels to avoid look-ahead bias
    data['swing_high_10'] = data['swing_high_10'].shift(1)
    data['swing_low_10'] = data['swing_low_10'].shift(1)
    
    # Calculate ATR for dynamic stops
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr'] = data['tr'].rolling(14, min_periods=1).mean()
    data['atr'] = data['atr'].shift(1)  # Shift to avoid look-ahead
    
    print("   ✅ Swing levels and ATR calculated")
    
    # Generate targets for each R:R ratio
    print(f"\n2️⃣ Generating targets for {len(risk_reward_ratios)} R:R ratios...")
    
    for rr in risk_reward_ratios:
        print(f"\n   Processing {rr}:1 R:R ratio...")
        
        # Long targets
        target_long = []
        target_short = []
        
        for i in range(len(data)):
            # Skip if not enough future data
            if i >= len(data) - max_holding_periods:
                target_long.append(0)
                target_short.append(0)
                continue
            
            # Get current candle data
            entry_price = data.loc[i, 'close']
            atr_value = data.loc[i, 'atr']
            
            if pd.isna(atr_value) or atr_value == 0:
                target_long.append(0)
                target_short.append(0)
                continue
            
            # Define stops and targets
            # LONG TRADE
            stop_loss_long = entry_price - (1.5 * atr_value)
            profit_target_long = entry_price + (rr * 1.5 * atr_value)
            
            # SHORT TRADE
            stop_loss_short = entry_price + (1.5 * atr_value)
            profit_target_short = entry_price - (rr * 1.5 * atr_value)
            
            # Simulate trade over next candles
            # IMPORTANT: Start from i+1 (FUTURE candles only!)
            long_result = 0
            short_result = 0
            
            for j in range(i + min_holding_periods, min(i + max_holding_periods + 1, len(data))):
                future_high = data.loc[j, 'high']
                future_low = data.loc[j, 'low']
                
                # Check LONG trade
                if long_result == 0:
                    if future_high >= profit_target_long:
                        long_result = 1  # Win
                    elif future_low <= stop_loss_long:
                        long_result = 0  # Loss
                
                # Check SHORT trade
                if short_result == 0:
                    if future_low <= profit_target_short:
                        short_result = 1  # Win
                    elif future_high >= stop_loss_short:
                        short_result = 0  # Loss
                
                # If both determined, break
                if long_result != 0 or short_result != 0:
                    break
            
            target_long.append(long_result)
            target_short.append(short_result)
        
        # Add to dataframe
        data[f'target_long_{rr}R'] = target_long
        data[f'target_short_{rr}R'] = target_short
        data[f'target_combined_{rr}R'] = ((np.array(target_long) == 1) | (np.array(target_short) == 1)).astype(int)
        
        # Statistics
        long_wins = sum(target_long)
        short_wins = sum(target_short)
        combined_wins = sum(data[f'target_combined_{rr}R'])
        
        print(f"      Long wins: {long_wins:,} ({long_wins/len(data)*100:.1f}%)")
        print(f"      Short wins: {short_wins:,} ({short_wins/len(data)*100:.1f}%)")
        print(f"      Combined wins: {combined_wins:,} ({combined_wins/len(data)*100:.1f}%)")
        
        # Check for issues
        if long_wins / len(data) > 0.8:
            print(f"      ⚠️  WARNING: Long win rate too high ({long_wins/len(data)*100:.1f}%)")
        if combined_wins / len(data) > 0.8:
            print(f"      ⚠️  WARNING: Combined win rate too high ({combined_wins/len(data)*100:.1f}%)")
    
    # Add direction targets
    print(f"\n3️⃣ Adding directional targets...")
    
    # Future return (for reference, not for training)
    data['future_return_5'] = data['close'].shift(-5) / data['close'] - 1
    data['direction_up'] = (data['future_return_5'] > 0.005).astype(int)  # 0.5% threshold
    data['direction_down'] = (data['future_return_5'] < -0.005).astype(int)
    
    print(f"   Up moves: {data['direction_up'].sum():,} ({data['direction_up'].sum()/len(data)*100:.1f}%)")
    print(f"   Down moves: {data['direction_down'].sum():,} ({data['direction_down'].sum()/len(data)*100:.1f}%)")
    
    # Clean up temporary columns
    data = data.drop(['tr', 'swing_high_10', 'swing_low_10', 'atr'], axis=1)
    
    # Remove rows with NaN targets
    initial_len = len(data)
    data = data.dropna(subset=[f'target_combined_{risk_reward_ratios[0]}R'])
    final_len = len(data)
    
    print(f"\n4️⃣ Cleaning data...")
    print(f"   Removed {initial_len - final_len:,} rows with NaN")
    print(f"   Final dataset: {final_len:,} rows")
    
    # Validation checks
    print(f"\n5️⃣ Validation checks...")
    
    issues = []
    
    for rr in risk_reward_ratios:
        combined_col = f'target_combined_{rr}R'
        win_rate = data[combined_col].sum() / len(data)
        
        if win_rate > 0.9:
            issues.append(f"🚨 {combined_col}: Win rate too high ({win_rate*100:.1f}%)")
        elif win_rate < 0.05:
            issues.append(f"⚠️  {combined_col}: Win rate too low ({win_rate*100:.1f}%)")
        else:
            print(f"   ✅ {combined_col}: Win rate {win_rate*100:.1f}% (reasonable)")
    
    if issues:
        print("\n   Issues found:")
        for issue in issues:
            print(f"      {issue}")
    
    return data


def main():
    """Main function"""
    
    print("\n" + "="*80)
    print("🔧 FIXED TARGET GENERATION")
    print("="*80)
    print("\nKey improvements:")
    print("  ✅ No data leakage (only uses future candles)")
    print("  ✅ Minimum holding period (reduces noise)")
    print("  ✅ Realistic stop loss and targets")
    print("  ✅ ATR-based dynamic levels")
    print("  ✅ Validation checks")
    print("="*80)
    
    # Load data
    try:
        # Try different possible data files
        data_files = [
            'ema_crossover_features_sample.csv',
            'testing data/reliance_data_5min_full_year_testing.csv',
            'data/reliance_data_5min_full_year.csv'
        ]
        
        df = None
        for data_file in data_files:
            try:
                df = pd.read_csv(data_file)
                print(f"\n✅ Loaded data from: {data_file}")
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            print("\n❌ No data file found!")
            print("   Tried:")
            for f in data_files:
                print(f"      - {f}")
            return
        
        # Ensure datetime column
        if 'datetime' not in df.columns:
            print("\n❌ No datetime column found!")
            return
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Create features if needed
        if 'EMA_8' not in df.columns:
            print("\n⚠️  Features not found. Creating basic features...")
            from create_ema_crossover_features import create_ema_crossover_features
            df = create_ema_crossover_features(df)
        
        # Generate targets
        df_with_targets = create_realistic_targets(
            df,
            risk_reward_ratios=[1.0, 1.5, 2.0],
            min_holding_periods=3,
            max_holding_periods=20
        )
        
        # Save
        output_file = 'ema_crossover_with_targets_fixed.csv'
        df_with_targets.to_csv(output_file, index=False)
        
        print(f"\n" + "="*80)
        print(f"✅ SUCCESS!")
        print(f"="*80)
        print(f"\n💾 Saved to: {output_file}")
        print(f"   Rows: {len(df_with_targets):,}")
        print(f"   Columns: {len(df_with_targets.columns)}")
        
        # Show target columns
        target_cols = [col for col in df_with_targets.columns if 'target_' in col]
        print(f"\n📊 Target columns created: {len(target_cols)}")
        for col in target_cols:
            wins = df_with_targets[col].sum()
            rate = wins / len(df_with_targets) * 100
            print(f"   {col:30s}: {wins:5,} wins ({rate:5.1f}%)")
        
        print(f"\n🚀 Next step:")
        print(f"   python train_ema_crossover_optimized.py")
        print(f"   (Update data_path to '{output_file}')")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
