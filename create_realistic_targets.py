"""
Create Realistic Targets - No Look-Ahead Bias
Predicts outcomes the model can realistically forecast
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def create_realistic_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create targets that don't use future information inappropriately
    
    Instead of asking "Will this trade hit target?", we ask:
    "Is this a favorable setup based on immediate price action?"
    """
    
    print("Creating realistic targets (no look-ahead bias)...")
    
    data = df.copy()
    
    # =================================================================
    # TARGET 1: IMMEDIATE DIRECTION (Next 1-3 candles)
    # =================================================================
    print("  Creating immediate direction targets...")
    
    # Next candle direction
    data['next_1_return'] = data['close'].shift(-1) / data['close'] - 1
    data['next_3_return'] = data['close'].shift(-3) / data['close'] - 1
    data['next_5_return'] = data['close'].shift(-5) / data['close'] - 1
    
    # Binary targets for different thresholds
    thresholds = [0.002, 0.003, 0.005]  # 0.2%, 0.3%, 0.5%
    
    for threshold in thresholds:
        # Positive movement targets
        data[f'target_up_{int(threshold*1000)}bp_1c'] = (data['next_1_return'] > threshold).astype(int)
        data[f'target_up_{int(threshold*1000)}bp_3c'] = (data['next_3_return'] > threshold).astype(int)
        data[f'target_up_{int(threshold*1000)}bp_5c'] = (data['next_5_return'] > threshold).astype(int)
        
        # Negative movement targets
        data[f'target_down_{int(threshold*1000)}bp_1c'] = (data['next_1_return'] < -threshold).astype(int)
        data[f'target_down_{int(threshold*1000)}bp_3c'] = (data['next_3_return'] < -threshold).astype(int)
        data[f'target_down_{int(threshold*1000)}bp_5c'] = (data['next_5_return'] < -threshold).astype(int)
    
    # =================================================================
    # TARGET 2: VOLATILITY BREAKOUT (Realistic)
    # =================================================================
    print("  Creating volatility breakout targets...")
    
    # Calculate recent volatility
    data['recent_volatility'] = data['close'].pct_change().rolling(20).std()
    
    # Target: Will next few candles have above-average movement?
    data['next_3_volatility'] = data['close'].pct_change().shift(-3).rolling(3).std()
    data['target_high_volatility'] = (
        data['next_3_volatility'] > data['recent_volatility'] * 1.5
    ).astype(int)
    
    # =================================================================
    # TARGET 3: TREND CONTINUATION (Pattern-based)
    # =================================================================
    print("  Creating trend continuation targets...")
    
    # Current trend strength
    data['ema_trend_strength'] = abs(data['EMA_Spread'])
    data['trend_direction'] = np.sign(data['EMA_Spread'])
    
    # Target: Will trend continue in same direction?
    data['future_trend_direction'] = np.sign(data['EMA_Spread'].shift(-5))
    data['target_trend_continuation'] = (
        data['trend_direction'] == data['future_trend_direction']
    ).astype(int)
    
    # =================================================================
    # TARGET 4: SETUP QUALITY (Most Realistic)
    # =================================================================
    print("  Creating setup quality targets...")
    
    # Define what makes a "good" setup based on current conditions
    def calculate_setup_quality(row):
        score = 0
        
        # EMA alignment
        if row['EMA_8'] > row['EMA_30'] and row['EMA_30_Slope_5'] > 0:
            score += 1  # Bullish alignment
        elif row['EMA_8'] < row['EMA_30'] and row['EMA_30_Slope_5'] < 0:
            score += 1  # Bearish alignment
        
        # Volume confirmation
        if row['Volume_Spike'] > 1.2:
            score += 1
        
        # Strong candle
        if row['Strong_Candle'] == 1:
            score += 1
        
        # Good time of day
        if row['Best_Hours'] == 1:
            score += 1
        
        # Not overextended
        if abs(row['Price_Distance_EMA8']) < 0.01:  # Within 1%
            score += 1
        
        return score
    
    data['setup_quality_score'] = data.apply(calculate_setup_quality, axis=1)
    
    # Target: High-quality setups (score >= 4 out of 5)
    data['target_high_quality_setup'] = (data['setup_quality_score'] >= 4).astype(int)
    
    # =================================================================
    # TARGET 5: CROSSOVER SUCCESS (Realistic Version)
    # =================================================================
    print("  Creating crossover success targets...")
    
    # Instead of "will hit target", predict "is this a valid crossover"
    def is_valid_crossover(row, idx):
        if idx < 5 or idx >= len(data) - 5:
            return 0
        
        # Bullish crossover validation
        if row['EMA_Cross_Above'] == 1:
            # Check if trend continues for next few candles
            future_ema_spread = data['EMA_Spread'].iloc[idx+1:idx+4].mean()
            if future_ema_spread > row['EMA_Spread']:
                return 1
        
        # Bearish crossover validation
        if row['EMA_Cross_Below'] == 1:
            # Check if trend continues for next few candles
            future_ema_spread = data['EMA_Spread'].iloc[idx+1:idx+4].mean()
            if future_ema_spread < row['EMA_Spread']:
                return 1
        
        return 0
    
    data['target_valid_crossover'] = 0
    for idx in range(5, len(data) - 5):
        data.at[idx, 'target_valid_crossover'] = is_valid_crossover(data.iloc[idx], idx)
    
    # =================================================================
    # CLEAN UP
    # =================================================================
    print("  Cleaning up...")
    
    # Remove the intermediate return columns (they contain future data)
    return_cols = [col for col in data.columns if 'next_' in col and 'return' in col]
    data = data.drop(columns=return_cols + ['future_trend_direction', 'next_3_volatility'])
    
    # Get target columns
    target_cols = [col for col in data.columns if col.startswith('target_')]
    
    print(f"  Created {len(target_cols)} realistic targets")
    
    # Show target statistics
    print("\n  Target Statistics:")
    for col in target_cols:
        success_rate = data[col].mean() * 100
        total_signals = data[col].sum()
        print(f"    {col}: {success_rate:.1f}% ({total_signals} signals)")
    
    return data


if __name__ == "__main__":
    print("Creating Realistic Targets (No Look-Ahead Bias)")
    print("="*60)
    
    try:
        # Load featured data
        df = pd.read_csv("ema_crossover_features_sample.csv")
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"Loaded {len(df)} rows")
        
        # Create realistic targets
        df_realistic = create_realistic_targets(df)
        
        # Save results
        df_realistic.to_csv('ema_crossover_realistic_targets.csv', index=False)
        print(f"\nSaved to ema_crossover_realistic_targets.csv")
        
        # Show sample
        target_cols = [col for col in df_realistic.columns if col.startswith('target_')]
        print(f"\nRealistic target columns: {len(target_cols)}")
        for col in target_cols:
            print(f"  {col}")
        
    except FileNotFoundError:
        print("Feature file not found. Run create_ema_crossover_features.py first.")
    except Exception as e:
        print(f"Error: {e}")