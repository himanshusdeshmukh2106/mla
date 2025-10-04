"""
FINAL OPTIMIZED EMA Trap Backtest
- Ultra-high confidence (0.80+)
- Trailing stop loss
- Better profit targets
- Best trading hours only
"""

import sys
sys.path.insert(0, '.')

# Import the improved backtester
from backtest_ema_trap_improved import ImprovedEMATrapBacktester
import pandas as pd

class FinalOptimizedBacktester(ImprovedEMATrapBacktester):
    """Final optimized version with all improvements"""
    
    def __init__(self, model_path):
        super().__init__(model_path)
        
        # OPTIMIZED PARAMETERS (From ATR grid search optimization)
        self.atr_multiplier = 2.0       # Stop loss = 2.0 x ATR (wider stops!)
        self.profit_target_atr = 3.0    # Profit target = 3.0 x ATR (1.5:1 R:R!)
        self.trailing_stop_atr = 1.5    # Trailing stop = 1.5 x ATR
        self.breakeven_rr = 1.5         # Move to breakeven at 1.5R
        self.trailing_start_rr = 2.0    # Start trailing at 2.0R
        self.max_holding_candles = 15   # Max 15 candles (75 min)
        
        # OPTIMIZED FILTERS (From grid search)
        self.confidence_threshold = 0.80  # 80% confidence
        self.min_adx = 30                 # ADX ≥ 30
        self.best_hours_only = True       # Only trade 10 AM - 2 PM
        self.use_confirmation = True
        self.ema_trend_filter = True      # CRITICAL: Only trade in uptrends!
        
        print(f"🎯 OPTIMIZED SETTINGS (From ATR Grid Search):")
        print(f"   Confidence: {self.confidence_threshold}")
        print(f"   Min ADX: {self.min_adx}")
        print(f"   Stop Loss: {self.atr_multiplier}x ATR")
        print(f"   Profit Target: {self.profit_target_atr}x ATR (1.5:1 R:R)")
        print(f"   Breakeven: {self.breakeven_rr}R")
        print(f"   Trailing Start: {self.trailing_start_rr}R")
        print(f"   Trailing Stop: {self.trailing_stop_atr}x ATR")
        print(f"   EMA Trend Filter: {self.ema_trend_filter}")
        print(f"   Trading Hours: 10:00 AM - 2:00 PM only")
    
    def generate_signals(self, df):
        """Generate signals with ultra-strict filters including EMA trend"""
        from datetime import time
        
        # Get predictions
        X = df[self.feature_names].values
        probabilities = self.model.predict_proba(X)[:, 1]
        
        df['signal_prob'] = probabilities
        df['signal'] = 0
        
        # Calculate EMA for trend filter
        if self.ema_trend_filter:
            df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['ema_slope'] = df['EMA_50'].diff(5)  # 5-candle slope
        
        # Apply ultra-strict filters
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # Filter 1: Ultra-high confidence
            if row['signal_prob'] < self.confidence_threshold:
                continue
            
            # Filter 2: Strong ADX only
            if row['ADX'] < self.min_adx:
                continue
            
            # Filter 3: Best hours only (10 AM - 2 PM)
            if self.best_hours_only:
                if not (time(10, 0) <= row['datetime'].time() <= time(14, 0)):
                    continue
            
            # Filter 4: EMA trend (CRITICAL!)
            if self.ema_trend_filter and idx >= 5:
                if row['ema_slope'] <= 0:  # Only trade in uptrend
                    continue
            
            df.at[idx, 'signal'] = 1
        
        signals_count = df['signal'].sum()
        print(f"✅ Generated {signals_count} ultra-high-quality signals")
        
        return df
    
    def simulate_trades(self, df):
        """Simulate trades with trailing stop"""
        from datetime import time
        import logging
        logger = logging.getLogger(__name__)
        
        print("🚀 Starting final optimized simulation...")
        
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
                if not breakeven_triggered and current_rr >= self.breakeven_rr:
                    stop_loss_price = entry_price  # Move to breakeven
                    breakeven_triggered = True
                
                # Trailing stop logic: Start trailing when target R:R reached
                if not trailing_active and current_rr >= self.trailing_start_rr:
                    trailing_active = True
                
                # Update trailing stop if active
                if trailing_active:
                    atr_value = row['ATR']
                    new_trailing_stop = highest_price - (atr_value * self.trailing_stop_atr)
                    if new_trailing_stop > stop_loss_price:
                        stop_loss_price = new_trailing_stop
                
                # Exit conditions
                exit_reason = None
                exit_price = current_price
                
                # 1. Profit target hit
                if row['high'] >= profit_target_price:
                    exit_price = profit_target_price
                    exit_reason = 'profit_target'
                
                # 2. Stop loss hit (ATR-based or trailing)
                elif row['low'] <= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_reason = 'stop_loss'
                
                # 3. Max holding period
                elif candles_held >= self.max_holding_candles:
                    exit_reason = 'max_holding'
                
                # 4. End of day
                elif row['datetime'].time() >= time(15, 15):
                    exit_reason = 'eod'
                
                # Exit trade
                if exit_reason:
                    pnl_pct = (exit_price - entry_price) / entry_price
                    
                    pnl_r = (exit_price - entry_price) / initial_risk if initial_risk > 0 else 0
                    
                    trade = {
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_time': row['datetime'],
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl_pct': pnl_pct * 100,
                        'pnl_points': exit_price - entry_price,
                        'pnl_r': pnl_r,
                        'candles_held': candles_held,
                        'win': 1 if pnl_pct > 0 else 0,
                        'stop_loss_pct': (stop_loss_price - entry_price) / entry_price * 100,
                        'entry_adx': df.iloc[entry_idx]['ADX'],
                        'entry_confidence': df.iloc[entry_idx]['signal_prob'],
                        'highest_price': highest_price,
                        'max_rr_reached': current_rr,
                        'breakeven_triggered': breakeven_triggered,
                        'trailing_triggered': trailing_active
                    }
                    
                    self.trades.append(trade)
                    in_trade = False
                    breakeven_triggered = False
                    trailing_active = False
                    
                continue
            
            # Check for entry signal
            if row['signal'] == 1:
                # Confirmation candle
                if self.use_confirmation and idx > 0:
                    if df.iloc[idx-1]['signal'] == 0:
                        continue
                
                # Enter trade
                in_trade = True
                entry_idx = idx
                entry_price = row['close']
                entry_time = row['datetime']
                highest_price = entry_price
                
                # ATR-based stop loss and profit target
                atr_value = row['ATR']
                stop_loss_price = entry_price - (atr_value * self.atr_multiplier)
                profit_target_price = entry_price + (atr_value * self.profit_target_atr)
                
                initial_risk = entry_price - stop_loss_price
                profit_target_pct = (profit_target_price - entry_price) / entry_price
                
                print(f"📈 ENTRY: {entry_time.strftime('%Y-%m-%d %H:%M')} @ {entry_price:.2f} | Target: {profit_target_price:.2f} (+{profit_target_pct*100:.1f}%) | ADX: {row['ADX']:.0f} | Conf: {row['signal_prob']:.2f}")
        
        # Close any open trade
        if in_trade:
            row = df.iloc[-1]
            pnl_pct = (row['close'] - entry_price) / entry_price
            pnl_r = (row['close'] - entry_price) / initial_risk if initial_risk > 0 else 0
            current_rr = (highest_price - entry_price) / initial_risk if initial_risk > 0 else 0
            
            trade = {
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': row['datetime'],
                'exit_price': row['close'],
                'exit_reason': 'end_of_data',
                'pnl_pct': pnl_pct * 100,
                'pnl_points': row['close'] - entry_price,
                'pnl_r': pnl_r,
                'candles_held': len(df) - entry_idx,
                'win': 1 if pnl_pct > 0 else 0,
                'stop_loss_pct': (stop_loss_price - entry_price) / entry_price * 100,
                'entry_adx': df.iloc[entry_idx]['ADX'],
                'entry_confidence': df.iloc[entry_idx]['signal_prob'],
                'highest_price': highest_price,
                'max_rr_reached': current_rr,
                'breakeven_triggered': breakeven_triggered,
                'trailing_triggered': trailing_active
            }
            self.trades.append(trade)
        
        print(f"✅ Simulation complete: {len(self.trades)} trades")
    
    def save_results(self, output_file='backtest_results_final.csv'):
        """Save results"""
        super().save_results(output_file)


if __name__ == "__main__":
    MODEL_PATH = "models/ema_trap_balanced_ml.pkl"
    TEST_DATA = "testing data/reliance_data_5min_full_year_testing.csv"
    
    print("\n" + "="*70)
    print("🎯 FINAL OPTIMIZED EMA TRAP BACKTEST")
    print("="*70)
    
    backtester = FinalOptimizedBacktester(MODEL_PATH)
    metrics, trades_df = backtester.run_backtest(TEST_DATA)
    
    # Additional analysis
    if len(trades_df) > 0:
        print("\n" + "="*70)
        print("📊 DETAILED ANALYSIS")
        print("="*70)
        
        print(f"\n🎯 Best Trades (Top 5):")
        best = trades_df.nlargest(5, 'pnl_pct')[['entry_time', 'pnl_pct', 'exit_reason', 'entry_confidence']]
        for _, trade in best.iterrows():
            print(f"   {trade['entry_time']}: +{trade['pnl_pct']:.2f}% ({trade['exit_reason']}) Conf: {trade['entry_confidence']:.2f}")
        
        print(f"\n📉 Worst Trades (Top 5):")
        worst = trades_df.nsmallest(5, 'pnl_pct')[['entry_time', 'pnl_pct', 'exit_reason', 'entry_confidence']]
        for _, trade in worst.iterrows():
            print(f"   {trade['entry_time']}: {trade['pnl_pct']:.2f}% ({trade['exit_reason']}) Conf: {trade['entry_confidence']:.2f}")
    
    print("\n" + "="*70)
    print("✅ FINAL BACKTEST COMPLETE!")
    print("="*70)
