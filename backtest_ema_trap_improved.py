"""
Improved EMA Trap Strategy Backtesting with ATR-based stops
Fixes identified issues from analysis
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

from src.logger import get_logger

logger = get_logger(__name__)


class ImprovedEMATrapBacktester:
    """Improved backtester with ATR stops and better filters"""
    
    def __init__(self, model_path):
        """Initialize backtester with trained model"""
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load model metadata
        import json
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_names = self.metadata['features']
        logger.info(f"Loaded {len(self.feature_names)} features from metadata")
        
        # IMPROVED TRADING PARAMETERS
        self.profit_target_pct = 0.005  # 0.5% profit target
        self.atr_multiplier = 1.5       # Stop loss = 1.5 x ATR
        self.max_holding_candles = 20   # Max 20 candles (100 min)
        
        # IMPROVED FILTERS
        self.confidence_threshold = 0.70  # Higher threshold (was 0.55)
        self.min_adx = 25                 # Only trade with ADX > 25
        self.skip_first_minutes = 15      # Skip 9:15-9:30
        self.use_confirmation = True      # Wait 1 candle for confirmation
        
        logger.info(f"Using confidence threshold: {self.confidence_threshold}")
        logger.info(f"Using ATR-based stop loss: {self.atr_multiplier}x ATR")
        logger.info(f"Minimum ADX: {self.min_adx}")
        
        # Results tracking
        self.trades = []
        
    def load_test_data(self, test_file):
        """Load and prepare test data"""
        logger.info(f"Loading test data from {test_file}")
        
        # Load raw data
        df = pd.read_csv(test_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        
        logger.info(f"Loaded {len(df)} candles from {df.index.min()} to {df.index.max()}")
        
        # Create features
        df = self.create_training_features(df)
        
        # Calculate ATR for stop loss
        import pandas_ta as ta
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['ATR_pct'] = (df['ATR'] / df['close']) * 100
        
        # Remove rows with NaN
        df = df.dropna()
        df = df.reset_index()
        
        logger.info(f"After preprocessing: {len(df)} candles ready for backtesting")
        
        return df
    
    def create_training_features(self, df):
        """Create features exactly as done in training script"""
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
        
        # Distance from EMA over time
        df['Distance_EMA_Change'] = df['Distance_From_EMA21_Pct'].diff()
        df['Distance_EMA_Trend'] = df['Distance_From_EMA21_Pct'].rolling(3).mean()
        
        # ADX
        adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['ADX'] = adx_result['ADX_14']
        df['ADX_Change'] = df['ADX'].diff()
        
        # ADX ranges
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
        
        df['Candle_Efficiency'] = np.where(
            df['Candle_Range_Pct'] > 0,
            df['Candle_Body_Pct'] / df['Candle_Range_Pct'],
            0
        )
        
        # Candle sizes
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
        
        # Combined signals
        df['EMA_ADX_Signal'] = ((df['ADX'] >= 20) & (df['ADX'] <= 36) & 
                                (abs(df['Distance_From_EMA21_Pct']) < 1.0)).astype(int)
        
        df['Volume_Candle_Signal'] = ((df['Volume_Ratio'] > 0.8) & 
                                      (df['Candle_Body_Pct'] > 0.15)).astype(int)
        
        df['Time_EMA_Signal'] = ((df['Hour'].between(9, 14)) & 
                                 (abs(df['Distance_From_EMA21_Pct']) < 2.0)).astype(int)
        
        logger.info(f"Created {len([c for c in df.columns if c not in ['open','high','low','close','volume']])} features")
        
        return df
    
    def generate_signals(self, df):
        """Generate trading signals with improved filters"""
        # Get predictions
        X = df[self.feature_names].values
        probabilities = self.model.predict_proba(X)[:, 1]
        
        df['signal_prob'] = probabilities
        df['signal'] = 0
        
        # Apply filters
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # Filter 1: Confidence threshold
            if row['signal_prob'] < self.confidence_threshold:
                continue
            
            # Filter 2: ADX filter (trend strength)
            if row['ADX'] < self.min_adx:
                continue
            
            # Filter 3: Skip first 15 minutes
            if row['datetime'].time() < time(9, 30):
                continue
            
            # Filter 4: Only during market hours
            if not (time(9, 30) <= row['datetime'].time() <= time(15, 15)):
                continue
            
            # Filter 5: Don't enter in last hour
            if row['datetime'].time() >= time(14, 30):
                continue
            
            df.at[idx, 'signal'] = 1
        
        signals_count = df['signal'].sum()
        logger.info(f"Generated {signals_count} signals after filtering (from {(probabilities >= self.confidence_threshold).sum()} high-confidence predictions)")
        
        return df
    
    def simulate_trades(self, df):
        """Simulate trades with ATR-based stops and confirmation"""
        logger.info("Starting trade simulation with improved logic...")
        
        in_trade = False
        entry_idx = None
        entry_price = None
        entry_time = None
        stop_loss_price = None
        profit_target_price = None
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # Skip if in trade
            if in_trade:
                # Check exit conditions
                current_price = row['close']
                candles_held = idx - entry_idx
                
                # Exit conditions
                exit_reason = None
                exit_price = current_price
                
                # 1. Profit target hit
                if row['high'] >= profit_target_price:
                    exit_price = profit_target_price
                    exit_reason = 'profit_target'
                
                # 2. Stop loss hit (ATR-based)
                elif row['low'] <= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_reason = 'stop_loss'
                
                # 3. Max holding period
                elif candles_held >= self.max_holding_candles:
                    exit_reason = 'max_holding'
                
                # 4. End of day (after 3:15 PM)
                elif row['datetime'].time() >= time(15, 15):
                    exit_reason = 'eod'
                
                # Exit trade if condition met
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
                        'candles_held': candles_held,
                        'win': 1 if pnl_pct > 0 else 0,
                        'stop_loss_pct': (stop_loss_price - entry_price) / entry_price * 100,
                        'entry_adx': df.iloc[entry_idx]['ADX'],
                        'entry_confidence': df.iloc[entry_idx]['signal_prob']
                    }
                    
                    self.trades.append(trade)
                    in_trade = False
                    entry_idx = None
                    
                continue
            
            # Check for entry signal
            if row['signal'] == 1:
                # Confirmation candle logic
                if self.use_confirmation and idx > 0:
                    # Check if previous candle also had signal
                    prev_signal = df.iloc[idx-1]['signal']
                    if prev_signal == 0:
                        continue  # Wait for confirmation
                
                # Enter trade
                in_trade = True
                entry_idx = idx
                entry_price = row['close']
                entry_time = row['datetime']
                
                # Calculate ATR-based stop loss
                atr_value = row['ATR']
                stop_loss_price = entry_price - (atr_value * self.atr_multiplier)
                
                # Calculate profit target
                profit_target_price = entry_price * (1 + self.profit_target_pct)
                
                logger.info(f"ENTRY: {entry_time} @ {entry_price:.2f} | SL: {stop_loss_price:.2f} ({((stop_loss_price-entry_price)/entry_price*100):.2f}%) | Target: {profit_target_price:.2f} | ADX: {row['ADX']:.1f} | Conf: {row['signal_prob']:.3f}")
        
        # Close any open trade at end
        if in_trade:
            row = df.iloc[-1]
            pnl_pct = (row['close'] - entry_price) / entry_price
            
            trade = {
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': row['datetime'],
                'exit_price': row['close'],
                'exit_reason': 'end_of_data',
                'pnl_pct': pnl_pct * 100,
                'pnl_points': row['close'] - entry_price,
                'candles_held': len(df) - entry_idx,
                'win': 1 if pnl_pct > 0 else 0,
                'stop_loss_pct': (stop_loss_price - entry_price) / entry_price * 100,
                'entry_adx': df.iloc[entry_idx]['ADX'],
                'entry_confidence': df.iloc[entry_idx]['signal_prob']
            }
            self.trades.append(trade)
        
        logger.info(f"Simulation complete: {len(self.trades)} trades executed")
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            logger.warning("No trades to analyze")
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = trades_df['win'].sum()
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100
        
        # P&L metrics
        total_pnl_pct = trades_df['pnl_pct'].sum()
        avg_pnl_pct = trades_df['pnl_pct'].mean()
        
        winning_pnl = trades_df[trades_df['win'] == 1]['pnl_pct'].sum()
        losing_pnl = trades_df[trades_df['win'] == 0]['pnl_pct'].sum()
        
        avg_win = trades_df[trades_df['win'] == 1]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['win'] == 0]['pnl_pct'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        profit_factor = abs(winning_pnl / losing_pnl) if losing_pnl != 0 else float('inf')
        expectancy = avg_pnl_pct
        
        # Holding period
        avg_holding = trades_df['candles_held'].mean()
        
        # Exit reason breakdown
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        
        # ATR stop loss stats
        avg_sl_pct = trades_df['stop_loss_pct'].mean()
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl_pct': total_pnl_pct,
            'avg_pnl_pct': avg_pnl_pct,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_holding_candles': avg_holding,
            'exit_reasons': exit_reasons,
            'avg_stop_loss_pct': avg_sl_pct
        }
        
        return metrics
    
    def print_results(self, metrics):
        """Print formatted backtest results"""
        print("\n" + "="*70)
        print("IMPROVED EMA TRAP STRATEGY - BACKTEST RESULTS")
        print("="*70)
        
        print(f"\n📊 TRADE STATISTICS:")
        print(f"   Total Trades:        {metrics['total_trades']}")
        print(f"   Winning Trades:      {metrics['winning_trades']}")
        print(f"   Losing Trades:       {metrics['losing_trades']}")
        print(f"   Win Rate:            {metrics['win_rate']:.2f}%")
        
        print(f"\n💰 P&L METRICS:")
        print(f"   Total P&L:           {metrics['total_pnl_pct']:.2f}%")
        print(f"   Average P&L:         {metrics['avg_pnl_pct']:.3f}%")
        print(f"   Average Win:         {metrics['avg_win_pct']:.3f}%")
        print(f"   Average Loss:        {metrics['avg_loss_pct']:.3f}%")
        print(f"   Profit Factor:       {metrics['profit_factor']:.2f}")
        print(f"   Expectancy:          {metrics['expectancy']:.3f}%")
        
        print(f"\n⏱️  HOLDING PERIOD:")
        print(f"   Avg Candles Held:    {metrics['avg_holding_candles']:.1f} ({metrics['avg_holding_candles']*5:.0f} minutes)")
        
        print(f"\n🛡️  RISK MANAGEMENT:")
        print(f"   Avg ATR Stop Loss:   {metrics['avg_stop_loss_pct']:.3f}%")
        
        print(f"\n🚪 EXIT REASONS:")
        for reason, count in metrics['exit_reasons'].items():
            pct = (count / metrics['total_trades']) * 100
            print(f"   {reason:20s} {count:3d} ({pct:5.1f}%)")
        
        print("\n" + "="*70)
    
    def save_results(self, output_file='backtest_results_improved.csv'):
        """Save detailed trade results to CSV"""
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(output_file, index=False)
            logger.info(f"Saved detailed results to {output_file}")
            print(f"\n💾 Detailed results saved to: {output_file}")
    
    def run_backtest(self, test_file, save_results=True):
        """Run complete backtest pipeline"""
        # Load and prepare data
        df = self.load_test_data(test_file)
        
        # Generate signals
        df = self.generate_signals(df)
        
        # Simulate trades
        self.simulate_trades(df)
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Print results
        self.print_results(metrics)
        
        # Save results
        if save_results:
            self.save_results()
        
        return metrics, pd.DataFrame(self.trades)


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "models/ema_trap_balanced_ml.pkl"
    TEST_DATA = "testing data/reliance_data_5min_full_year_testing.csv"
    
    print("🚀 Starting IMPROVED EMA Trap Strategy Backtest...")
    print(f"📁 Model: {MODEL_PATH}")
    print(f"📁 Test Data: {TEST_DATA}")
    
    # Run backtest
    backtester = ImprovedEMATrapBacktester(MODEL_PATH)
    metrics, trades_df = backtester.run_backtest(TEST_DATA)
    
    print("\n✅ Backtest complete!")
