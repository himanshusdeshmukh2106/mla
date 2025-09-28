#!/usr/bin/env python3
"""
Backtesting script for the Reliance 5-minute intraday XGBoost model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from backtesting import Backtest, Strategy

def create_technical_indicators(df):
    """
    Create technical indicators for 5-minute data.
    NOTE: This function is copied from train_reliance_model.py to ensure
    that the exact same features are generated for backtesting.
    """
    print("Creating technical indicators for backtest data...")
    
    # Price-based indicators
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['EMA_12'] = df['close'].ewm(span=12).mean()
    df['EMA_26'] = df['close'].ewm(span=26).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    df['BB_Middle'] = df['close'].rolling(window=bb_period).mean()
    bb_std_dev = df['close'].rolling(window=bb_period).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # Volume indicators
    df['Volume_SMA_20'] = df['volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_SMA_20']
    
    # Price momentum
    df['Price_Change'] = df['close'].pct_change()
    df['Price_Change_5'] = df['close'].pct_change(periods=5)
    df['Volatility_20'] = df['Price_Change'].rolling(window=20).std()
    
    # Trend indicators
    df['Price_SMA_20_Ratio'] = df['close'] / df['SMA_20']
    df['EMA_Cross'] = (df['EMA_12'] > df['EMA_26']).astype(int)

    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    # Williams %R
    df['Williams_R'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))

    # --- ADVANCED FEATURES ---
    df['ATR_Percentage'] = (df['ATR'] / df['close']) * 100
    df['Day_of_Week'] = df.index.dayofweek
    df['Time_Since_Open'] = (df.index.hour * 60 + df.index.minute) - 555
    df['RSI_Volume_Ratio'] = df['RSI'] * df['Volume_Ratio']
    df['RSI_x_Volatility'] = df['RSI'] * df['Volatility_20']
    
    return df

class XGBoostStrategy(Strategy):
    """
    Trading strategy that uses a pre-trained XGBoost model to make decisions.
    """
    def init(self):
        # Load the trained model
        print("Loading trained XGBoost model...")
        with open("models/reliance_5min_xgboost.pkl", "rb") as f:
            self.model = pickle.load(f)
        
        # Load the feature names used for training
        with open("models/reliance_5min_features.txt", "r") as f:
            self.feature_cols = [line.strip() for line in f.readlines()]
        
        # Pre-calculate features and predictions for the entire dataset
        print("Pre-calculating features and predictions...")
        self.features = self.I(lambda x: x, self.data.df[self.feature_cols])
        self.predictions = self.I(lambda x: self.model.predict(x), self.features, plot=False)
        
        # Pre-calculate ATR for stop-loss and take-profit
        self.atr = self.I(lambda x: x, self.data.ATR, plot=False)

    def next(self):
        # If we are already in a position, do nothing.
        if self.position:
            return

        # Check the prediction for the current candle
        if self.predictions[-1] == 1:
            # Define stop-loss and take-profit based on the Triple Barrier logic
            sl = self.data.Close[-1] - (self.atr[-1] * 1.0)
            tp = self.data.Close[-1] + (self.atr[-1] * 2.0)
            
            # Buy with a fixed size (e.g., 10 shares)
            self.buy(size=10, sl=sl, tp=tp)

def main():
    """Main backtesting pipeline"""
    print("=" * 60)
    print("RUNNING BACKTEST FOR RELIANCE XGBOOST STRATEGY")
    print("=" * 60)

    # Load test data
    data_file = Path("testing data/reliance_data_5min_full_year_testing.csv")
    df = pd.read_csv(data_file, index_col='datetime', parse_dates=True)
    
    # Create technical indicators
    df = create_technical_indicators(df)
    
    # Rename columns for backtesting.py compatibility
    df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    }, inplace=True)
    
    # Drop rows with NaNs created by indicators
    df.dropna(inplace=True)

    # Set up the backtest
    bt = Backtest(
        df,
        XGBoostStrategy,
        cash=100_000,  # Initial capital
        commission=.002,  # 0.2% commission for buy and sell
        exclusive_orders=True
    )

    # Run the backtest
    stats = bt.run()

    # Print the results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(stats)
    
    # Plot the equity curve and other stats
    print("\nGenerating backtest plot...")
    bt.plot()
    print("Backtest plot saved to the current directory.")

if __name__ == "__main__":
    main()
