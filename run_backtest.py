"""
Quick script to run backtest on testing data
"""

from backtest_ema_trap import EMATrapBacktester

# Configuration
MODEL_PATH = "models/ema_trap_balanced_ml.pkl"  # Using the balanced EMA trap model
TEST_DATA = "testing data/reliance_data_5min_full_year_testing.csv"

print("🚀 EMA Trap Strategy Backtest")
print("="*70)
print(f"Model:     {MODEL_PATH}")
print(f"Test Data: {TEST_DATA}")
print("="*70)

# Run backtest
backtester = EMATrapBacktester(MODEL_PATH)
metrics, trades_df = backtester.run_backtest(TEST_DATA)

# Show first few trades
print("\n📋 Sample Trades (first 10):")
print(trades_df.head(10).to_string(index=False))

print("\n✅ Backtest Complete!")
print(f"📊 Check 'backtest_results.csv' for detailed trade-by-trade results")
