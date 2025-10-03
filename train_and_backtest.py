"""
Train model locally and immediately backtest
"""

import subprocess
import os
import sys

print("="*70)
print("STEP 1: Training EMA Trap Model (Balanced Version)")
print("="*70)

# Train the model
result = subprocess.run([sys.executable, "train_ema_trap_balanced.py"], 
                       capture_output=False)

if result.returncode != 0:
    print("❌ Training failed!")
    sys.exit(1)

print("\n" + "="*70)
print("STEP 2: Running Backtest on Testing Data")
print("="*70)

# Check if model was created
model_path = "models/ema_trap_balanced.pkl"
if not os.path.exists(model_path):
    print(f"❌ Model not found at {model_path}")
    sys.exit(1)

# Run backtest
from backtest_ema_trap import EMATrapBacktester

TEST_DATA = "testing data/reliance_data_5min_full_year_testing.csv"

backtester = EMATrapBacktester(model_path)
metrics, trades_df = backtester.run_backtest(TEST_DATA)

print("\n" + "="*70)
print("✅ COMPLETE: Model trained and backtested!")
print("="*70)
