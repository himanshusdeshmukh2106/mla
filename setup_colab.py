#!/usr/bin/env python3
"""
Google Colab Setup Script for EMA Trap Strategy Training
Run this first in Colab to set up the environment
"""

import subprocess
import sys
import os

def setup_colab_environment():
    """Complete setup for Google Colab environment"""
    
    print("🚀 Setting up EMA Trap Strategy Training Environment for Google Colab")
    print("=" * 70)
    
    # 1. Install requirements
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False
    
    # 2. Check data file
    print("\n📁 Checking for data file...")
    data_paths = [
        "/content/reliance_data_5min_full_year.csv",
        "/content/data/reliance_data_5min_full_year.csv",
        "data/reliance_data_5min_full_year.csv"
    ]
    
    data_found = False
    for path in data_paths:
        if os.path.exists(path):
            print(f"✅ Data file found at: {path}")
            data_found = True
            break
    
    if not data_found:
        print("⚠️  Data file not found. Please upload 'reliance_data_5min_full_year.csv' to:")
        print("   - /content/ (Colab root directory)")
        print("   - Or create a 'data' folder and upload there")
        print("\n📋 To upload in Colab:")
        print("   1. Click the folder icon on the left sidebar")
        print("   2. Click 'Upload to session storage'")
        print("   3. Select your CSV file")
    
    # 3. Test imports
    print("\n🧪 Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        import xgboost as xgb
        import pandas_ta as ta
        from sklearn.model_selection import GridSearchCV
        print("✅ All imports successful!")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # 4. Final instructions
    print("\n" + "=" * 70)
    print("✅ Setup Complete! You can now run the training script:")
    print("   !python train_ema_trap_model.py")
    print("\n📊 The script will:")
    print("   - Load and process the Reliance 5-minute data")
    print("   - Create EMA trap features")
    print("   - Train an XGBoost model with hyperparameter optimization")
    print("   - Evaluate performance with detailed metrics")
    print("   - Save the trained model and results")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = setup_colab_environment()
    if not success:
        print("\n❌ Setup failed. Please check the errors above and try again.")
    else:
        print("\n🎉 Ready to train your EMA Trap model!")