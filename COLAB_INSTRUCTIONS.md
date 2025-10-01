# EMA Trap Strategy Training - Google Colab Instructions

## Quick Start Guide

### 1. Clone the Repository
```bash
!git clone https://github.com/your-username/your-repo-name.git
%cd your-repo-name
```

### 2. Setup Environment
```python
!python setup_colab.py
```

### 3. Upload Your Data
- Upload `reliance_data_5min_full_year.csv` to `/content/` directory
- Or create a `data` folder and upload there

### 4. Run Training
```python
!python train_ema_trap_model.py
```

## What the Training Script Does

### 🔧 **Feature Engineering**
- Creates 21-period EMA and ADX indicators
- Detects EMA trap patterns (bearish and bullish)
- Generates time-based features for entry windows
- Analyzes candle patterns and volume

### 🎯 **Target Generation**
- Identifies profitable EMA trap entry signals
- Uses 0.4% profit/loss thresholds
- Looks ahead 2 candles (10 minutes) for validation

### 🤖 **Model Training**
- XGBoost with hyperparameter optimization
- Time-series cross-validation (5 folds)
- Advanced parameter grid search
- Early stopping to prevent overfitting

### 📊 **Evaluation**
- Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Confusion matrix analysis
- Feature importance ranking
- Visualization plots

### 💾 **Results**
- Trained model saved with full metadata
- Evaluation plots (confusion matrix, feature importance)
- Detailed performance report
- Model can be downloaded for deployment

## Expected Output

```
EMA Trap Strategy - Model Training Pipeline
============================================================

1. Loading Reliance 5-minute data...
   Loaded 50,000+ data points from 2023-01-02 to 2023-12-29

2. Creating EMA trap features...
   Features created: 48,000 rows, 66 columns

3. Generating trading targets...
   Entry Analysis:
     Bearish entries: 150, profitable: 85
     Bullish entries: 140, profitable: 78
     Total signals: 290
     Total targets: 163

4. Analyzing entry signals...
   Signals by hour: {9: 45, 10: 120, 11: 85, ...}

5. Selecting features for training...
   Using 20 features for training

6. Preparing training data...
   Training data: 47,500 samples, 20 features
   Target distribution: {0: 47337, 1: 163}

7. Training XGBoost model with advanced pipeline...
   Best parameters: {'max_depth': 6, 'learning_rate': 0.1, ...}
   Training accuracy: 0.9845

8. Evaluating model performance with advanced metrics...
   Accuracy: 0.9823
   Precision: 0.7234
   Recall: 0.6891
   F1-Score: 0.7058
   ROC-AUC: 0.8456

9. Saving model and results with advanced persistence...
   Model saved comprehensively to: models/ema_trap/ema_trap_xgboost_v1

============================================================
✅ EMA Trap Model Training Complete!
📊 Dataset: 47,500 samples
🎯 Entry signals: 290
🤖 Model saved: models/ema_trap/ema_trap_xgboost_v1
📈 Test accuracy: 0.9823
📈 Test F1-score: 0.7058
📈 Test ROC-AUC: 0.8456
📊 Confusion matrix plots saved
📊 Feature importance plots saved
📄 Detailed evaluation report saved
============================================================
```

## Files Generated

After training, you'll have these files:
- `ema_trap_confusion_matrix.png` - Confusion matrix visualization
- `ema_trap_feature_importance.png` - Feature importance plot
- `ema_trap_evaluation_report.txt` - Detailed performance report
- `ema_trap_analysis_results.json` - Complete analysis results
- `models/ema_trap/ema_trap_xgboost_v1/` - Saved model directory

## Troubleshooting

### Data File Not Found
```python
# Check if file exists
import os
print("Files in /content/:", os.listdir("/content/"))

# Upload file manually
from google.colab import files
uploaded = files.upload()
```

### Memory Issues
If you run out of memory, try:
```python
# Reduce data size for testing
# In train_ema_trap_model.py, add this after loading data:
# raw_data = raw_data.iloc[-10000:]  # Use last 10k rows only
```

### Package Installation Issues
```python
# Install packages individually
!pip install xgboost pandas-ta scikit-learn
```

## Next Steps

1. **Download the trained model** for deployment
2. **Analyze feature importance** to understand what drives predictions
3. **Backtest the strategy** using the trained model
4. **Deploy for live trading** (with proper risk management)

Happy trading! 🚀📈