# 🚀 Optimized EMA Crossover Training Guide

## What's New?

This optimized training pipeline implements **4 major improvements** over the original approach:

### 1. ✅ **Optuna Hyperparameter Optimization** (Replaces GridSearch)
- **10x faster** than GridSearchCV
- Uses Bayesian optimization instead of exhaustive search
- Finds better parameters with fewer iterations
- **Before**: 2,916 combinations × 3 CV folds = 8,748 model fits (~2 hours)
- **After**: 100 trials with smart sampling = 100 model fits (~15 minutes)

### 2. ✅ **Ensemble Models** (XGBoost + LightGBM)
- Combines predictions from multiple models
- **XGBoost**: Great for complex patterns
- **LightGBM**: Faster training, handles large datasets well
- **Ensemble**: Voting classifier for best of both worlds
- Typically **2-5% better accuracy** than single model

### 3. ✅ **Walk-Forward Analysis**
- More realistic performance estimation
- Simulates real trading by retraining on expanding window
- Detects overfitting better than single train/test split
- **5 splits** = 5 different time periods tested

### 4. ✅ **Feature Selection** (78 → ~30 features)
- Reduces from 78 to ~30 most important features
- **Benefits**:
  - Faster training (2-3x speedup)
  - Less overfitting
  - Better generalization
  - Easier to interpret
- Uses XGBoost feature importance

---

## 📊 Your Current Features (78 Total)

### By Category:

| Category | Count | Examples |
|----------|-------|----------|
| **Core EMA** | 11 | EMA_8, EMA_30, EMA_Spread, EMA_8_Slope_3 |
| **Crossovers** | 4 | EMA_Cross_Above, Cross_Above_Last_5 |
| **Price-EMA Distance** | 9 | Price_Distance_EMA8, Price_Position_Flag |
| **Price Action** | 10 | Candle_Body_Size, Green_Candle, Strong_Candle |
| **Volume** | 6 | Volume_Spike, Volume_ROC, Volume_MA20 |
| **Volatility** | 3 | Rolling_Volatility, Volatility_Ratio |
| **Swing Levels** | 18 | Swing_High_5, Distance_to_Swing_Low_10 |
| **Lag Features** | 12 | EMA_Spread_Lag_1, Returns_Lag_3 |
| **Time** | 6 | Hour, Best_Hours, Morning_Session |
| **Composite Signals** | 4 | Bullish_Cross_Signal, Bearish_Retest_Signal |

### Features That Can Be Reduced:

**High Redundancy:**
- Multiple swing levels (5, 10, 20 periods) → Keep only 10
- Multiple lag features (1, 2, 3, 5) → Keep only 1, 3
- Multiple slope calculations → Keep only 5-period

**Low Importance (Usually):**
- Pattern recognition (Doji, Hammer) - rarely useful
- Some composite signals (already captured by base features)
- Excessive time granularity (Minute, Time_Slot)

**Expected Reduction:**
- **From**: 78 features
- **To**: ~30 features (60% reduction)
- **Keep**: Top features that contribute 90% of importance

---

## 🚀 Quick Start

### Step 1: Install New Dependencies

```bash
pip install lightgbm catboost optuna
```

Or update all:
```bash
pip install -r requirements.txt
```

### Step 2: Analyze Your Features (Optional but Recommended)

```bash
python analyze_ema_crossover_features.py
```

This will show you:
- Feature importance rankings
- Which features to keep/remove
- Correlation analysis
- Category-wise importance

**Output:**
- `ema_crossover_feature_importance.csv` - Detailed rankings

### Step 3: Run Optimized Training

```bash
python train_ema_crossover_optimized.py
```

**What it does:**
1. Loads your data with targets
2. Selects best ~30 features automatically
3. Runs walk-forward analysis (5 splits)
4. Optimizes XGBoost with Optuna (100 trials)
5. Optimizes LightGBM with Optuna (100 trials)
6. Trains final ensemble model
7. Saves all models and metadata

**Expected Time:**
- Feature selection: ~2 minutes
- Walk-forward analysis: ~15 minutes (5 splits × 30 trials each)
- Final training: ~10 minutes (100 trials each)
- **Total**: ~30 minutes (vs 2+ hours with GridSearch)

### Step 4: Use the Models

```python
import joblib

# Load ensemble model (best performance)
ensemble = joblib.load('models/ema_crossover_ensemble.pkl')

# Or load individual models
xgb_model = joblib.load('models/ema_crossover_xgboost.pkl')
lgb_model = joblib.load('models/ema_crossover_lightgbm.pkl')

# Load metadata to see which features were selected
import json
with open('models/ema_crossover_ensemble_metadata.json', 'r') as f:
    metadata = json.load(f)
    
selected_features = metadata['feature_names']
print(f"Model uses {len(selected_features)} features")

# Make predictions
# X should have only the selected features in the same order
probabilities = ensemble.predict_proba(X)[:, 1]
predictions = ensemble.predict(X)
```

---

## 📈 Expected Performance Improvements

### Training Speed:
- **Before**: ~2 hours (GridSearch with 2,916 combinations)
- **After**: ~30 minutes (Optuna with 100 trials)
- **Speedup**: 4x faster

### Model Performance:
- **Single XGBoost**: F1 ~0.65-0.70
- **Ensemble**: F1 ~0.68-0.73 (2-5% improvement)
- **Walk-forward validated**: More realistic estimates

### Feature Efficiency:
- **Before**: 78 features
- **After**: ~30 features
- **Training speed**: 2-3x faster
- **Overfitting**: Reduced

---

## 🎯 Understanding the Output

### Walk-Forward Results (`walk_forward_results.csv`)

```
split | train_size | test_size | xgb_f1 | lgb_f1 | ensemble_f1
------|------------|-----------|--------|--------|-------------
1     | 10000      | 2000      | 0.68   | 0.67   | 0.70
2     | 12000      | 2000      | 0.69   | 0.68   | 0.71
...
```

**What to look for:**
- Consistent performance across splits (not declining)
- Ensemble should be equal or better than individual models
- If performance drops significantly in later splits → model drift

### Feature Importance (`ema_crossover_selected_features.txt`)

Shows the 30 features selected by the algorithm. Typically includes:
- Core EMA features (EMA_8, EMA_30, EMA_Spread)
- Key crossover signals
- Price-EMA distances
- Volume confirmation
- Best time features

### Metadata (`ema_crossover_ensemble_metadata.json`)

```json
{
  "training_date": "2025-01-21T10:30:00",
  "original_features": 78,
  "selected_features": 30,
  "feature_names": [...],
  "xgb_params": {...},
  "lgb_params": {...},
  "positive_rate": 0.15
}
```

---

## 🔧 Customization Options

### Change Number of Features

```python
# In train_ema_crossover_optimized.py, line ~250
trainer.select_features(method='importance', n_features=30)  # Change 30 to your preference
```

**Recommendations:**
- **20 features**: Fastest, may miss some patterns
- **30 features**: Good balance (recommended)
- **40 features**: More comprehensive, slower training

### Change Optimization Trials

```python
# Walk-forward analysis (line ~280)
wf_results = trainer.walk_forward_analysis(n_splits=5, optimize_trials=30)  # Increase for better optimization

# Final training (line ~290)
metrics = trainer.train_final_ensemble(optimize_trials=100)  # Increase for better optimization
```

**Recommendations:**
- **Quick test**: 20 trials (~5 min)
- **Balanced**: 50 trials (~10 min)
- **Thorough**: 100 trials (~20 min)
- **Exhaustive**: 200+ trials (~40+ min)

### Change Walk-Forward Splits

```python
# Line ~280
wf_results = trainer.walk_forward_analysis(n_splits=5, optimize_trials=30)  # Change n_splits
```

**Recommendations:**
- **3 splits**: Faster, less robust
- **5 splits**: Good balance (recommended)
- **10 splits**: More robust, much slower

### Use Different Feature Selection Method

```python
# Line ~250
trainer.select_features(method='recursive', n_features=30)  # Use RFECV instead
```

**Methods:**
- **'importance'**: Fast, based on XGBoost feature importance (recommended)
- **'recursive'**: Slower, more thorough, uses RFECV

---

## 📊 Comparing with Original Training

| Aspect | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **Hyperparameter Search** | GridSearch | Optuna | 10x faster |
| **Models** | XGBoost only | XGBoost + LightGBM + Ensemble | 2-5% better |
| **Validation** | Single split | Walk-forward (5 splits) | More realistic |
| **Features** | 78 | ~30 (selected) | 2-3x faster training |
| **Training Time** | ~2 hours | ~30 minutes | 4x faster |
| **Overfitting Risk** | Higher | Lower | Better generalization |

---

## 🐛 Troubleshooting

### Error: "ema_crossover_with_targets.csv not found"
**Solution**: Run `create_ema_crossover_targets.py` first to generate targets

### Error: "No module named 'lightgbm'"
**Solution**: `pip install lightgbm catboost optuna`

### Training is too slow
**Solutions**:
- Reduce `optimize_trials` (e.g., 50 instead of 100)
- Reduce `n_splits` in walk-forward (e.g., 3 instead of 5)
- Use fewer features (e.g., 20 instead of 30)

### Walk-forward performance is declining
**Possible causes**:
- Market regime change
- Model overfitting to early data
- Need more frequent retraining

**Solutions**:
- Use shorter training windows
- Add regime detection features
- Implement online learning

### Ensemble not better than individual models
**Possible causes**:
- Models are too similar
- Not enough diversity

**Solutions**:
- Add CatBoost to ensemble
- Use different feature subsets for each model
- Try different model architectures

---

## 📚 Next Steps

1. ✅ **Run the analysis**: `python analyze_ema_crossover_features.py`
2. ✅ **Train optimized models**: `python train_ema_crossover_optimized.py`
3. ✅ **Compare results**: Check walk-forward performance
4. ✅ **Backtest**: Use ensemble model in your backtesting script
5. ✅ **Monitor**: Track performance on new data

---

## 💡 Advanced Tips

### Add CatBoost to Ensemble

CatBoost is another excellent gradient boosting library. To add it:

```python
# In train_ema_crossover_optimized.py
from catboost import CatBoostClassifier

# Add optimization method
def optimize_catboost(self, X_train, y_train, X_val, y_val, n_trials=50):
    def objective(trial):
        params = {
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'iterations': trial.suggest_int('iterations', 100, 500),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_state': 42,
            'verbose': False
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# Add to ensemble
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model)  # Add CatBoost
    ],
    voting='soft'
)
```

### Implement Online Learning

For adapting to changing markets:

```python
# Retrain model incrementally with new data
model.fit(X_new, y_new, xgb_model=model)  # XGBoost supports incremental training
```

### Add Regime Detection

Detect market regimes and use different models:

```python
from hmmlearn import hmm

# Train HMM on returns
regime_model = hmm.GaussianHMM(n_components=3)  # 3 regimes
regime_model.fit(returns.reshape(-1, 1))

# Use different models for different regimes
if regime == 0:  # Trending
    model = trending_model
elif regime == 1:  # Ranging
    model = ranging_model
else:  # Volatile
    model = volatile_model
```

---

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the error messages carefully
3. Ensure all dependencies are installed
4. Verify your data has the required target columns

---

**Happy Trading! 🚀📈**
