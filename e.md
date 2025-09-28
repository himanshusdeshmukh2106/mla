[CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=9, reg_alpha=0, reg_lambda=1, subsample=0.9; total time=   0.7s
[CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=9, reg_alpha=0, reg_lambda=1, subsample=0.9; total time=   0.7s
Best parameters found: {'subsample': 0.9, 'reg_lambda': 3, 'reg_alpha': 0, 'max_depth': 3, 'learning_rate': 0.05, 'colsample_bytree': 0.7}
Model training completed
Test Accuracy: 0.9136

Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.95      0.95      4298
           1       0.21      0.21      0.21       248

    accuracy                           0.91      4546
   macro avg       0.58      0.58      0.58      4546
weighted avg       0.91      0.91      0.91      4546

Confusion Matrix:
[[4101  197]
 [ 196   52]]

Top 10 Most Important Features:
 1. Time_of_Day          0.0791
 2. ATR                  0.0687
 3. Volume_SMA_20        0.0611
 4. Volume_Ratio         0.0573
 5. Volatility_20        0.0397
 6. RSI_Volume_Ratio     0.0388
 7. EMA_26               0.0382
 8. Price_SMA_20_Ratio   0.0372
 9. BB_Upper             0.0363
10. SMA_20               0.0359
Saving model and results...
Model saved to: models/reliance_5min_xgboost.pkl
Features saved to: models/reliance_5min_features.txt
Feature importance saved to: models/reliance_5min_feature_importance.csv

============================================================
TRAINING COMPLETED SUCCESSFULLY!
Your Reliance 5-min intraday model is ready!
============================================================