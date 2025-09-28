[CV] END colsample_bytree=1.0, learning_rate=0.2, max_depth=9, n_estimators=300, reg_alpha=0.5, reg_lambda=2, subsample=1.0; total time=   1.6s
[CV] END colsample_bytree=1.0, learning_rate=0.2, max_depth=9, n_estimators=300, reg_alpha=0.5, reg_lambda=2, subsample=1.0; total time=   1.8s
Best parameters found: {'colsample_bytree': 0.8, 'learning_rate': 0.2, 'max_depth': 9, 'n_estimators': 300, 'reg_alpha': 0.1, 'reg_lambda': 1, 'subsample': 0.8}
Model training completed
Test Accuracy: 0.9435

Classification Report:
              precision    recall  f1-score   support

           0       0.95      1.00      0.97      4298
           1       0.09      0.00      0.01       248

    accuracy                           0.94      4546
   macro avg       0.52      0.50      0.49      4546
weighted avg       0.90      0.94      0.92      4546

Confusion Matrix:
[[4288   10]
 [ 247    1]]

Top 10 Most Important Features:
 1. EMA_Cross            0.2244
 2. Volume_Ratio         0.0675
 3. MACD                 0.0591
 4. Day_of_Week          0.0520
 5. Volume_SMA_20        0.0455
 6. ATR                  0.0387
 7. Time_of_Day          0.0346
 8. MACD_Signal          0.0344
 9. RSI_Volume_Ratio     0.0306
10. BB_Width             0.0301
Saving model and results...
Model saved to: models/reliance_5min_xgboost.pkl
Features saved to: models/reliance_5min_features.txt
Feature importance saved to: models/reliance_5min_feature_importance.csv

============================================================
TRAINING COMPLETED SUCCESSFULLY!
Your Reliance 5-min intraday model is ready!
============================================================