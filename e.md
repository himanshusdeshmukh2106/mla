[CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=9, reg_alpha=0, reg_lambda=1, subsample=0.9; total time=   2.5s
Best parameters found: {'subsample': 0.9, 'reg_lambda': 2, 'reg_alpha': 0.1, 'max_depth': 12, 'learning_rate': 0.05, 'colsample_bytree': 0.8}
Model training completed
Test Accuracy: 0.5942

Classification Report:
              precision    recall  f1-score   support

         0.0       0.63      0.80      0.70      2758
         1.0       0.47      0.28      0.35      1779

    accuracy                           0.59      4537
   macro avg       0.55      0.54      0.53      4537
weighted avg       0.57      0.59      0.57      4537

Confusion Matrix:
[[2196  562]
 [1279  500]]

Top 10 Most Important Features:
 1. EMA_26               0.0515
 2. Time_Since_Open      0.0464
 3. BB_Upper             0.0438
 4. SMA_50               0.0422
 5. SMA_20               0.0412
 6. BB_Middle            0.0407
 7. EMA_12               0.0402
 8. BB_Lower             0.0393
 9. Day_of_Week          0.0387
10. MACD_Signal          0.0375
Saving model and results...
Model saved to: models/reliance_5min_xgboost.pkl
Features saved to: models/reliance_5min_features.txt
Feature importance saved to: models/reliance_5min_feature_importance.csv

============================================================
TRAINING COMPLETED SUCCESSFULLY!
Your Reliance 5-min intraday model is ready!
==============================================