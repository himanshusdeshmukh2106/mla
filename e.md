Best parameters found: {'subsample': 0.9, 'reg_lambda': 1, 'reg_alpha': 0, 'max_depth': 3, 'learning_rate': 0.05, 'colsample_bytree': 1.0}
Model training completed
Test Accuracy: 0.9281

Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.97      0.96      4298
           1       0.23      0.14      0.18       248

    accuracy                           0.93      4546
   macro avg       0.59      0.56      0.57      4546
weighted avg       0.91      0.93      0.92      4546

Confusion Matrix:
[[4184  114]
 [ 213   35]]

Top 10 Most Important Features:
 1. ATR                  0.0849
 2. Time_Since_Open      0.0786
 3. Volume_SMA_20        0.0624
 4. Volume_Ratio         0.0556
 5. ATR_Percentage       0.0430
 6. EMA_12               0.0380
 7. SMA_50               0.0361
 8. BB_Upper             0.0359
 9. RSI_x_Volatility     0.0355
10. Price_SMA_20_Ratio   0.0352
Saving model and results...
Model saved to: models/reliance_5min_xgboost.pkl
Features saved to: models/reliance_5min_features.txt
Feature importance saved to: models/reliance_5min_feature_importance.csv

============================================================
TRAINING COMPLETED SUCCESSFULLY!
Your Reliance 5-min intraday model is ready!