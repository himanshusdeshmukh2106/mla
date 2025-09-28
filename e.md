Best parameters found: {'subsample': 1.0, 'reg_lambda': 1.5, 'reg_alpha': 0.5, 'max_depth': 12, 'learning_rate': 0.05, 'colsample_bytree': 0.8}
Model training completed
Test Accuracy: 0.7469

Classification Report:
              precision    recall  f1-score   support

         0.0       0.79      0.92      0.85      3500
         1.0       0.37      0.16      0.22      1039

    accuracy                           0.75      4539
   macro avg       0.58      0.54      0.53      4539
weighted avg       0.69      0.75      0.71      4539

Confusion Matrix:
[[3227  273]
 [ 876  163]]

Top 10 Most Important Features:
 1. Time_Since_Open      0.0591
 2. EMA_26               0.0525
 3. SMA_50               0.0464
 4. BB_Upper             0.0457
 5. EMA_12               0.0447
 6. BB_Middle            0.0423
 7. BB_Lower             0.0418
 8. MACD_Signal          0.0370
 9. Volume_SMA_20        0.0366
10. Day_of_Week          0.0363
Saving model and results...
Model saved to: models/reliance_5min_xgboost.pkl
Features saved to: models/reliance_5min_features.txt
Feature importance saved to: models/reliance_5min_feature_importance.csv

============================================================