Enhance Feature Engineering: I will modify the create_technical_indicators function in train_reliance_model.py to add two new categories of
      features:
   2. Implement Robust Time-Series Cross-Validation: I will update the train_xgboost_model function. To prevent overfitting and get a more 
      realistic performance estimate, I will replace the standard cross-validation in GridSearchCV with TimeSeriesSplit. This ensures that the 
      model is always trained on past data and validated on future data, mimicking real-world trading.
   3. Retrain and Evaluate: I will execute the updated script. This will apply the new features, use the robust cross-validation to find the best 
      hyperparameters, retrain the final model on the full (SMOTE-resampled) training set, and then evaluate its performance on the unseen test 
      set.
  Shall I proceed?

