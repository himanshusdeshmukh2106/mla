[CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=9, reg_alpha=0, reg_lambda=1, subsample=0.9; total time=   0.0s
[CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=9, reg_alpha=0, reg_lambda=1, subsample=0.9; total time=   0.0s
[CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=9, reg_alpha=0, reg_lambda=1, subsample=0.9; total time=   0.0s
[CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=9, reg_alpha=0, reg_lambda=1, subsample=0.9; total time=   0.0s
Training failed: 
All the 500 fits failed.
It is very likely that your model is misconfigured.
You can try to debug the error by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
500 fits failed with the following error:
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/sklearn/model_selection/_validation.py", line 866, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "/usr/local/lib/python3.12/dist-packages/xgboost/core.py", line 729, in inner_f
    return func(**kwargs)
           ^^^^^^^^^^^^^^
TypeError: XGBClassifier.fit() got an unexpected keyword argument 'early_stopping_rounds'

Traceback (most recent call last):
  File "/content/mla/train_reliance_model.py", line 347, in main
    model, feature_importance = train_xgboost_model(X_train, y_train, X_test, y_test, feature_cols)
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/content/mla/train_reliance_model.py", line 247, in train_xgboost_model
    random_search.fit(
  File "/usr/local/lib/python3.12/dist-packages/sklearn/base.py", line 1389, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/sklearn/model_selection/_search.py", line 1024, in fit
    self._run_search(evaluate_candidates)
  File "/usr/local/lib/python3.12/dist-packages/sklearn/model_selection/_search.py", line 1951, in _run_search
    evaluate_candidates(
  File "/usr/local/lib/python3.12/dist-packages/sklearn/model_selection/_search.py", line 1001, in evaluate_candidates
    _warn_or_raise_about_fit_failures(out, self.error_score)
  File "/usr/local/lib/python3.12/dist-packages/sklearn/model_selection/_validation.py", line 517, in _warn_or_raise_about_fit_failures
    raise ValueError(all_fits_failed_message)
ValueError: 
All the 500 fits failed.
It is very likely that your model is misconfigured.
You can try to debug the error by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
500 fits failed with the following error:
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/sklearn/model_selection/_validation.py", line 866, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "/usr/local/lib/python3.12/dist-packages/xgboost/core.py", line 729, in inner_f
    return func(**kwargs)
           ^^^^^^^^^^^^^^
TypeError: XGBClassifier.fit() got an unexpected keyword argument 'early_stopping_rounds'
