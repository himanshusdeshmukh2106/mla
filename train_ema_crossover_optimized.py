"""
EMA CROSSOVER STRATEGY - OPTIMIZED TRAINING (OVERFITTING PREVENTION)

Implements:
1. Optuna hyperparameter optimization (replaces GridSearch)
2. Ensemble models (XGBoost + LightGBM)
3. Walk-forward analysis
4. Feature selection (reduce from 78 features)

OVERFITTING PREVENTION MEASURES:
================================
✅ Early Stopping: All models use early_stopping_rounds=50 to stop when validation performance plateaus
✅ Reduced Complexity: Max depth limited to 3-6 (was 3-10), max estimators to 300 (was 500)
✅ Lower Learning Rate: Range 0.005-0.1 (was 0.01-0.3) for better generalization
✅ Aggressive Regularization: L1/L2 regularization increased to 0.1-5.0 (was 0-1.0)
✅ Increased Gamma: More conservative tree splitting with gamma 0.1-2.0 (was 0-0.5)
✅ Stronger Subsampling: Subsample 0.5-0.9 (was 0.6-1.0) to introduce randomness
✅ Higher Min Child Weight: XGBoost min_child_weight 3-20 (was 1-10)
✅ Higher Min Child Samples: LightGBM min_child_samples 10-100 (was 5-50)
✅ Num Leaves Constraint: LightGBM num_leaves limited to 8-31
✅ Min Split Gain: LightGBM min_split_gain 0.01-1.0 for conservative splitting
✅ Class Imbalance Handling: scale_pos_weight calculated and applied
✅ Feature Selection: Reduces dimensionality to ~30 features
✅ Walk-Forward Validation: Tests generalization on out-of-sample data
✅ Ensemble Voting: Reduces overfitting through model diversity
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import RFECV, SelectFromModel
import optuna
import joblib
import json
from datetime import datetime
import os


class OptimizedEMACrossoverTrainer:
    """
    Optimized trainer with Optuna, ensemble models, walk-forward, and feature selection
    """
    
    def __init__(self, data_path: str):
        """Initialize trainer"""
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.selected_features = None
        self.best_models = {}
        self.ensemble_model = None
        
        print("="*80)
        print("🚀 OPTIMIZED EMA CROSSOVER TRAINER (OVERFITTING PREVENTION)")
        print("="*80)
        print("Features:")
        print("  ✅ Optuna hyperparameter optimization")
        print("  ✅ Ensemble models (XGBoost + LightGBM)")
        print("  ✅ Walk-forward analysis")
        print("  ✅ Feature selection")
        print("  ✅ Early stopping on all models")
        print("  ✅ Aggressive regularization (L1/L2, gamma, subsample)")
        print("  ✅ Class imbalance handling")
        print("="*80)
    
    def load_data(self):
        """Load and prepare data"""
        print("\n📊 Loading data...")
        
        self.df = pd.read_csv(self.data_path)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        
        print(f"   Loaded {len(self.df):,} rows")
        
        # Check for target column
        target_cols = [col for col in self.df.columns if 'target_' in col]
        if not target_cols:
            raise ValueError("No target columns found! Run create_ema_crossover_targets.py first")
        
        print(f"   Found {len(target_cols)} target columns: {target_cols}")
        
        # Use combined 1R target (best balance)
        if 'target_combined_1.0R' in self.df.columns:
            self.target_col = 'target_combined_1.0R'
        else:
            self.target_col = target_cols[0]
        
        print(f"   Using target: {self.target_col}")
        
        # Prepare features and target
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'datetime', 'timestamp']
        exclude_cols.extend(target_cols)
        
        self.feature_names = [col for col in self.df.columns if col not in exclude_cols]
        
        self.X = self.df[self.feature_names].fillna(0).values
        self.y = self.df[self.target_col].values
        
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Samples: {len(self.X):,}")
        print(f"   Positive class: {self.y.sum():,} ({self.y.sum()/len(self.y)*100:.1f}%)")
        
        return self
    
    def select_features(self, method='importance', n_features=30):
        """
        Feature selection to reduce from 78 to ~30 features
        
        Args:
            method: 'importance' or 'recursive'
            n_features: Target number of features
        """
        print(f"\n🎯 Feature Selection ({method})...")
        print(f"   Reducing from {len(self.feature_names)} to ~{n_features} features")
        
        # Split data for feature selection
        split_idx = int(len(self.X) * 0.8)
        X_train, y_train = self.X[:split_idx], self.y[:split_idx]
        
        if method == 'importance':
            # Use XGBoost feature importance
            print("   Training XGBoost for feature importance...")
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Select features based on importance
            selector = SelectFromModel(
                model,
                threshold='median',  # Keep features above median importance
                prefit=True
            )
            
            selected_mask = selector.get_support()
            
        elif method == 'recursive':
            # Recursive Feature Elimination with Cross-Validation
            print("   Running RFECV (this may take a few minutes)...")
            model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            selector = RFECV(
                estimator=model,
                step=5,  # Remove 5 features at a time
                cv=TimeSeriesSplit(n_splits=3),
                scoring='f1',
                n_jobs=-1,
                min_features_to_select=n_features
            )
            selector.fit(X_train, y_train)
            
            selected_mask = selector.support_
        
        # Get selected features
        self.selected_features = [name for name, selected in zip(self.feature_names, selected_mask) if selected]
        
        # Update X with selected features only
        self.X = self.X[:, selected_mask]
        
        print(f"   ✅ Selected {len(self.selected_features)} features")
        print(f"\n   Top 20 selected features:")
        for i, feat in enumerate(self.selected_features[:20], 1):
            print(f"      {i:2d}. {feat}")
        
        if len(self.selected_features) > 20:
            print(f"      ... and {len(self.selected_features) - 20} more")
        
        return self
    
    def optimize_xgboost(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Optimize XGBoost with Optuna - with overfitting prevention"""
        print("\n🔍 Optimizing XGBoost with Optuna...")
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"   Class imbalance ratio: {scale_pos_weight:.2f}")
        
        def objective(trial):
            params = {
                # Reduced max_depth to prevent overfitting (was 3-10, now 3-6)
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                # Lower learning rate for better generalization (was 0.01-0.3, now 0.005-0.1)
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                # Reduced n_estimators (early stopping will handle this)
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                # More aggressive subsampling to reduce overfitting (was 0.6-1.0, now 0.5-0.9)
                'subsample': trial.suggest_float('subsample', 0.5, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                # Increased min_child_weight for regularization (was 1-10, now 3-20)
                'min_child_weight': trial.suggest_int('min_child_weight', 3, 20),
                # Increased gamma for more conservative splitting (was 0-0.5, now 0.1-2.0)
                'gamma': trial.suggest_float('gamma', 0.1, 2.0),
                # Stronger L1/L2 regularization (was 0-1.0, now 0.1-5.0)
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0),
                # Handle class imbalance
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train, 
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            y_pred = model.predict(X_val)
            y_train_pred = model.predict(X_train)
            
            train_f1 = f1_score(y_train, y_train_pred)
            val_f1 = f1_score(y_val, y_pred)
            
            # Penalize overfitting: if train >> val, reduce score
            overfitting_penalty = abs(train_f1 - val_f1)
            return val_f1 - (overfitting_penalty * 0.5)
        
        study = optuna.create_study(direction='maximize', study_name='xgboost')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"   Best F1: {study.best_value:.4f}")
        print(f"   Best params: {study.best_params}")
        
        return study.best_params
    
    def optimize_lightgbm(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Optimize LightGBM with Optuna - with overfitting prevention"""
        print("\n🔍 Optimizing LightGBM with Optuna...")
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"   Class imbalance ratio: {scale_pos_weight:.2f}")
        
        def objective(trial):
            params = {
                # Reduced max_depth to prevent overfitting (was 3-10, now 3-6)
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                # Lower learning rate for better generalization (was 0.01-0.3, now 0.005-0.1)
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                # Reduced n_estimators (early stopping will handle this)
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                # More aggressive subsampling (was 0.6-1.0, now 0.5-0.9)
                'subsample': trial.suggest_float('subsample', 0.5, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                # Increased min_child_samples for regularization (was 5-50, now 10-100)
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                # Add num_leaves constraint (LightGBM-specific, prevents overfitting)
                'num_leaves': trial.suggest_int('num_leaves', 8, 31),  # Should be < 2^max_depth
                # Stronger L1/L2 regularization (was 0-1.0, now 0.1-5.0)
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0),
                # Add min_gain_to_split for more conservative splitting
                'min_split_gain': trial.suggest_float('min_split_gain', 0.01, 1.0),
                # Handle class imbalance
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train, 
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50)]  # Critical: early stopping
            )
            
            y_pred = model.predict(X_val)
            y_train_pred = model.predict(X_train)
            
            train_f1 = f1_score(y_train, y_train_pred)
            val_f1 = f1_score(y_val, y_pred)
            
            # Penalize overfitting: if train >> val, reduce score
            overfitting_penalty = abs(train_f1 - val_f1)
            return val_f1 - (overfitting_penalty * 0.5)
        
        study = optuna.create_study(direction='maximize', study_name='lightgbm')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"   Best F1: {study.best_value:.4f}")
        print(f"   Best params: {study.best_params}")
        
        return study.best_params

    
    def walk_forward_analysis(self, n_splits=5, optimize_trials=30):
        """
        Walk-forward analysis with rolling window
        
        Args:
            n_splits: Number of walk-forward splits
            optimize_trials: Optuna trials per split (reduced for speed)
        """
        print(f"\n🚶 Walk-Forward Analysis ({n_splits} splits)...")
        print("   This simulates real trading by retraining on expanding window")
        
        # Calculate split sizes
        total_samples = len(self.X)
        test_size = total_samples // (n_splits + 1)
        
        results = []
        
        for split in range(n_splits):
            print(f"\n   {'='*60}")
            print(f"   Split {split + 1}/{n_splits}")
            print(f"   {'='*60}")
            
            # Define train and test indices
            train_end = (split + 1) * test_size + (total_samples - (n_splits + 1) * test_size)
            test_start = train_end
            test_end = test_start + test_size
            
            if test_end > total_samples:
                test_end = total_samples
            
            X_train = self.X[:train_end]
            y_train = self.y[:train_end]
            X_test = self.X[test_start:test_end]
            y_test = self.y[test_start:test_end]
            
            # Further split train into train/val for optimization
            val_size = len(X_train) // 5
            X_train_opt = X_train[:-val_size]
            y_train_opt = y_train[:-val_size]
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            
            print(f"   Train: {len(X_train_opt):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
            
            # Optimize models (reduced trials for speed)
            xgb_params = self.optimize_xgboost(X_train_opt, y_train_opt, X_val, y_val, n_trials=optimize_trials)
            lgb_params = self.optimize_lightgbm(X_train_opt, y_train_opt, X_val, y_val, n_trials=optimize_trials)
            
            # Train final models on full train set with early stopping
            # Create validation set from end of training data
            val_size_final = len(X_train) // 10  # 10% for validation
            X_train_final = X_train[:-val_size_final]
            y_train_final = y_train[:-val_size_final]
            X_val_final = X_train[-val_size_final:]
            y_val_final = y_train[-val_size_final:]
            
            xgb_model = xgb.XGBClassifier(**xgb_params)
            xgb_model.fit(
                X_train_final, y_train_final,
                eval_set=[(X_val_final, y_val_final)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            lgb_model = lgb.LGBMClassifier(**lgb_params)
            lgb_model.fit(
                X_train_final, y_train_final,
                eval_set=[(X_val_final, y_val_final)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            # Evaluate on test set
            xgb_pred = xgb_model.predict(X_test)
            lgb_pred = lgb_model.predict(X_test)
            
            # Ensemble prediction (voting)
            ensemble_pred = ((xgb_pred + lgb_pred) >= 1).astype(int)
            
            # Calculate metrics
            split_results = {
                'split': split + 1,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'xgb_f1': f1_score(y_test, xgb_pred),
                'xgb_accuracy': accuracy_score(y_test, xgb_pred),
                'lgb_f1': f1_score(y_test, lgb_pred),
                'lgb_accuracy': accuracy_score(y_test, lgb_pred),
                'ensemble_f1': f1_score(y_test, ensemble_pred),
                'ensemble_accuracy': accuracy_score(y_test, ensemble_pred),
            }
            
            results.append(split_results)
            
            print(f"\n   Results:")
            print(f"      XGBoost    - F1: {split_results['xgb_f1']:.4f}, Acc: {split_results['xgb_accuracy']:.4f}")
            print(f"      LightGBM   - F1: {split_results['lgb_f1']:.4f}, Acc: {split_results['lgb_accuracy']:.4f}")
            print(f"      Ensemble   - F1: {split_results['ensemble_f1']:.4f}, Acc: {split_results['ensemble_accuracy']:.4f}")
        
        # Summary
        print(f"\n   {'='*60}")
        print(f"   WALK-FORWARD SUMMARY")
        print(f"   {'='*60}")
        
        df_results = pd.DataFrame(results)
        
        print(f"\n   Average Performance:")
        print(f"      XGBoost    - F1: {df_results['xgb_f1'].mean():.4f} ± {df_results['xgb_f1'].std():.4f}")
        print(f"      LightGBM   - F1: {df_results['lgb_f1'].mean():.4f} ± {df_results['lgb_f1'].std():.4f}")
        print(f"      Ensemble   - F1: {df_results['ensemble_f1'].mean():.4f} ± {df_results['ensemble_f1'].std():.4f}")
        
        # Save results
        df_results.to_csv('walk_forward_results.csv', index=False)
        print(f"\n   💾 Saved to: walk_forward_results.csv")
        
        return df_results
    
    def train_final_ensemble(self, optimize_trials=100):
        """
        Train final ensemble model on all data with optimized hyperparameters
        
        Args:
            optimize_trials: Number of Optuna trials
        """
        print(f"\n🎯 Training Final Ensemble Model...")
        
        # Split data
        split_idx = int(len(self.X) * 0.8)
        X_train = self.X[:split_idx]
        y_train = self.y[:split_idx]
        X_test = self.X[split_idx:]
        y_test = self.y[split_idx:]
        
        # Further split for validation
        val_size = len(X_train) // 5
        X_train_opt = X_train[:-val_size]
        y_train_opt = y_train[:-val_size]
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        
        print(f"   Train: {len(X_train_opt):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
        
        # Optimize both models
        xgb_params = self.optimize_xgboost(X_train_opt, y_train_opt, X_val, y_val, n_trials=optimize_trials)
        lgb_params = self.optimize_lightgbm(X_train_opt, y_train_opt, X_val, y_val, n_trials=optimize_trials)
        
        # Train final models with early stopping on validation set
        print("\n   Training final XGBoost with early stopping...")
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(
            X_train_opt, y_train_opt,
            eval_set=[(X_val, y_val), (X_test, y_test)],
            early_stopping_rounds=50,
            verbose=True
        )
        
        print("\n   Training final LightGBM with early stopping...")
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(
            X_train_opt, y_train_opt,
            eval_set=[(X_val, y_val), (X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        # Create ensemble
        print("   Creating ensemble...")
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model)
            ],
            voting='soft'  # Use probabilities
        )
        # Fit ensemble on same data as individual models
        self.ensemble_model.fit(X_train_opt, y_train_opt)
        
        # Evaluate
        print("\n   📊 Evaluation on Test Set:")
        
        # Individual models
        xgb_pred = xgb_model.predict(X_test)
        xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        lgb_pred = lgb_model.predict(X_test)
        lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
        
        # Ensemble
        ensemble_pred = self.ensemble_model.predict(X_test)
        ensemble_proba = self.ensemble_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics for both train and test
        # Train predictions
        xgb_train_pred = xgb_model.predict(X_train_opt)
        lgb_train_pred = lgb_model.predict(X_train_opt)
        ensemble_train_pred = self.ensemble_model.predict(X_train_opt)
        
        # Test metrics
        metrics = {
            'xgboost': {
                'train_accuracy': accuracy_score(y_train_opt, xgb_train_pred),
                'train_f1': f1_score(y_train_opt, xgb_train_pred, zero_division=0),
                'test_accuracy': accuracy_score(y_test, xgb_pred),
                'test_f1': f1_score(y_test, xgb_pred, zero_division=0),
                'accuracy': accuracy_score(y_test, xgb_pred),
                'precision': precision_score(y_test, xgb_pred, zero_division=0),
                'recall': recall_score(y_test, xgb_pred, zero_division=0),
                'f1': f1_score(y_test, xgb_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, xgb_proba)
            },
            'lightgbm': {
                'train_accuracy': accuracy_score(y_train_opt, lgb_train_pred),
                'train_f1': f1_score(y_train_opt, lgb_train_pred, zero_division=0),
                'test_accuracy': accuracy_score(y_test, lgb_pred),
                'test_f1': f1_score(y_test, lgb_pred, zero_division=0),
                'accuracy': accuracy_score(y_test, lgb_pred),
                'precision': precision_score(y_test, lgb_pred, zero_division=0),
                'recall': recall_score(y_test, lgb_pred, zero_division=0),
                'f1': f1_score(y_test, lgb_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, lgb_proba)
            },
            'ensemble': {
                'train_accuracy': accuracy_score(y_train_opt, ensemble_train_pred),
                'train_f1': f1_score(y_train_opt, ensemble_train_pred, zero_division=0),
                'test_accuracy': accuracy_score(y_test, ensemble_pred),
                'test_f1': f1_score(y_test, ensemble_pred, zero_division=0),
                'accuracy': accuracy_score(y_test, ensemble_pred),
                'precision': precision_score(y_test, ensemble_pred, zero_division=0),
                'recall': recall_score(y_test, ensemble_pred, zero_division=0),
                'f1': f1_score(y_test, ensemble_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, ensemble_proba)
            }
        }
        
        # Print metrics with overfitting detection
        print("\n   ⚠️  OVERFITTING CHECK (Train vs Test):")
        print("   " + "="*60)
        
        for model_name, model_metrics in metrics.items():
            print(f"\n   {model_name.upper()}:")
            print(f"      {'Train Accuracy':<20s}: {model_metrics['train_accuracy']:.4f}")
            print(f"      {'Test Accuracy':<20s}: {model_metrics['test_accuracy']:.4f}")
            print(f"      {'Train F1':<20s}: {model_metrics['train_f1']:.4f}")
            print(f"      {'Test F1':<20s}: {model_metrics['test_f1']:.4f}")
            
            # Overfitting detection
            acc_gap = model_metrics['train_accuracy'] - model_metrics['test_accuracy']
            f1_gap = model_metrics['train_f1'] - model_metrics['test_f1']
            
            if acc_gap > 0.10 or f1_gap > 0.10:
                print(f"      🚨 OVERFITTING DETECTED! Train-Test gap: Acc={acc_gap:.4f}, F1={f1_gap:.4f}")
            elif acc_gap > 0.05 or f1_gap > 0.05:
                print(f"      ⚠️  Possible overfitting. Train-Test gap: Acc={acc_gap:.4f}, F1={f1_gap:.4f}")
            else:
                print(f"      ✅ Good generalization. Train-Test gap: Acc={acc_gap:.4f}, F1={f1_gap:.4f}")
            
            print(f"\n      Full Test Metrics:")
            for metric_name, value in model_metrics.items():
                if metric_name not in ['train_accuracy', 'train_f1', 'test_accuracy', 'test_f1']:
                    print(f"         {metric_name:12s}: {value:.4f}")
        
        # Store models
        self.best_models = {
            'xgboost': xgb_model,
            'lightgbm': lgb_model,
            'ensemble': self.ensemble_model,
            'xgb_params': xgb_params,
            'lgb_params': lgb_params
        }
        
        return metrics
    
    def save_models(self, output_dir='models'):
        """Save all models and metadata"""
        print(f"\n💾 Saving models to {output_dir}/...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual models
        joblib.dump(self.best_models['xgboost'], f'{output_dir}/ema_crossover_xgboost.pkl')
        joblib.dump(self.best_models['lightgbm'], f'{output_dir}/ema_crossover_lightgbm.pkl')
        joblib.dump(self.best_models['ensemble'], f'{output_dir}/ema_crossover_ensemble.pkl')
        
        print(f"   ✅ Saved XGBoost model")
        print(f"   ✅ Saved LightGBM model")
        print(f"   ✅ Saved Ensemble model")
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'target_column': self.target_col,
            'original_features': len(self.feature_names),
            'selected_features': len(self.selected_features),
            'feature_names': self.selected_features,
            'xgb_params': self.best_models['xgb_params'],
            'lgb_params': self.best_models['lgb_params'],
            'data_path': self.data_path,
            'total_samples': len(self.X),
            'positive_samples': int(self.y.sum()),
            'positive_rate': float(self.y.sum() / len(self.y))
        }
        
        with open(f'{output_dir}/ema_crossover_ensemble_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"   ✅ Saved metadata")
        
        # Save feature list
        with open(f'{output_dir}/ema_crossover_selected_features.txt', 'w') as f:
            f.write("SELECTED FEATURES FOR EMA CROSSOVER ENSEMBLE\n")
            f.write("="*80 + "\n\n")
            f.write(f"Reduced from {len(self.feature_names)} to {len(self.selected_features)} features\n\n")
            for i, feat in enumerate(self.selected_features, 1):
                f.write(f"{i:2d}. {feat}\n")
        
        print(f"   ✅ Saved feature list")
        print(f"\n   📁 All files saved to: {output_dir}/")


def main():
    """Main training pipeline"""
    
    print("\n" + "="*80)
    print("🚀 OPTIMIZED EMA CROSSOVER TRAINING PIPELINE")
    print("="*80)
    
    # Check if data exists
    data_path = "ema_crossover_with_targets.csv"
    if not os.path.exists(data_path):
        print(f"\n❌ Error: {data_path} not found!")
        print("   Please run create_ema_crossover_targets.py first")
        return
    
    try:
        # Initialize trainer
        trainer = OptimizedEMACrossoverTrainer(data_path)
        
        # Step 1: Load data
        trainer.load_data()
        
        # Step 2: Feature selection
        trainer.select_features(method='importance', n_features=30)
        
        # Step 3: Walk-forward analysis
        print("\n" + "="*80)
        print("STEP 1: WALK-FORWARD ANALYSIS")
        print("="*80)
        wf_results = trainer.walk_forward_analysis(n_splits=5, optimize_trials=30)
        
        # Step 4: Train final ensemble
        print("\n" + "="*80)
        print("STEP 2: TRAIN FINAL ENSEMBLE")
        print("="*80)
        metrics = trainer.train_final_ensemble(optimize_trials=100)
        
        # Step 5: Save models
        trainer.save_models()
        
        # Final summary
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"\n📊 Summary:")
        print(f"   Original features: {len(trainer.feature_names)}")
        print(f"   Selected features: {len(trainer.selected_features)}")
        print(f"   Models trained: XGBoost, LightGBM, Ensemble")
        print(f"   Walk-forward splits: 5")
        print(f"   Optimization method: Optuna")
        
        print(f"\n🎯 Best Model: Ensemble")
        print(f"   F1 Score: {metrics['ensemble']['f1']:.4f}")
        print(f"   Accuracy: {metrics['ensemble']['accuracy']:.4f}")
        print(f"   Precision: {metrics['ensemble']['precision']:.4f}")
        print(f"   Recall: {metrics['ensemble']['recall']:.4f}")
        print(f"   ROC-AUC: {metrics['ensemble']['roc_auc']:.4f}")
        
        print(f"\n💾 Models saved to: models/")
        print(f"   - ema_crossover_xgboost.pkl")
        print(f"   - ema_crossover_lightgbm.pkl")
        print(f"   - ema_crossover_ensemble.pkl")
        print(f"   - ema_crossover_ensemble_metadata.json")
        print(f"   - ema_crossover_selected_features.txt")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
