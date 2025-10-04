"""
EMA Crossover Strategy - Model Training
Train XGBoost models for EMA crossover + retest strategy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def prepare_training_data(df: pd.DataFrame, target_column: str = 'target_combined_1.0R'):
    """
    Prepare data for model training
    """
    
    print(f"Preparing training data for target: {target_column}")
    
    # Define feature columns (exclude OHLCV, datetime, and target columns)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'datetime', 'timestamp']
    target_cols = [col for col in df.columns if 'target_' in col or 'direction_' in col]
    exclude_cols.extend(target_cols)
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Total samples: {len(df)}")
    
    # Prepare features and target
    X = df[feature_cols].copy()
    y = df[target_column].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Check target distribution
    target_dist = y.value_counts()
    print(f"  Target distribution:")
    for value, count in target_dist.items():
        print(f"    {value}: {count} ({count/len(y)*100:.1f}%)")
    
    return X, y, feature_cols


def train_xgboost_model(X, y, feature_names, model_name="ema_crossover"):
    """
    Train XGBoost model with proper validation
    """
    
    print(f"\nTraining XGBoost model: {model_name}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Define XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Train model
    print("  Training model...")
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    print("  Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Print classification report
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n  Top 15 Most Important Features:")
    for idx, row in feature_importance.head(15).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    # Cross-validation
    print("\n  Cross-validation scores:")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"    CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Test different probability thresholds
    print("\n  Threshold Analysis:")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        accuracy = (y_pred_thresh == y_test).mean()
        precision = ((y_pred_thresh == 1) & (y_test == 1)).sum() / max((y_pred_thresh == 1).sum(), 1)
        recall = ((y_pred_thresh == 1) & (y_test == 1)).sum() / max((y_test == 1).sum(), 1)
        
        print(f"    Threshold {threshold}: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}")
    
    return model, feature_importance, X_test, y_test, y_pred_proba


def save_model_and_metadata(model, feature_names, feature_importance, model_name):
    """
    Save model and metadata
    """
    
    print(f"\nSaving model: {model_name}")
    
    # Save model
    model_path = f"models/{model_name}.pkl"
    joblib.dump(model, model_path)
    print(f"  Model saved to: {model_path}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'model_type': 'XGBClassifier',
        'features': feature_names,
        'n_features': len(feature_names),
        'training_date': datetime.now().isoformat(),
        'feature_importance': feature_importance.to_dict('records')
    }
    
    metadata_path = f"models/{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to: {metadata_path}")
    
    # Save feature importance
    importance_path = f"models/{model_name}_feature_importance.csv"
    feature_importance.to_csv(importance_path, index=False)
    print(f"  Feature importance saved to: {importance_path}")


def train_multiple_models(df: pd.DataFrame):
    """
    Train models for different targets
    """
    
    print("="*80)
    print("TRAINING MULTIPLE EMA CROSSOVER MODELS")
    print("="*80)
    
    models_trained = []
    
    # Model configurations
    model_configs = [
        {
            'target': 'target_combined_1.0R',
            'name': 'ema_crossover_1R',
            'description': '1:1 Risk-Reward Combined Model'
        },
        {
            'target': 'target_combined_2.0R',
            'name': 'ema_crossover_2R',
            'description': '1:2 Risk-Reward Combined Model'
        },
        {
            'target': 'target_long_1.0R',
            'name': 'ema_crossover_long_1R',
            'description': 'Long-only 1:1 Risk-Reward Model'
        },
        {
            'target': 'target_short_1.0R',
            'name': 'ema_crossover_short_1R',
            'description': 'Short-only 1:1 Risk-Reward Model'
        }
    ]
    
    for config in model_configs:
        print(f"\n{'='*60}")
        print(f"TRAINING: {config['description']}")
        print(f"{'='*60}")
        
        try:
            # Prepare data
            X, y, feature_names = prepare_training_data(df, config['target'])
            
            # Skip if insufficient positive samples
            if y.sum() < 50:
                print(f"  Skipping {config['name']} - insufficient positive samples ({y.sum()})")
                continue
            
            # Train model
            model, feature_importance, X_test, y_test, y_pred_proba = train_xgboost_model(
                X, y, feature_names, config['name']
            )
            
            # Save model
            save_model_and_metadata(model, feature_names, feature_importance, config['name'])
            
            models_trained.append({
                'name': config['name'],
                'target': config['target'],
                'description': config['description'],
                'n_features': len(feature_names),
                'positive_samples': y.sum(),
                'total_samples': len(y)
            })
            
        except Exception as e:
            print(f"  Error training {config['name']}: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    
    for model_info in models_trained:
        print(f"\n✅ {model_info['name']}")
        print(f"   Target: {model_info['target']}")
        print(f"   Description: {model_info['description']}")
        print(f"   Features: {model_info['n_features']}")
        print(f"   Positive samples: {model_info['positive_samples']}/{model_info['total_samples']}")
    
    return models_trained


if __name__ == "__main__":
    print("EMA Crossover Model Training")
    print("="*80)
    
    try:
        # Load data with targets
        df = pd.read_csv("ema_crossover_with_targets.csv")
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"Loaded {len(df)} rows of training data")
        
        # Train models
        models_trained = train_multiple_models(df)
        
        print(f"\n🎉 Training complete! {len(models_trained)} models trained successfully.")
        
        # Recommendations
        print(f"\n📋 RECOMMENDATIONS:")
        print(f"   • Start with 'ema_crossover_1R' for best balance of signals and accuracy")
        print(f"   • Use 'ema_crossover_2R' for higher quality but fewer signals")
        print(f"   • Consider separate long/short models if directional bias exists")
        print(f"   • Test different probability thresholds (0.6-0.8) for entry filtering")
        
    except FileNotFoundError:
        print("Training data not found. Please run create_ema_crossover_targets.py first.")
    except Exception as e:
        print(f"Error: {e}")