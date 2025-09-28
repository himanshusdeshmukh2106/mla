"""
Example usage of ModelTrainer for XGBoost model training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.trainer import ModelTrainer, TrainingConfig, HyperparameterGrid
from src.models.data_splitter import DataSplitter


def create_sample_data(n_samples=2000):
    """Create sample financial data for demonstration"""
    np.random.seed(42)
    
    # Create datetime index
    dates = pd.date_range(
        start='2023-01-01', 
        periods=n_samples, 
        freq='5min'
    )
    
    # Create realistic financial features
    price = 100 + np.cumsum(np.random.randn(n_samples) * 0.1)
    
    # Technical indicators (simplified)
    sma_20 = pd.Series(price).rolling(20).mean()
    rsi = pd.Series(price).pct_change().rolling(14).apply(
        lambda x: 100 - (100 / (1 + (x[x > 0].mean() / abs(x[x < 0].mean()))))
    )
    volatility = pd.Series(price).pct_change().rolling(20).std()
    volume = np.random.lognormal(10, 0.5, n_samples)
    
    # Create target: 1 if next period price increases, 0 otherwise
    target = (pd.Series(price).shift(-1) > pd.Series(price)).astype(int)
    
    data = pd.DataFrame({
        'datetime': dates,
        'price': price,
        'sma_20': sma_20,
        'rsi': rsi,
        'volatility': volatility,
        'volume': volume,
        'target': target
    })
    
    # Remove NaN values
    data = data.dropna().reset_index(drop=True)
    
    return data


def main():
    """Main example function"""
    print("XGBoost Model Training Example")
    print("=" * 40)
    
    # Create sample data
    print("Creating sample financial data...")
    data = create_sample_data(2000)
    print(f"Created dataset with {len(data)} samples and {len(data.columns)-2} features")
    print(f"Target distribution: {data['target'].value_counts().to_dict()}")
    
    # Split data chronologically
    print("\nSplitting data chronologically...")
    splitter = DataSplitter(test_size=0.2, target_column='target', datetime_column='datetime')
    data_split = splitter.split_data_chronologically(data)
    
    print(f"Training samples: {len(data_split.X_train)}")
    print(f"Test samples: {len(data_split.X_test)}")
    print(f"Features: {data_split.feature_names}")
    
    # Configure model training
    print("\nConfiguring model training...")
    config = TrainingConfig(
        cv_splits=3,
        early_stopping_rounds=None,  # Disable for cross-validation
        verbose=False
    )
    
    # Define hyperparameter grid (reduced for faster execution)
    grid = HyperparameterGrid(
        max_depth=[3, 5, 7],
        learning_rate=[0.01, 0.1, 0.2],
        n_estimators=[50, 100, 200],
        gamma=[0, 0.1],
        subsample=[0.8, 1.0]
    )
    
    # Create trainer
    trainer = ModelTrainer(config=config, hyperparameter_grid=grid)
    
    # Train and evaluate model
    print("\nTraining XGBoost model with hyperparameter tuning...")
    print("This may take a few minutes...")
    
    model, metrics = trainer.train_and_evaluate(data_split)
    
    # Display results
    print("\nTraining Results:")
    print("=" * 20)
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall: {metrics.recall:.4f}")
    print(f"F1-Score: {metrics.f1_score:.4f}")
    
    print(f"\nBest hyperparameters: {trainer.best_params}")
    
    print("\nFeature Importance:")
    for feature, importance in sorted(metrics.feature_importance.items(), 
                                    key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(metrics.confusion_matrix)
    
    # Save model
    print("\nSaving trained model...")
    model_path = "trained_xgboost_model.pkl"
    trainer.save_model(model, model_path, {
        'training_date': datetime.now().isoformat(),
        'data_samples': len(data),
        'accuracy': metrics.accuracy
    })
    print(f"Model saved to {model_path}")
    
    # Demonstrate model loading
    print("\nLoading model...")
    loaded_model, metadata = trainer.load_model(model_path)
    print(f"Model loaded successfully")
    print(f"Metadata: {metadata}")
    
    # Make predictions on a few samples
    print("\nMaking predictions on test samples...")
    sample_predictions = loaded_model.predict(data_split.X_test[:5])
    sample_probabilities = loaded_model.predict_proba(data_split.X_test[:5])
    
    print("Sample predictions:")
    for i, (pred, prob) in enumerate(zip(sample_predictions, sample_probabilities)):
        print(f"  Sample {i+1}: Prediction={pred}, Probability=[{prob[0]:.3f}, {prob[1]:.3f}]")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()