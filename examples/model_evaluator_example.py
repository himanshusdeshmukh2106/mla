"""
Example usage of ModelEvaluator for comprehensive model evaluation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)

from src.models.trainer import ModelTrainer, TrainingConfig, HyperparameterGrid
from src.models.data_splitter import DataSplitter
from src.models.evaluator import ModelEvaluator
from src.features.engineer import FeatureEngineer
from src.features.target_generator import TargetGenerator


def create_sample_data():
    """Create sample OHLCV data for demonstration"""
    np.random.seed(42)
    n_samples = 2000
    
    # Create datetime index
    dates = pd.date_range(
        start='2023-01-01', 
        periods=n_samples, 
        freq='5min'
    )
    
    # Create realistic OHLCV data with some trends
    base_price = 100
    price_changes = np.random.randn(n_samples) * 0.02
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'datetime': dates,
        'open': prices * (1 + np.random.randn(n_samples) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n_samples)) * 0.002),
        'low': prices * (1 - np.abs(np.random.randn(n_samples)) * 0.002),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Ensure OHLC consistency
    data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
    data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
    
    return data


def main():
    """Main example function"""
    print("=" * 60)
    print("MODEL EVALUATOR EXAMPLE")
    print("=" * 60)
    
    # Step 1: Create sample data
    print("\n1. Creating sample OHLCV data...")
    raw_data = create_sample_data()
    print(f"Created {len(raw_data)} samples of OHLCV data")
    
    # Step 2: Engineer features
    print("\n2. Engineering features...")
    feature_config = {
        'trend': {
            'sma_periods': [20, 50],
            'ema_periods': [12, 26]
        },
        'momentum': {
            'rsi_period': 14,
            'macd': {'fast': 12, 'slow': 26, 'signal': 9}
        },
        'volatility': {
            'bollinger_bands': {'period': 20, 'std': 2},
            'atr_period': 14
        },
        'volume': {
            'volume_sma_periods': [20]
        }
    }
    feature_engineer = FeatureEngineer(feature_config)
    featured_data = feature_engineer.create_features(raw_data)
    print(f"Created {len([col for col in featured_data.columns if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume']])} features")
    
    # Step 3: Generate target variable
    print("\n3. Generating target variable...")
    target_config = {
        'lookahead_periods': 1,
        'profit_threshold': 0.001,
        'loss_threshold': -0.001,
        'method': 'next_period_return'
    }
    target_generator = TargetGenerator(target_config)
    final_data = target_generator.generate_targets(featured_data)
    print(f"Generated binary target with {final_data['target'].sum()} positive samples out of {len(final_data)}")
    
    # Step 4: Split data
    print("\n4. Splitting data chronologically...")
    splitter = DataSplitter(
        test_size=0.2, 
        target_column='target_binary',
        exclude_columns=['target', 'future_return', 'future_max_return', 'future_min_return']
    )
    data_split = splitter.split_data_chronologically(final_data)
    print(f"Training samples: {len(data_split.X_train)}")
    print(f"Testing samples: {len(data_split.X_test)}")
    
    # Step 5: Train model
    print("\n5. Training XGBoost model...")
    config = TrainingConfig(cv_splits=3, verbose=False, early_stopping_rounds=None)
    grid = HyperparameterGrid(
        max_depth=[3, 5],
        learning_rate=[0.1, 0.2],
        n_estimators=[50, 100]
    )
    trainer = ModelTrainer(config=config, hyperparameter_grid=grid)
    
    model, basic_metrics = trainer.train_and_evaluate(data_split)
    print(f"Model trained with accuracy: {basic_metrics.accuracy:.4f}")
    
    # Step 6: Comprehensive model evaluation
    print("\n6. Performing comprehensive model evaluation...")
    evaluator = ModelEvaluator(feature_names=data_split.feature_names)
    
    # Detailed evaluation
    detailed_metrics = evaluator.evaluate_model(
        model, 
        data_split.X_test, 
        data_split.y_test, 
        data_split.feature_names
    )
    
    print(f"Detailed evaluation completed:")
    print(f"  - Accuracy: {detailed_metrics.accuracy:.4f}")
    print(f"  - Precision: {detailed_metrics.precision:.4f}")
    print(f"  - Recall: {detailed_metrics.recall:.4f}")
    print(f"  - F1-Score: {detailed_metrics.f1_score:.4f}")
    print(f"  - ROC-AUC: {detailed_metrics.roc_auc:.4f}")
    print(f"  - Average Precision: {detailed_metrics.average_precision:.4f}")
    
    # Step 7: Confusion matrix analysis
    print("\n7. Analyzing confusion matrix...")
    cm_analysis = evaluator.analyze_confusion_matrix(detailed_metrics.confusion_matrix)
    
    print(f"Confusion Matrix Analysis:")
    print(f"  - True Positives: {cm_analysis.true_positives}")
    print(f"  - True Negatives: {cm_analysis.true_negatives}")
    print(f"  - False Positives: {cm_analysis.false_positives}")
    print(f"  - False Negatives: {cm_analysis.false_negatives}")
    print(f"  - Sensitivity (Recall): {cm_analysis.sensitivity:.4f}")
    print(f"  - Specificity: {cm_analysis.specificity:.4f}")
    print(f"  - Positive Predictive Value (Precision): {cm_analysis.positive_predictive_value:.4f}")
    print(f"  - Negative Predictive Value: {cm_analysis.negative_predictive_value:.4f}")
    
    # Step 8: Feature importance analysis
    print("\n8. Analyzing feature importance...")
    feature_analysis = evaluator.analyze_feature_importance(
        model, 
        data_split.feature_names, 
        top_n=10
    )
    
    print(f"Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(feature_analysis.ranked_features[:10], 1):
        print(f"  {i:2d}. {feature:<25} {importance:.4f}")
    
    # Step 9: Generate comprehensive report
    print("\n9. Generating comprehensive evaluation report...")
    report = evaluator.generate_evaluation_report(
        detailed_metrics, 
        cm_analysis, 
        feature_analysis
    )
    
    print("\n" + report)
    
    # Step 10: Model comparison example
    print("\n10. Demonstrating model comparison...")
    
    # Train a second model with different parameters for comparison
    config2 = TrainingConfig(cv_splits=3, verbose=False, early_stopping_rounds=None)
    grid2 = HyperparameterGrid(
        max_depth=[2, 4],
        learning_rate=[0.05, 0.15],
        n_estimators=[30, 80]
    )
    trainer2 = ModelTrainer(config=config2, hyperparameter_grid=grid2)
    
    model2, _ = trainer2.train_and_evaluate(data_split)
    
    # Evaluate second model
    detailed_metrics2 = evaluator.evaluate_model(
        model2, 
        data_split.X_test, 
        data_split.y_test, 
        data_split.feature_names
    )
    
    # Compare models
    comparison_df = evaluator.compare_models({
        'Model_1 (Original)': detailed_metrics,
        'Model_2 (Alternative)': detailed_metrics2
    })
    
    print("\nModel Comparison:")
    print(comparison_df.round(4))
    
    # Determine best model
    best_model_name = comparison_df['Accuracy'].idxmax()
    print(f"\nBest model by accuracy: {best_model_name}")
    print(f"Best accuracy: {comparison_df.loc[best_model_name, 'Accuracy']:.4f}")
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION EXAMPLE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()