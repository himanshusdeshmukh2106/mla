"""
Example demonstrating model persistence functionality
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.persistence import ModelPersistence, PersistenceConfig, ModelState
from src.models.trainer import ModelTrainer, TrainingConfig, HyperparameterGrid
from src.models.data_splitter import DataSplitter
from src.features.engineer import FeatureEngineer
from src.features.target_generator import TargetGenerator
from src.config_manager import ConfigManager
from src.logger import setup_logging


def create_sample_data():
    """Create sample OHLCV data for demonstration"""
    np.random.seed(42)
    
    # Generate sample price data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    
    # Generate realistic OHLCV data
    base_price = 100.0
    prices = []
    volume_base = 1000
    
    for i in range(len(dates)):
        # Random walk with some trend
        change = np.random.normal(0, 0.5)
        base_price += change
        
        # Generate OHLCV
        open_price = base_price
        high_price = open_price + abs(np.random.normal(0, 0.3))
        low_price = open_price - abs(np.random.normal(0, 0.3))
        close_price = open_price + np.random.normal(0, 0.2)
        volume = volume_base + np.random.randint(-200, 200)
        
        prices.append({
            'datetime': dates[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        base_price = close_price
    
    return pd.DataFrame(prices)


def demonstrate_model_persistence():
    """Demonstrate complete model persistence workflow"""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model persistence demonstration...")
    
    try:
        # 1. Create sample data and prepare features
        logger.info("Creating sample data...")
        data = create_sample_data()
        
        # 2. Engineer features
        logger.info("Engineering features...")
        feature_config = {
            'trend_periods': [10, 20],
            'momentum_periods': {'rsi': 14, 'macd_fast': 12, 'macd_slow': 26},
            'volatility_periods': {'bb': 20, 'atr': 14},
            'volume_periods': [10]
        }
        
        feature_engineer = FeatureEngineer(feature_config)
        data_with_features = feature_engineer.create_features(data)
        
        # 3. Generate targets
        logger.info("Generating target variables...")
        target_config = {
            'lookahead_periods': 1,
            'profit_threshold': 0.001,
            'loss_threshold': -0.001,
            'method': 'next_period_return'
        }
        target_generator = TargetGenerator(target_config)
        data_with_targets = target_generator.generate_targets(data_with_features)
        
        # Use the binary target for training (rename target_binary to target)
        if 'target_binary' in data_with_targets.columns:
            data_with_targets['target'] = data_with_targets['target_binary']
        
        # 4. Split data
        logger.info("Splitting data...")
        data_splitter = DataSplitter()
        data_split = data_splitter.split_data_chronologically(data_with_targets)
        
        # 5. Train model
        logger.info("Training model...")
        training_config = TrainingConfig(
            test_size=0.2,
            cv_splits=3,  # Reduced for demo
            early_stopping_rounds=5,
            verbose=True
        )
        
        hyperparameter_grid = HyperparameterGrid(
            max_depth=[3, 5],  # Reduced grid for demo
            learning_rate=[0.1, 0.2],
            n_estimators=[50, 100]
        )
        
        # Configure persistence
        persistence_config = PersistenceConfig(
            base_path="demo_models",
            compress_model=True,
            backup_existing=True,
            validate_on_load=True
        )
        
        trainer = ModelTrainer(
            config=training_config,
            hyperparameter_grid=hyperparameter_grid,
            persistence_config=persistence_config
        )
        
        # Train and evaluate model
        model, metrics = trainer.train_and_evaluate(data_split)
        
        logger.info(f"Model training completed. Accuracy: {metrics.accuracy:.4f}")
        
        # 6. Save model using comprehensive persistence
        logger.info("Saving model with comprehensive persistence...")
        model_name = f"demo_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        training_metadata = {
            'training_duration': 120,  # seconds
            'data_samples': len(data_with_targets),
            'feature_count': len(data_split.feature_names),
            'training_date': datetime.now().isoformat(),
            'data_period': f"{data['datetime'].min()} to {data['datetime'].max()}"
        }
        
        saved_path = trainer.save_model_comprehensive(
            model=model,
            model_name=model_name,
            feature_config=feature_config,
            performance_metrics=metrics,
            training_metadata=training_metadata
        )
        
        logger.info(f"Model saved to: {saved_path}")
        
        # 7. List saved models
        logger.info("Listing saved models...")
        saved_models = trainer.list_saved_models()
        
        print("\n" + "="*60)
        print("SAVED MODELS:")
        print("="*60)
        
        for model_info in saved_models:
            print(f"Name: {model_info['name']}")
            print(f"Created: {model_info['created_at']}")
            print(f"Type: {model_info['model_type']}")
            print(f"Features: {model_info['feature_count']}")
            if model_info['performance_metrics']:
                print(f"Accuracy: {model_info['performance_metrics'].get('accuracy', 'N/A')}")
            print("-" * 40)
        
        # 8. Load model back
        logger.info(f"Loading model '{model_name}'...")
        loaded_model, model_state_dict, metadata = trainer.load_model_comprehensive(model_name)
        
        print("\n" + "="*60)
        print("LOADED MODEL STATE:")
        print("="*60)
        print(f"Model Type: {model_state_dict['model_type']}")
        print(f"Model Version: {model_state_dict['model_version']}")
        print(f"Feature Count: {len(model_state_dict['feature_names'])}")
        print(f"Created At: {model_state_dict['created_at']}")
        print(f"Hyperparameters: {model_state_dict['hyperparameters']}")
        
        if model_state_dict['performance_metrics']:
            print(f"Performance Metrics:")
            for metric, value in model_state_dict['performance_metrics'].items():
                if metric != 'confusion_matrix':  # Skip matrix for readability
                    print(f"  {metric}: {value}")
        
        # 9. Test loaded model
        logger.info("Testing loaded model...")
        test_predictions = loaded_model.predict(data_split.X_test[:10])
        original_predictions = model.predict(data_split.X_test[:10])
        
        predictions_match = np.array_equal(test_predictions, original_predictions)
        print(f"\nPredictions match original model: {predictions_match}")
        
        # 10. Export model
        logger.info("Exporting model...")
        export_path = "exported_models"
        exported_path = trainer.export_saved_model(model_name, export_path)
        logger.info(f"Model exported to: {exported_path}")
        
        # 11. Demonstrate direct persistence usage
        logger.info("Demonstrating direct persistence usage...")
        
        persistence = ModelPersistence(persistence_config)
        
        # Save another model directly
        direct_model_name = f"direct_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        direct_saved_path = persistence.save_model(
            model=model,
            model_name=direct_model_name,
            feature_names=data_split.feature_names,
            hyperparameters=trainer.best_params,
            training_config=training_config.__dict__,
            feature_config=feature_config,
            performance_metrics=metrics,
            training_metadata=training_metadata
        )
        
        logger.info(f"Direct model saved to: {direct_saved_path}")
        
        # Load it back
        direct_loaded_model, direct_model_state, direct_metadata = persistence.load_model(direct_model_name)
        
        print(f"\nDirect persistence - Model loaded successfully")
        print(f"Feature names match: {direct_model_state.feature_names == data_split.feature_names}")
        print(f"Hyperparameters match: {direct_model_state.hyperparameters == trainer.best_params}")
        
        logger.info("Model persistence demonstration completed successfully!")
        
        return {
            'saved_models': saved_models,
            'model_name': model_name,
            'direct_model_name': direct_model_name,
            'predictions_match': predictions_match
        }
        
    except Exception as e:
        logger.error(f"Error in model persistence demonstration: {str(e)}")
        raise


def cleanup_demo_files():
    """Clean up demonstration files"""
    import shutil
    
    directories_to_clean = ['demo_models', 'exported_models']
    
    for directory in directories_to_clean:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                print(f"Cleaned up {directory}")
            except Exception as e:
                print(f"Could not clean up {directory}: {e}")


if __name__ == "__main__":
    try:
        # Run demonstration
        results = demonstrate_model_persistence()
        
        print("\n" + "="*60)
        print("DEMONSTRATION SUMMARY:")
        print("="*60)
        print(f"Models saved: {len(results['saved_models'])}")
        print(f"Main model: {results['model_name']}")
        print(f"Direct model: {results['direct_model_name']}")
        print(f"Predictions consistent: {results['predictions_match']}")
        print("="*60)
        
        # Ask user if they want to clean up
        response = input("\nClean up demonstration files? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            cleanup_demo_files()
            print("Cleanup completed.")
        else:
            print("Demo files preserved for inspection.")
            
    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()