#!/usr/bin/env python3
"""
EMA Trap Strategy Model Training Script
Train XGBoost model on Reliance 5-minute data with EMA trap features

USAGE:
------
Local Environment:
1. Ensure data file is in: data/reliance_data_5min_full_year.csv
2. Run: python train_ema_trap_model.py

Google Colab:
1. Clone repository: !git clone <your-repo-url>
2. Change directory: %cd <repo-name>
3. Upload data file to Colab or place in /content/
4. Install requirements: !pip install -r requirements.txt
5. Run: !python train_ema_trap_model.py

The script will automatically detect the environment and adapt accordingly.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def setup_environment():
    """Setup environment for both local and Colab execution"""
    
    # Check if running in Google Colab
    if 'google.colab' in sys.modules:
        print("🔍 Detected Google Colab environment")
        
        # Install additional packages if needed
        import subprocess
        packages_to_install = ['pandas-ta']
        
        for package in packages_to_install:
            try:
                __import__(package.replace('-', '_'))
                print(f"✅ {package} already installed")
            except ImportError:
                print(f"📦 Installing {package}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ {package} installed successfully")
        
        print("🚀 Colab environment setup complete!")
    else:
        print("🖥️  Detected local environment")
    
    # Add src to path
    sys.path.append('src')

# Setup environment
setup_environment()

from features.engineer import FeatureEngineer
from features.target_generator import TargetGenerator
from models.trainer import ModelTrainer
from models.evaluator import ModelEvaluator
from models.persistence import ModelPersistence
from config_manager import ConfigManager
from logger import get_logger

logger = get_logger(__name__)

def load_reliance_data():
    """Load and prepare Reliance 5-minute data"""
    
    # Try multiple possible data paths (local, Colab, etc.)
    possible_paths = [
        r"C:\Users\Lenovo\Desktop\ml\data\reliance_data_5min_full_year.csv",  # Local Windows
        "data/reliance_data_5min_full_year.csv",  # Local relative
        "/content/reliance_data_5min_full_year.csv",  # Colab root
        "/content/data/reliance_data_5min_full_year.csv",  # Colab data folder
        "./reliance_data_5min_full_year.csv",  # Current directory
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(f"Data file not found in any of these locations: {possible_paths}")
    
    logger.info(f"Loading data from: {data_path}")
    
    # Load the data
    df = pd.read_csv(data_path)
    
    # Parse datetime and set as index
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Sort by datetime to ensure proper order
    df.sort_index(inplace=True)
    
    # Basic data validation
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Remove any rows with NaN values
    initial_rows = len(df)
    df.dropna(inplace=True)
    final_rows = len(df)
    
    if initial_rows != final_rows:
        logger.warning(f"Removed {initial_rows - final_rows} rows with NaN values")
    
    logger.info(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
    
    return df

def create_ema_trap_features(data):
    """Create EMA trap features from raw OHLCV data"""
    
    logger.info("Creating EMA trap features...")
    
    # Load feature configuration
    config_manager = ConfigManager()
    feature_config = config_manager.get_config('feature')
    
    # Initialize feature engineer
    engineer = FeatureEngineer(feature_config)
    
    # Create all features
    features_df = engineer.create_features(data)
    
    logger.info(f"Features created: {len(features_df)} rows, {len(features_df.columns)} columns")
    
    return features_df

def generate_ema_trap_targets(features_df):
    """Generate targets for EMA trap strategy"""
    
    logger.info("Generating EMA trap targets...")
    
    # Target configuration for EMA trap strategy
    target_config = {
        'method': 'ema_trap',
        'lookahead_periods': 2,     # Look ahead 2 candles (10 minutes)
        'profit_threshold': 0.004,  # 0.4% profit target
        'loss_threshold': -0.004    # -0.4% loss threshold
    }
    
    # Initialize target generator
    target_generator = TargetGenerator(target_config)
    
    # Generate targets
    targets_df = target_generator.generate_targets(features_df)
    
    # Get target distribution
    distribution = target_generator.get_target_distribution(targets_df)
    logger.info(f"Target distribution: {distribution}")
    
    # Validate targets
    is_valid, validation_info = target_generator.validate_targets(targets_df)
    if not is_valid:
        logger.error(f"Target validation failed: {validation_info}")
        raise ValueError("Target validation failed")
    
    logger.info(f"Target validation passed: {validation_info}")
    
    return targets_df

def select_features_for_training(data):
    """Select the most relevant features for EMA trap strategy"""
    
    # Core EMA trap features
    ema_trap_features = [
        'EMA_21', 'ADX', 'Distance_From_EMA21_Pct',
        'Bearish_Trap_Confirmed', 'Bullish_Trap_Confirmed',
        'In_Entry_Window', 'ADX_In_Range', 'Candle_Body_Size_Pct'
    ]
    
    # Supporting technical indicators
    technical_features = [
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'BB_Position', 'ATR', 'Volatility_Ratio',
        'Volume_Ratio_20', 'OBV'
    ]
    
    # Time-based features
    time_features = [
        'Hour', 'Minute', 'Entry_Window_1', 'Entry_Window_2',
        'Market_Open_Hour', 'First_Hour'
    ]
    
    # Candle pattern features
    candle_features = [
        'Green_Candle', 'Red_Candle', 'Upper_Shadow_Pct', 
        'Lower_Shadow_Pct', 'Body_To_Range_Ratio'
    ]
    
    # Combine all feature groups
    selected_features = ema_trap_features + technical_features + time_features + candle_features
    
    # Filter to only include features that exist in the data
    available_features = [f for f in selected_features if f in data.columns]
    missing_features = [f for f in selected_features if f not in data.columns]
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    
    logger.info(f"Selected {len(available_features)} features for training")
    
    return available_features

def prepare_training_data(targets_df, selected_features):
    """Prepare data for model training"""
    
    logger.info("Preparing training data...")
    
    # Select features and target
    X = targets_df[selected_features].copy()
    y = targets_df['target_binary'].copy()
    
    # Remove any remaining NaN values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    logger.info(f"Training data prepared: {len(X)} samples, {len(selected_features)} features")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Check for class imbalance
    class_counts = y.value_counts()
    if len(class_counts) > 1:
        minority_class_pct = min(class_counts) / len(y) * 100
        logger.info(f"Minority class percentage: {minority_class_pct:.2f}%")
        
        if minority_class_pct < 5:
            logger.warning("Severe class imbalance detected. Consider using class weights or resampling.")
    
    return X, y

def train_ema_trap_model(X, y, selected_features):
    """Train XGBoost model for EMA trap strategy using advanced training pipeline"""
    
    logger.info("Training EMA trap XGBoost model with advanced features...")
    
    # Import advanced training components
    from models.trainer import ModelTrainer, TrainingConfig, HyperparameterGrid
    from models.data_splitter import DataSplitter
    from models.persistence import PersistenceConfig
    
    # Advanced training configuration optimized for EMA trap strategy
    training_config = TrainingConfig(
        algorithm="xgboost",
        objective="binary:logistic",
        test_size=0.2,
        cv_splits=5,
        early_stopping_rounds=50,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbose=True
    )
    
    # Advanced hyperparameter grid for EMA trap strategy
    hyperparameter_grid = HyperparameterGrid(
        max_depth=[4, 6, 8, 10],
        learning_rate=[0.05, 0.1, 0.15, 0.2],
        n_estimators=[200, 300, 500, 800],
        gamma=[0, 0.1, 0.2, 0.3],
        subsample=[0.7, 0.8, 0.9, 1.0],
        colsample_bytree=[0.7, 0.8, 0.9, 1.0],
        min_child_weight=[1, 3, 5, 7],
        reg_alpha=[0, 0.1, 0.5, 1.0],
        reg_lambda=[1, 1.5, 2, 3]
    )
    
    # Advanced persistence configuration
    persistence_config = PersistenceConfig(
        base_path="models/ema_trap",
        model_filename="ema_trap_model.pkl",
        state_filename="ema_trap_state.json",
        metadata_filename="ema_trap_metadata.pkl",
        compress_model=True,
        backup_existing=True,
        validate_on_load=True
    )
    
    # Initialize advanced model trainer
    trainer = ModelTrainer(
        config=training_config,
        hyperparameter_grid=hyperparameter_grid,
        persistence_config=persistence_config
    )
    
    # Create advanced data splitter for time-series aware splitting
    data_splitter = DataSplitter({
        'method': 'time_series',
        'test_size': 0.2,
        'validation_size': 0.1,
        'shuffle': False,  # Important for time series
        'random_state': 42
    })
    
    # Prepare data with feature names
    data_df = pd.DataFrame(X, columns=selected_features)
    data_df['target'] = y
    
    # Create time-series aware data split
    data_split = data_splitter.split_data(data_df)
    
    logger.info(f"Data split - Train: {len(data_split.X_train)}, "
               f"Val: {len(data_split.X_val) if hasattr(data_split, 'X_val') else 0}, "
               f"Test: {len(data_split.X_test)}")
    
    # Train model using advanced pipeline
    model, metrics = trainer.train_and_evaluate(data_split)
    
    # Get training results
    training_results = {
        'best_params': trainer.best_params,
        'feature_names': trainer.feature_names,
        'training_config': training_config.__dict__,
        'hyperparameter_grid': hyperparameter_grid.__dict__,
        'data_split_info': {
            'train_samples': len(data_split.X_train),
            'test_samples': len(data_split.X_test),
            'feature_count': len(selected_features)
        }
    }
    
    logger.info("Advanced model training completed")
    logger.info(f"Best parameters: {trainer.best_params}")
    logger.info(f"Training accuracy: {metrics.accuracy:.4f}")
    logger.info(f"Training F1-score: {metrics.f1_score:.4f}")
    
    return model, training_results, metrics, data_split

def evaluate_model_performance(model, data_split, selected_features, training_results):
    """Evaluate model performance with advanced detailed metrics"""
    
    logger.info("Evaluating model performance with advanced metrics...")
    
    # Import advanced evaluation components
    from models.evaluator import ModelEvaluator, DetailedModelMetrics
    
    # Initialize advanced evaluator with feature names
    evaluator = ModelEvaluator(feature_names=selected_features)
    
    # Perform comprehensive evaluation
    detailed_metrics = evaluator.evaluate_model(
        model=model,
        X_test=data_split.X_test,
        y_test=data_split.y_test,
        feature_names=selected_features
    )
    
    # Analyze confusion matrix in detail
    cm_analysis = evaluator.analyze_confusion_matrix(detailed_metrics.confusion_matrix)
    
    # Analyze feature importance in detail
    feature_analysis = evaluator.analyze_feature_importance(
        model=model,
        feature_names=selected_features,
        top_n=15
    )
    
    # Generate comprehensive evaluation report
    evaluation_report = evaluator.generate_evaluation_report(
        metrics=detailed_metrics,
        cm_analysis=cm_analysis,
        feature_analysis=feature_analysis
    )
    
    # Log detailed results
    logger.info("Advanced Model Evaluation Results:")
    logger.info(f"  Accuracy:           {detailed_metrics.accuracy:.4f}")
    logger.info(f"  Precision:          {detailed_metrics.precision:.4f}")
    logger.info(f"  Recall:             {detailed_metrics.recall:.4f}")
    logger.info(f"  F1-Score:           {detailed_metrics.f1_score:.4f}")
    logger.info(f"  ROC-AUC:            {detailed_metrics.roc_auc:.4f}")
    logger.info(f"  Average Precision:  {detailed_metrics.average_precision:.4f}")
    
    logger.info("Confusion Matrix Analysis:")
    logger.info(f"  True Positives:     {cm_analysis.true_positives}")
    logger.info(f"  True Negatives:     {cm_analysis.true_negatives}")
    logger.info(f"  False Positives:    {cm_analysis.false_positives}")
    logger.info(f"  False Negatives:    {cm_analysis.false_negatives}")
    logger.info(f"  Sensitivity (TPR):  {cm_analysis.sensitivity:.4f}")
    logger.info(f"  Specificity (TNR):  {cm_analysis.specificity:.4f}")
    
    logger.info("Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(feature_analysis.ranked_features[:10], 1):
        logger.info(f"  {i:2d}. {feature:<25} {importance:.4f}")
    
    # Save evaluation plots
    try:
        # Plot and save confusion matrix
        cm_fig = evaluator.plot_confusion_matrix(
            detailed_metrics.confusion_matrix,
            class_names=['No Trade', 'Trade'],
            title="EMA Trap Strategy - Confusion Matrix"
        )
        cm_fig.savefig('ema_trap_confusion_matrix.png', dpi=300, bbox_inches='tight')
        logger.info("Confusion matrix plot saved to 'ema_trap_confusion_matrix.png'")
        
        # Plot and save feature importance
        fi_fig = evaluator.plot_feature_importance(
            feature_analysis,
            top_n=15,
            title="EMA Trap Strategy - Feature Importance"
        )
        fi_fig.savefig('ema_trap_feature_importance.png', dpi=300, bbox_inches='tight')
        logger.info("Feature importance plot saved to 'ema_trap_feature_importance.png'")
        
    except Exception as e:
        logger.warning(f"Could not save plots: {e}")
    
    # Save detailed evaluation report
    with open('ema_trap_evaluation_report.txt', 'w') as f:
        f.write(evaluation_report)
    logger.info("Detailed evaluation report saved to 'ema_trap_evaluation_report.txt'")
    
    return {
        'detailed_metrics': detailed_metrics,
        'confusion_matrix_analysis': cm_analysis,
        'feature_importance_analysis': feature_analysis,
        'evaluation_report': evaluation_report
    }

def save_model_and_results(model, training_results, evaluation_results, selected_features, trainer):
    """Save the trained model and results using advanced persistence system"""
    
    logger.info("Saving model and results with advanced persistence...")
    
    # Prepare comprehensive metadata
    training_metadata = {
        'strategy_name': 'EMA Trap Strategy',
        'algorithm': 'XGBoost',
        'data_source': 'Reliance 5-minute OHLCV data',
        'training_date': datetime.now().isoformat(),
        'target_config': {
            'method': 'ema_trap',
            'lookahead_periods': 2,
            'profit_threshold': 0.004,
            'loss_threshold': -0.004
        },
        'strategy_rules': {
            'ema_period': 21,
            'adx_period': 14,
            'adx_range': [20, 36],
            'entry_windows': ['9:15-9:30', '10:00-11:00'],
            'max_candle_body_pct': 0.20
        },
        'data_split_info': training_results.get('data_split_info', {}),
        'evaluation_summary': {
            'accuracy': evaluation_results['detailed_metrics'].accuracy,
            'precision': evaluation_results['detailed_metrics'].precision,
            'recall': evaluation_results['detailed_metrics'].recall,
            'f1_score': evaluation_results['detailed_metrics'].f1_score,
            'roc_auc': evaluation_results['detailed_metrics'].roc_auc
        }
    }
    
    # Feature configuration
    feature_config = {
        'selected_features': selected_features,
        'feature_groups': {
            'ema_trap_features': [
                'EMA_21', 'ADX', 'Distance_From_EMA21_Pct',
                'Bearish_Trap_Confirmed', 'Bullish_Trap_Confirmed',
                'In_Entry_Window', 'ADX_In_Range', 'Candle_Body_Size_Pct'
            ],
            'technical_indicators': [
                'RSI', 'MACD', 'MACD_Signal', 'BB_Position', 'ATR'
            ],
            'time_features': [
                'Hour', 'Minute', 'Entry_Window_1', 'Entry_Window_2'
            ],
            'candle_features': [
                'Green_Candle', 'Red_Candle', 'Upper_Shadow_Pct', 'Lower_Shadow_Pct'
            ]
        },
        'feature_engineering_config': {
            'ema_periods': [12, 21, 26, 50],
            'sma_periods': [20, 50, 200],
            'rsi_period': 14,
            'adx_period': 14,
            'bollinger_bands_period': 20
        }
    }
    
    # Convert evaluation metrics to the format expected by persistence system
    performance_metrics = evaluation_results['detailed_metrics']
    
    # Use the trainer's comprehensive save method
    model_path = trainer.save_model_comprehensive(
        model=model,
        model_name='ema_trap_xgboost_v1',
        feature_config=feature_config,
        performance_metrics=performance_metrics,
        training_metadata=training_metadata
    )
    
    logger.info(f"Model saved comprehensively to: {model_path}")
    
    # Also save additional analysis results
    analysis_results = {
        'confusion_matrix_analysis': evaluation_results['confusion_matrix_analysis'].__dict__,
        'feature_importance_analysis': {
            'ranked_features': evaluation_results['feature_importance_analysis'].ranked_features,
            'top_features': evaluation_results['feature_importance_analysis'].top_features
        },
        'training_results': training_results,
        'model_path': model_path
    }
    
    # Save analysis results as JSON
    import json
    with open('ema_trap_analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    logger.info("Additional analysis results saved to 'ema_trap_analysis_results.json'")
    
    return model_path

def analyze_entry_signals(targets_df):
    """Analyze the entry signals generated by the strategy"""
    
    logger.info("Analyzing entry signals...")
    
    # Count entry signals
    bearish_signals = targets_df['bearish_entry_signal'].sum()
    bullish_signals = targets_df['bullish_entry_signal'].sum()
    total_signals = targets_df['any_entry_signal'].sum()
    
    logger.info(f"Entry Signals Analysis:")
    logger.info(f"  Bearish signals: {bearish_signals}")
    logger.info(f"  Bullish signals: {bullish_signals}")
    logger.info(f"  Total signals: {total_signals}")
    
    if total_signals > 0:
        # Analyze signal distribution by time
        entry_signals = targets_df[targets_df['any_entry_signal'] == 1]
        
        # By hour
        hourly_dist = entry_signals['Hour'].value_counts().sort_index()
        logger.info(f"  Signals by hour: {hourly_dist.to_dict()}")
        
        # By entry window
        window1_signals = entry_signals['Entry_Window_1'].sum()
        window2_signals = entry_signals['Entry_Window_2'].sum()
        logger.info(f"  Window 1 (9:15-9:30): {window1_signals}")
        logger.info(f"  Window 2 (10:00-11:00): {window2_signals}")
        
        # Success rate analysis
        if 'future_return' in entry_signals.columns:
            profitable_signals = ((entry_signals['future_return'] >= 0.004) | 
                                (entry_signals['future_return'] <= -0.004)).sum()
            success_rate = profitable_signals / len(entry_signals) * 100
            logger.info(f"  Success rate: {success_rate:.2f}%")
    
    return {
        'bearish_signals': bearish_signals,
        'bullish_signals': bullish_signals,
        'total_signals': total_signals
    }

def main():
    """Main training pipeline"""
    
    print("EMA Trap Strategy - Model Training Pipeline")
    print("=" * 60)
    
    try:
        # 1. Load data
        print("\n1. Loading Reliance 5-minute data...")
        raw_data = load_reliance_data()
        
        # 2. Create features
        print("\n2. Creating EMA trap features...")
        features_df = create_ema_trap_features(raw_data)
        
        # 3. Generate targets
        print("\n3. Generating trading targets...")
        targets_df = generate_ema_trap_targets(features_df)
        
        # 4. Analyze entry signals
        print("\n4. Analyzing entry signals...")
        signal_analysis = analyze_entry_signals(targets_df)
        
        # 5. Select features
        print("\n5. Selecting features for training...")
        selected_features = select_features_for_training(targets_df)
        
        # 6. Prepare training data
        print("\n6. Preparing training data...")
        X, y = prepare_training_data(targets_df, selected_features)
        
        # Check if we have enough positive samples
        positive_samples = y.sum()
        if positive_samples < 50:
            logger.warning(f"Very few positive samples ({positive_samples}). Model may not train effectively.")
            print(f"⚠️  Warning: Only {positive_samples} positive samples found.")
            print("   Consider relaxing the strategy conditions or using more data.")
        
        # 7. Train model with advanced features
        print("\n7. Training XGBoost model with advanced pipeline...")
        model, training_results, training_metrics, data_split = train_ema_trap_model(X, y, selected_features)
        
        # 8. Evaluate model with advanced metrics
        print("\n8. Evaluating model performance with advanced metrics...")
        evaluation_results = evaluate_model_performance(model, data_split, selected_features, training_results)
        
        # 9. Save model with advanced persistence
        print("\n9. Saving model and results with advanced persistence...")
        # We need to get the trainer instance for comprehensive saving
        from models.trainer import ModelTrainer, TrainingConfig, HyperparameterGrid
        from models.persistence import PersistenceConfig
        
        # Recreate trainer for saving (in a real scenario, we'd pass it from training function)
        persistence_config = PersistenceConfig(
            base_path="models/ema_trap",
            compress_model=True,
            backup_existing=True
        )
        trainer = ModelTrainer(persistence_config=persistence_config)
        trainer.feature_names = selected_features
        trainer.best_params = training_results.get('best_params', {})
        
        model_path = save_model_and_results(model, training_results, evaluation_results, selected_features, trainer)
        
        # 10. Summary
        print("\n" + "=" * 60)
        print("✅ EMA Trap Model Training Complete!")
        print(f"📊 Dataset: {len(targets_df)} samples")
        print(f"🎯 Entry signals: {signal_analysis['total_signals']}")
        print(f"🤖 Model saved: {model_path}")
        print(f"📈 Test accuracy: {evaluation_results['detailed_metrics'].accuracy:.4f}")
        print(f"📈 Test F1-score: {evaluation_results['detailed_metrics'].f1_score:.4f}")
        print(f"📈 Test ROC-AUC: {evaluation_results['detailed_metrics'].roc_auc:.4f}")
        print(f"📊 Confusion matrix plots saved")
        print(f"📊 Feature importance plots saved")
        print(f"📄 Detailed evaluation report saved")
        print("=" * 60)
        
        return model, evaluation_results
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    model, results = main()