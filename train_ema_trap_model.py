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
    
    # Add current directory and src to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    src_dir = os.path.join(current_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    print(f"📁 Added to Python path: {current_dir}")
    print(f"📁 Added to Python path: {src_dir}")

# Setup environment
setup_environment()

try:
    # Try importing from src package structure
    from src.features.engineer import FeatureEngineer
    from src.features.target_generator import TargetGenerator
    from src.models.trainer import ModelTrainer
    from src.models.evaluator import ModelEvaluator
    from src.models.persistence import ModelPersistence
    from src.config_manager import ConfigManager
    from src.logger import get_logger
except ImportError:
    # Fallback to direct imports
    try:
        from features.engineer import FeatureEngineer
        from features.target_generator import TargetGenerator
        from models.trainer import ModelTrainer
        from models.evaluator import ModelEvaluator
        from models.persistence import ModelPersistence
        from config_manager import ConfigManager
        from logger import get_logger
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Trying alternative import method...")
        
        # Add more paths and try again
        import importlib.util
        
        def import_from_path(module_name, file_path):
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        
        # Import modules directly from file paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(current_dir, 'src')
        
        try:
            # Import required modules
            engineer_module = import_from_path("engineer", os.path.join(src_dir, "features", "engineer.py"))
            FeatureEngineer = engineer_module.FeatureEngineer
            
            target_module = import_from_path("target_generator", os.path.join(src_dir, "features", "target_generator.py"))
            TargetGenerator = target_module.TargetGenerator
            
            print("✅ Successfully imported modules using alternative method")
        except Exception as e2:
            print(f"❌ Failed to import modules: {e2}")
            print("💡 Please ensure you're running from the project root directory")
            sys.exit(1)

# Simple logger fallback
try:
    logger = get_logger(__name__)
except:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

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
    """Train XGBoost model for EMA trap strategy using simplified approach"""
    
    print("Training EMA trap XGBoost model...")
    
    # Import XGBoost and sklearn components directly
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Split data (time-series aware)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Hyperparameter grid for EMA trap strategy
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'n_estimators': [200, 300, 500],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }
    
    # Base XGBoost model
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)  # Reduced for faster training
    
    # Grid search with cross-validation
    print("Performing hyperparameter optimization...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Best parameters: {best_params}")
    print(f"Best CV F1-score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
    }
    
    # Feature importance
    feature_importance = {}
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = dict(zip(selected_features, best_model.feature_importances_))
    
    # Training results
    training_results = {
        'best_params': best_params,
        'best_cv_score': grid_search.best_score_,
        'feature_names': selected_features,
        'feature_importance': feature_importance,
        'data_split_info': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(selected_features)
        }
    }
    
    # Create simple data split object
    class SimpleDataSplit:
        def __init__(self, X_train, X_test, y_train, y_test):
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
    
    data_split = SimpleDataSplit(X_train, X_test, y_train, y_test)
    
    print("Model training completed")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1-score: {metrics['f1_score']:.4f}")
    
    return best_model, training_results, metrics, data_split

def evaluate_model_performance(model, data_split, selected_features, training_results):
    """Evaluate model performance with simplified metrics"""
    
    print("Evaluating model performance...")
    
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get predictions
    y_pred = model.predict(data_split.X_test)
    y_pred_proba = model.predict_proba(data_split.X_test)[:, 1]
    
    # Calculate metrics (already done in training, but let's be explicit)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(data_split.y_test, y_pred),
        'precision': precision_score(data_split.y_test, y_pred, zero_division=0),
        'recall': recall_score(data_split.y_test, y_pred, zero_division=0),
        'f1_score': f1_score(data_split.y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(data_split.y_test, y_pred_proba) if len(np.unique(data_split.y_test)) > 1 else 0.0
    }
    
    # Confusion matrix
    cm = confusion_matrix(data_split.y_test, y_pred)
    
    # Feature importance
    feature_importance = training_results.get('feature_importance', {})
    
    # Log results
    print("Model Evaluation Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    if feature_importance:
        print("\nTop 10 Most Important Features:")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"  {i:2d}. {feature:<25} {importance:.4f}")
    
    # Create simple plots
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion matrix plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=['No Trade', 'Trade'],
                   yticklabels=['No Trade', 'Trade'])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Feature importance plot
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
            features, importances = zip(*top_features)
            
            y_pos = np.arange(len(features))
            axes[1].barh(y_pos, importances, color='lightgreen')
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels(features)
            axes[1].invert_yaxis()
            axes[1].set_xlabel('Importance Score')
            axes[1].set_title('Top 15 Feature Importance')
        
        plt.tight_layout()
        plt.savefig('ema_trap_evaluation_plots.png', dpi=300, bbox_inches='tight')
        print("Evaluation plots saved to 'ema_trap_evaluation_plots.png'")
        plt.show()
        
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    # Save classification report
    try:
        report = classification_report(data_split.y_test, y_pred, zero_division=0)
        with open('ema_trap_classification_report.txt', 'w') as f:
            f.write("EMA Trap Strategy - Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
            f.write(f"\n\nMetrics Summary:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        
        print("Classification report saved to 'ema_trap_classification_report.txt'")
    except Exception as e:
        print(f"Could not save report: {e}")
    
    return {
        'metrics': metrics,
        'confusion_matrix': cm,
        'feature_importance': feature_importance,
        'predictions': y_pred,
        'prediction_probabilities': y_pred_proba
    }

def save_model_and_results(model, training_results, evaluation_results, selected_features):
    """Save the trained model and results using simple approach"""
    
    print("Saving model and results...")
    
    import joblib
    import json
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model_path = 'models/ema_trap_xgboost_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Prepare comprehensive metadata
    metadata = {
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
        'selected_features': selected_features,
        'training_results': training_results,
        'evaluation_metrics': evaluation_results['metrics'],
        'model_path': model_path
    }
    
    # Save metadata as JSON
    metadata_path = 'models/ema_trap_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"Metadata saved to: {metadata_path}")
    
    # Save feature importance separately
    if evaluation_results.get('feature_importance'):
        feature_importance_path = 'models/ema_trap_feature_importance.json'
        with open(feature_importance_path, 'w') as f:
            json.dump(evaluation_results['feature_importance'], f, indent=2)
        print(f"Feature importance saved to: {feature_importance_path}")
    
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
        
        # 7. Train model
        print("\n7. Training XGBoost model...")
        model, training_results, training_metrics, data_split = train_ema_trap_model(X, y, selected_features)
        
        # 8. Evaluate model
        print("\n8. Evaluating model performance...")
        evaluation_results = evaluate_model_performance(model, data_split, selected_features, training_results)
        
        # 9. Save model
        print("\n9. Saving model and results...")
        model_path = save_model_and_results(model, training_results, evaluation_results, selected_features)
        
        # 10. Summary
        print("\n" + "=" * 60)
        print("✅ EMA Trap Model Training Complete!")
        print(f"📊 Dataset: {len(targets_df)} samples")
        print(f"🎯 Entry signals: {signal_analysis['total_signals']}")
        print(f"🤖 Model saved: {model_path}")
        print(f"📈 Test accuracy: {evaluation_results['metrics']['accuracy']:.4f}")
        print(f"📈 Test F1-score: {evaluation_results['metrics']['f1_score']:.4f}")
        print(f"📈 Test ROC-AUC: {evaluation_results['metrics']['roc_auc']:.4f}")
        print(f"📊 Evaluation plots saved")
        print(f"📄 Classification report saved")
        print("=" * 60)
        
        return model, evaluation_results
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    model, results = main()