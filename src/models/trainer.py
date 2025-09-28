"""
XGBoost model training with hyperparameter optimization and time-series validation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import logging
from datetime import datetime
import pickle
import os

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

from ..interfaces import IModelTrainer, ModelMetrics
from ..exceptions import TradingSystemError
from .data_splitter import DataSplit
from .persistence import ModelPersistence, PersistenceConfig


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    algorithm: str = "xgboost"
    objective: str = "binary:logistic"
    test_size: float = 0.2
    cv_splits: int = 5
    early_stopping_rounds: Optional[int] = 10
    eval_metric: str = "logloss"
    random_state: int = 42
    n_jobs: int = -1
    verbose: bool = False


@dataclass
class HyperparameterGrid:
    """Hyperparameter search grid for XGBoost"""
    max_depth: List[int] = None
    learning_rate: List[float] = None
    n_estimators: List[int] = None
    gamma: List[float] = None
    subsample: List[float] = None
    colsample_bytree: List[float] = None
    reg_alpha: List[float] = None
    reg_lambda: List[float] = None
    
    def __post_init__(self):
        """Set default values if not provided"""
        if self.max_depth is None:
            self.max_depth = [3, 5, 7]
        if self.learning_rate is None:
            self.learning_rate = [0.01, 0.1, 0.2]
        if self.n_estimators is None:
            self.n_estimators = [100, 200, 300]
        if self.gamma is None:
            self.gamma = [0, 0.1, 0.2]
        if self.subsample is None:
            self.subsample = [0.8, 0.9, 1.0]
        if self.colsample_bytree is None:
            self.colsample_bytree = [0.8, 0.9, 1.0]
        if self.reg_alpha is None:
            self.reg_alpha = [0, 0.1, 0.5]
        if self.reg_lambda is None:
            self.reg_lambda = [1, 1.5, 2]
    
    def to_dict(self) -> Dict[str, List]:
        """Convert to dictionary for GridSearchCV"""
        return {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'gamma': self.gamma,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda
        }


@dataclass
class TrainingResult:
    """Result of model training process"""
    model: XGBClassifier
    best_params: Dict[str, Any]
    best_score: float
    cv_results: Dict[str, Any]
    training_time: float
    feature_names: List[str]
    training_samples: int
    validation_samples: int


class ModelTrainingError(TradingSystemError):
    """Raised when model training fails"""
    pass


class ModelTrainer(IModelTrainer):
    """
    XGBoost model trainer with hyperparameter optimization and time-series validation
    """
    
    def __init__(self, 
                 config: Optional[TrainingConfig] = None,
                 hyperparameter_grid: Optional[HyperparameterGrid] = None,
                 persistence_config: Optional[PersistenceConfig] = None):
        """
        Initialize model trainer
        
        Args:
            config: Training configuration
            hyperparameter_grid: Hyperparameter search grid
            persistence_config: Model persistence configuration
        """
        self.config = config or TrainingConfig()
        self.hyperparameter_grid = hyperparameter_grid or HyperparameterGrid()
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.best_params = None
        self.feature_names = None
        
        # Initialize persistence system
        self.persistence = ModelPersistence(persistence_config)
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for model training by extracting features and target
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            Tuple of (features, target)
        """
        if data.empty:
            raise ModelTrainingError("Input data is empty")
            
        # Assume target column is 'target' and exclude non-feature columns
        exclude_columns = ['target', 'datetime', 'timestamp', 'symbol']
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        if not feature_columns:
            raise ModelTrainingError("No feature columns found in data")
            
        if 'target' not in data.columns:
            raise ModelTrainingError("Target column 'target' not found in data")
            
        X = data[feature_columns].values
        y = data['target'].values
        
        # Store feature names for later use
        self.feature_names = feature_columns
        
        # Check for NaN values
        if np.isnan(X).any():
            raise ModelTrainingError("NaN values found in features")
            
        if np.isnan(y).any():
            raise ModelTrainingError("NaN values found in target")
            
        # Validate target values for binary classification
        unique_targets = np.unique(y)
        if not all(target in [0, 1] for target in unique_targets):
            raise ModelTrainingError("Target values must be 0 or 1 for binary classification")
            
        self.logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y
        
    def create_base_model(self) -> XGBClassifier:
        """
        Create base XGBoost classifier with default parameters
        
        Returns:
            Initialized XGBClassifier
        """
        return XGBClassifier(
            objective=self.config.objective,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            eval_metric=self.config.eval_metric,
            verbosity=0 if not self.config.verbose else 1
        )
        
    def tune_hyperparameters(self, 
                           X_train: np.ndarray, 
                           y_train: np.ndarray) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using TimeSeriesSplit cross-validation
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Best hyperparameters found
        """
        self.logger.info("Starting hyperparameter tuning...")
        
        # Create base model
        base_model = self.create_base_model()
        
        # Create time series cross-validator
        tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)
        
        # Get hyperparameter grid
        param_grid = self.hyperparameter_grid.to_dict()
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=self.config.n_jobs,
            verbose=1 if self.config.verbose else 0,
            error_score='raise'
        )
        
        try:
            grid_search.fit(X_train, y_train)
        except Exception as e:
            raise ModelTrainingError(f"Hyperparameter tuning failed: {str(e)}")
            
        self.best_params = grid_search.best_params_
        
        self.logger.info(f"Best parameters found: {self.best_params}")
        self.logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self.best_params
        
    def train_model(self, 
                   X_train: np.ndarray, 
                   y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None,
                   y_val: Optional[np.ndarray] = None,
                   hyperparameters: Optional[Dict[str, Any]] = None) -> XGBClassifier:
        """
        Train XGBoost model with optimal hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional, for early stopping)
            y_val: Validation targets (optional, for early stopping)
            hyperparameters: Custom hyperparameters (optional)
            
        Returns:
            Trained XGBClassifier model
        """
        self.logger.info("Training XGBoost model...")
        
        # Use provided hyperparameters or tune them
        if hyperparameters is None:
            if self.best_params is None:
                hyperparameters = self.tune_hyperparameters(X_train, y_train)
            else:
                hyperparameters = self.best_params
        
        # Create model with best parameters
        model_params = {
            'objective': self.config.objective,
            'random_state': self.config.random_state,
            'n_jobs': self.config.n_jobs,
            'eval_metric': self.config.eval_metric,
            'verbosity': 0 if not self.config.verbose else 1,
            **hyperparameters
        }
        
        # Add early stopping if validation data is provided and early stopping is configured
        if X_val is not None and y_val is not None and self.config.early_stopping_rounds is not None:
            model_params['early_stopping_rounds'] = self.config.early_stopping_rounds
            
        self.model = XGBClassifier(**model_params)
        
        # Prepare evaluation set for early stopping
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            
        try:
            # Train the model
            if eval_set is not None:
                self.model.fit(X_train, y_train, eval_set=eval_set)
            else:
                self.model.fit(X_train, y_train)
        except Exception as e:
            raise ModelTrainingError(f"Model training failed: {str(e)}")
            
        self.logger.info("Model training completed successfully")
        
        return self.model
        
    def evaluate_model(self, 
                      model: XGBClassifier, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray) -> ModelMetrics:
        """
        Evaluate trained model performance
        
        Args:
            model: Trained XGBoost model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            ModelMetrics with evaluation results
        """
        if model is None:
            raise ModelTrainingError("Model is not trained")
            
        try:
            # Generate predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            # Get feature importance
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                if self.feature_names and len(self.feature_names) == len(model.feature_importances_):
                    feature_importance = dict(zip(
                        self.feature_names, 
                        model.feature_importances_.tolist()
                    ))
                else:
                    # Fallback to generic feature names
                    feature_names = [f'feature_{i}' for i in range(len(model.feature_importances_))]
                    feature_importance = dict(zip(
                        feature_names,
                        model.feature_importances_.tolist()
                    ))
                
            self.logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}, "
                           f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            return ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                confusion_matrix=cm,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            raise ModelTrainingError(f"Model evaluation failed: {str(e)}")
            
    def train_and_evaluate(self, data_split: DataSplit) -> Tuple[XGBClassifier, ModelMetrics]:
        """
        Complete training and evaluation pipeline
        
        Args:
            data_split: Split data from DataSplitter
            
        Returns:
            Tuple of (trained_model, evaluation_metrics)
        """
        start_time = datetime.now()
        
        # Set feature names from data split
        self.feature_names = data_split.feature_names
        
        # Train model
        model = self.train_model(
            data_split.X_train, 
            data_split.y_train,
            data_split.X_test,  # Use test set for early stopping validation
            data_split.y_test
        )
        
        # Evaluate model
        metrics = self.evaluate_model(model, data_split.X_test, data_split.y_test)
        
        training_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Total training time: {training_time:.2f} seconds")
        
        return model, metrics
        
    def save_model_comprehensive(self,
                                model: XGBClassifier,
                                model_name: str,
                                feature_config: Dict[str, Any],
                                performance_metrics: Optional[ModelMetrics] = None,
                                training_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save complete model state using the persistence system
        
        Args:
            model: Trained model to save
            model_name: Name for the saved model
            feature_config: Feature engineering configuration
            performance_metrics: Model performance metrics
            training_metadata: Additional training metadata
            
        Returns:
            Path to the saved model directory
        """
        if not self.feature_names:
            raise ModelTrainingError("Feature names not available. Train model first.")
            
        if not self.best_params:
            raise ModelTrainingError("Hyperparameters not available. Train model first.")
            
        try:
            return self.persistence.save_model(
                model=model,
                model_name=model_name,
                feature_names=self.feature_names,
                hyperparameters=self.best_params,
                training_config=self.config.__dict__,
                feature_config=feature_config,
                performance_metrics=performance_metrics,
                training_metadata=training_metadata
            )
        except Exception as e:
            raise ModelTrainingError(f"Failed to save model comprehensively: {str(e)}")
            
    def load_model_comprehensive(self, model_name: str) -> Tuple[XGBClassifier, Dict[str, Any], Dict[str, Any]]:
        """
        Load complete model state using the persistence system
        
        Args:
            model_name: Name of the saved model
            
        Returns:
            Tuple of (model, model_state_dict, metadata)
        """
        try:
            model, model_state, metadata = self.persistence.load_model(model_name)
            
            # Restore trainer state
            self.feature_names = model_state.feature_names
            self.best_params = model_state.hyperparameters
            self.model = model
            
            return model, model_state.__dict__, metadata
        except Exception as e:
            raise ModelTrainingError(f"Failed to load model comprehensively: {str(e)}")
            
    def list_saved_models(self) -> List[Dict[str, Any]]:
        """
        List all saved models
        
        Returns:
            List of model information dictionaries
        """
        try:
            return self.persistence.list_saved_models()
        except Exception as e:
            raise ModelTrainingError(f"Failed to list saved models: {str(e)}")
            
    def delete_saved_model(self, model_name: str, confirm: bool = False) -> bool:
        """
        Delete a saved model
        
        Args:
            model_name: Name of the model to delete
            confirm: Confirmation flag
            
        Returns:
            True if deletion was successful
        """
        try:
            return self.persistence.delete_model(model_name, confirm=confirm)
        except Exception as e:
            raise ModelTrainingError(f"Failed to delete model: {str(e)}")
            
    def export_saved_model(self, model_name: str, export_path: str) -> str:
        """
        Export a saved model to a different location
        
        Args:
            model_name: Name of the model to export
            export_path: Path where to export the model
            
        Returns:
            Path to the exported model
        """
        try:
            return self.persistence.export_model(model_name, export_path)
        except Exception as e:
            raise ModelTrainingError(f"Failed to export model: {str(e)}")
    
    def save_model(self, 
                   model: XGBClassifier, 
                   filepath: str,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save trained model and metadata to disk (legacy method)
        
        Args:
            model: Trained model to save
            filepath: Path to save the model
            metadata: Additional metadata to save with model
        """
        try:
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(filepath)
            if dir_path:  # Only create directory if there is one
                os.makedirs(dir_path, exist_ok=True)
            
            # Save model
            joblib.dump(model, filepath)
            
            # Save metadata
            if metadata or self.feature_names or self.best_params:
                metadata_dict = metadata or {}
                metadata_dict.update({
                    'feature_names': self.feature_names,
                    'best_params': self.best_params,
                    'config': self.config.__dict__,
                    'saved_at': datetime.now().isoformat()
                })
                
                metadata_path = filepath.replace('.pkl', '_metadata.pkl')
                with open(metadata_path, 'wb') as f:
                    pickle.dump(metadata_dict, f)
                    
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to save model: {str(e)}")
            
    def load_model(self, filepath: str) -> Tuple[XGBClassifier, Dict[str, Any]]:
        """
        Load trained model and metadata from disk (legacy method)
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Tuple of (loaded_model, metadata)
        """
        try:
            # Load model
            model = joblib.load(filepath)
            
            # Load metadata
            metadata = {}
            metadata_path = filepath.replace('.pkl', '_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    
                # Restore feature names and best params
                self.feature_names = metadata.get('feature_names')
                self.best_params = metadata.get('best_params')
                
            self.logger.info(f"Model loaded from {filepath}")
            
            return model, metadata
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to load model: {str(e)}")