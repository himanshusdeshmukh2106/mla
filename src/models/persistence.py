"""
Model persistence system for XGBoost trading models
Handles serialization, configuration saving, and state restoration
"""

import os
import pickle
import json
import joblib
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import hashlib

from xgboost import XGBClassifier
import pandas as pd
import numpy as np

from ..interfaces import ModelMetrics
from ..exceptions import TradingSystemError


@dataclass
class ModelState:
    """Complete model state for persistence"""
    model_type: str
    model_version: str
    feature_names: List[str]
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    feature_config: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    training_metadata: Dict[str, Any]
    created_at: str
    model_hash: str


@dataclass
class PersistenceConfig:
    """Configuration for model persistence"""
    base_path: str = "models"
    model_filename: str = "model.pkl"
    state_filename: str = "model_state.json"
    metadata_filename: str = "metadata.pkl"
    compress_model: bool = True
    backup_existing: bool = True
    validate_on_load: bool = True


class ModelPersistenceError(TradingSystemError):
    """Raised when model persistence operations fail"""
    pass


class ModelPersistence:
    """
    Comprehensive model persistence system for XGBoost trading models
    """
    
    def __init__(self, config: Optional[PersistenceConfig] = None):
        """
        Initialize model persistence system
        
        Args:
            config: Persistence configuration
        """
        self.config = config or PersistenceConfig()
        self.logger = logging.getLogger(__name__)
        
        # Ensure base directory exists
        os.makedirs(self.config.base_path, exist_ok=True)
        
    def save_model(self,
                   model: XGBClassifier,
                   model_name: str,
                   feature_names: List[str],
                   hyperparameters: Dict[str, Any],
                   training_config: Dict[str, Any],
                   feature_config: Dict[str, Any],
                   performance_metrics: Optional[ModelMetrics] = None,
                   training_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save complete model state including model, configuration, and metadata
        
        Args:
            model: Trained XGBoost model
            model_name: Name for the saved model
            feature_names: List of feature names used in training
            hyperparameters: Model hyperparameters
            training_config: Training configuration
            feature_config: Feature engineering configuration
            performance_metrics: Model performance metrics
            training_metadata: Additional training metadata
            
        Returns:
            Path to the saved model directory
        """
        if model is None:
            raise ModelPersistenceError("Model cannot be None")
            
        if not feature_names:
            raise ModelPersistenceError("Feature names cannot be empty")
            
        try:
            self.logger.info(f"Saving model '{model_name}'...")
            
            # Create model directory
            model_dir = os.path.join(self.config.base_path, model_name)
            
            # Backup existing model if requested
            if self.config.backup_existing and os.path.exists(model_dir):
                self._backup_existing_model(model_dir)
                
            os.makedirs(model_dir, exist_ok=True)
            
            # Save the model
            model_path = os.path.join(model_dir, self.config.model_filename)
            self._save_model_file(model, model_path)
            
            # Prepare performance metrics for serialization
            metrics_dict = {}
            if performance_metrics:
                metrics_dict = {
                    'accuracy': performance_metrics.accuracy,
                    'precision': performance_metrics.precision,
                    'recall': performance_metrics.recall,
                    'f1_score': performance_metrics.f1_score,
                    'confusion_matrix': performance_metrics.confusion_matrix.tolist(),
                    'feature_importance': performance_metrics.feature_importance
                }
            
            # Create model state
            model_state = ModelState(
                model_type="XGBClassifier",
                model_version=self._get_model_version(model),
                feature_names=feature_names,
                hyperparameters=hyperparameters,
                training_config=training_config,
                feature_config=feature_config,
                performance_metrics=metrics_dict,
                training_metadata=training_metadata or {},
                created_at=datetime.now().isoformat(),
                model_hash=self._calculate_model_hash(model)
            )
            
            # Save model state as JSON
            state_path = os.path.join(model_dir, self.config.state_filename)
            self._save_model_state(model_state, state_path)
            
            # Save additional metadata as pickle for complex objects
            metadata_path = os.path.join(model_dir, self.config.metadata_filename)
            metadata = {
                'model_state': model_state,
                'raw_performance_metrics': performance_metrics,
                'save_timestamp': datetime.now(),
                'feature_names': feature_names,
                'hyperparameters': hyperparameters
            }
            self._save_metadata(metadata, metadata_path)
            
            self.logger.info(f"Model '{model_name}' saved successfully to {model_dir}")
            return model_dir
            
        except Exception as e:
            raise ModelPersistenceError(f"Failed to save model '{model_name}': {str(e)}")
            
    def load_model(self, model_name: str) -> Tuple[XGBClassifier, ModelState, Dict[str, Any]]:
        """
        Load complete model state including model, configuration, and metadata
        
        Args:
            model_name: Name of the saved model
            
        Returns:
            Tuple of (model, model_state, metadata)
        """
        try:
            self.logger.info(f"Loading model '{model_name}'...")
            
            model_dir = os.path.join(self.config.base_path, model_name)
            
            if not os.path.exists(model_dir):
                raise ModelPersistenceError(f"Model directory '{model_dir}' does not exist")
                
            # Load the model
            model_path = os.path.join(model_dir, self.config.model_filename)
            model = self._load_model_file(model_path)
            
            # Load model state
            state_path = os.path.join(model_dir, self.config.state_filename)
            model_state = self._load_model_state(state_path)
            
            # Load metadata
            metadata_path = os.path.join(model_dir, self.config.metadata_filename)
            metadata = self._load_metadata(metadata_path)
            
            # Validate model if requested
            if self.config.validate_on_load:
                self._validate_loaded_model(model, model_state)
                
            self.logger.info(f"Model '{model_name}' loaded successfully")
            return model, model_state, metadata
            
        except Exception as e:
            raise ModelPersistenceError(f"Failed to load model '{model_name}': {str(e)}")
            
    def list_saved_models(self) -> List[Dict[str, Any]]:
        """
        List all saved models with their metadata
        
        Returns:
            List of model information dictionaries
        """
        try:
            models = []
            
            if not os.path.exists(self.config.base_path):
                return models
                
            for item in os.listdir(self.config.base_path):
                model_dir = os.path.join(self.config.base_path, item)
                
                if not os.path.isdir(model_dir):
                    continue
                    
                state_path = os.path.join(model_dir, self.config.state_filename)
                
                if os.path.exists(state_path):
                    try:
                        model_state = self._load_model_state(state_path)
                        models.append({
                            'name': item,
                            'created_at': model_state.created_at,
                            'model_type': model_state.model_type,
                            'model_version': model_state.model_version,
                            'feature_count': len(model_state.feature_names),
                            'performance_metrics': model_state.performance_metrics,
                            'path': model_dir
                        })
                    except Exception as e:
                        self.logger.warning(f"Could not load state for model '{item}': {str(e)}")
                        
            # Sort by creation date (newest first)
            models.sort(key=lambda x: x['created_at'], reverse=True)
            
            self.logger.info(f"Found {len(models)} saved models")
            return models
            
        except Exception as e:
            raise ModelPersistenceError(f"Failed to list saved models: {str(e)}")
            
    def delete_model(self, model_name: str, confirm: bool = False) -> bool:
        """
        Delete a saved model and all its files
        
        Args:
            model_name: Name of the model to delete
            confirm: Confirmation flag to prevent accidental deletion
            
        Returns:
            True if deletion was successful
        """
        if not confirm:
            raise ModelPersistenceError("Model deletion requires explicit confirmation")
            
        try:
            model_dir = os.path.join(self.config.base_path, model_name)
            
            if not os.path.exists(model_dir):
                raise ModelPersistenceError(f"Model '{model_name}' does not exist")
                
            # Remove all files in the model directory
            import shutil
            shutil.rmtree(model_dir)
            
            self.logger.info(f"Model '{model_name}' deleted successfully")
            return True
            
        except Exception as e:
            raise ModelPersistenceError(f"Failed to delete model '{model_name}': {str(e)}")
            
    def export_model(self, model_name: str, export_path: str) -> str:
        """
        Export a saved model to a different location
        
        Args:
            model_name: Name of the model to export
            export_path: Path where to export the model
            
        Returns:
            Path to the exported model
        """
        try:
            model_dir = os.path.join(self.config.base_path, model_name)
            
            if not os.path.exists(model_dir):
                raise ModelPersistenceError(f"Model '{model_name}' does not exist")
                
            # Create export directory
            os.makedirs(export_path, exist_ok=True)
            
            # Copy all model files
            import shutil
            export_model_dir = os.path.join(export_path, model_name)
            shutil.copytree(model_dir, export_model_dir, dirs_exist_ok=True)
            
            self.logger.info(f"Model '{model_name}' exported to {export_model_dir}")
            return export_model_dir
            
        except Exception as e:
            raise ModelPersistenceError(f"Failed to export model '{model_name}': {str(e)}")
            
    def _save_model_file(self, model: XGBClassifier, filepath: str) -> None:
        """Save the XGBoost model to file"""
        try:
            if self.config.compress_model:
                joblib.dump(model, filepath, compress=3)
            else:
                joblib.dump(model, filepath)
        except Exception as e:
            raise ModelPersistenceError(f"Failed to save model file: {str(e)}")
            
    def _load_model_file(self, filepath: str) -> XGBClassifier:
        """Load the XGBoost model from file"""
        try:
            if not os.path.exists(filepath):
                raise ModelPersistenceError(f"Model file '{filepath}' does not exist")
            return joblib.load(filepath)
        except Exception as e:
            raise ModelPersistenceError(f"Failed to load model file: {str(e)}")
            
    def _save_model_state(self, model_state: ModelState, filepath: str) -> None:
        """Save model state as JSON"""
        try:
            with open(filepath, 'w') as f:
                json.dump(asdict(model_state), f, indent=2, default=str)
        except Exception as e:
            raise ModelPersistenceError(f"Failed to save model state: {str(e)}")
            
    def _load_model_state(self, filepath: str) -> ModelState:
        """Load model state from JSON"""
        try:
            if not os.path.exists(filepath):
                raise ModelPersistenceError(f"Model state file '{filepath}' does not exist")
                
            with open(filepath, 'r') as f:
                state_dict = json.load(f)
                
            return ModelState(**state_dict)
        except Exception as e:
            raise ModelPersistenceError(f"Failed to load model state: {str(e)}")
            
    def _save_metadata(self, metadata: Dict[str, Any], filepath: str) -> None:
        """Save metadata as pickle"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(metadata, f)
        except Exception as e:
            raise ModelPersistenceError(f"Failed to save metadata: {str(e)}")
            
    def _load_metadata(self, filepath: str) -> Dict[str, Any]:
        """Load metadata from pickle"""
        try:
            if not os.path.exists(filepath):
                return {}  # Return empty dict if metadata doesn't exist
                
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load metadata: {str(e)}")
            return {}
            
    def _backup_existing_model(self, model_dir: str) -> None:
        """Create backup of existing model"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{model_dir}_backup_{timestamp}"
            
            import shutil
            shutil.copytree(model_dir, backup_dir)
            
            self.logger.info(f"Created backup at {backup_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to create backup: {str(e)}")
            
    def _get_model_version(self, model: XGBClassifier) -> str:
        """Get XGBoost version information"""
        try:
            import xgboost
            return xgboost.__version__
        except:
            return "unknown"
            
    def _calculate_model_hash(self, model: XGBClassifier) -> str:
        """Calculate hash of the model for integrity checking"""
        try:
            # Use model parameters and feature importances for hash
            model_data = {
                'params': model.get_params(),
                'n_features': model.n_features_in_ if hasattr(model, 'n_features_in_') else 0
            }
            
            if hasattr(model, 'feature_importances_'):
                model_data['feature_importances'] = model.feature_importances_.tolist()
                
            model_str = json.dumps(model_data, sort_keys=True, default=str)
            return hashlib.md5(model_str.encode()).hexdigest()
        except Exception:
            return "unknown"
            
    def _validate_loaded_model(self, model: XGBClassifier, model_state: ModelState) -> None:
        """Validate loaded model against saved state"""
        try:
            # Check model type
            if not isinstance(model, XGBClassifier):
                raise ModelPersistenceError("Loaded model is not an XGBClassifier")
                
            # Check feature count
            if hasattr(model, 'n_features_in_'):
                if model.n_features_in_ != len(model_state.feature_names):
                    raise ModelPersistenceError(
                        f"Feature count mismatch: model has {model.n_features_in_}, "
                        f"state has {len(model_state.feature_names)}"
                    )
                    
            # Check model hash if available
            current_hash = self._calculate_model_hash(model)
            if model_state.model_hash != "unknown" and current_hash != "unknown":
                if current_hash != model_state.model_hash:
                    self.logger.warning("Model hash mismatch - model may have been modified")
                    
            self.logger.info("Model validation passed")
            
        except Exception as e:
            raise ModelPersistenceError(f"Model validation failed: {str(e)}")