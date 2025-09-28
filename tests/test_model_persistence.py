"""
Unit tests for model persistence system
"""

import pytest
import os
import tempfile
import shutil
import json
import pickle
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import pandas as pd
import numpy as np
from xgboost import XGBClassifier

from src.models.persistence import (
    ModelPersistence, ModelState, PersistenceConfig, ModelPersistenceError
)
from src.interfaces import ModelMetrics


class TestModelPersistence:
    """Test cases for ModelPersistence class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    @pytest.fixture
    def persistence_config(self, temp_dir):
        """Create test persistence configuration"""
        return PersistenceConfig(
            base_path=temp_dir,
            model_filename="test_model.pkl",
            state_filename="test_state.json",
            metadata_filename="test_metadata.pkl",
            compress_model=False,
            backup_existing=False,
            validate_on_load=True
        )
        
    @pytest.fixture
    def model_persistence(self, persistence_config):
        """Create ModelPersistence instance for testing"""
        return ModelPersistence(persistence_config)
        
    @pytest.fixture
    def sample_model(self):
        """Create sample XGBoost model for testing"""
        model = XGBClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42
        )
        
        # Create sample training data
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Train the model
        model.fit(X, y)
        
        return model
        
    @pytest.fixture
    def sample_feature_names(self):
        """Sample feature names"""
        return ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        
    @pytest.fixture
    def sample_hyperparameters(self):
        """Sample hyperparameters"""
        return {
            'n_estimators': 10,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
    @pytest.fixture
    def sample_training_config(self):
        """Sample training configuration"""
        return {
            'test_size': 0.2,
            'cv_splits': 5,
            'early_stopping_rounds': 10
        }
        
    @pytest.fixture
    def sample_feature_config(self):
        """Sample feature configuration"""
        return {
            'trend_periods': [20, 50],
            'momentum_periods': {'rsi': 14, 'macd_fast': 12},
            'volatility_periods': {'bb': 20, 'atr': 14}
        }
        
    @pytest.fixture
    def sample_performance_metrics(self):
        """Sample performance metrics"""
        return ModelMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            confusion_matrix=np.array([[45, 5], [7, 43]]),
            feature_importance={'feature_1': 0.3, 'feature_2': 0.25, 'feature_3': 0.2, 'feature_4': 0.15, 'feature_5': 0.1}
        )
        
    def test_init_creates_base_directory(self, temp_dir):
        """Test that initialization creates base directory"""
        config = PersistenceConfig(base_path=os.path.join(temp_dir, "new_models"))
        persistence = ModelPersistence(config)
        
        assert os.path.exists(config.base_path)
        
    def test_save_model_success(self, model_persistence, sample_model, sample_feature_names,
                               sample_hyperparameters, sample_training_config, 
                               sample_feature_config, sample_performance_metrics):
        """Test successful model saving"""
        model_name = "test_model"
        
        result_path = model_persistence.save_model(
            model=sample_model,
            model_name=model_name,
            feature_names=sample_feature_names,
            hyperparameters=sample_hyperparameters,
            training_config=sample_training_config,
            feature_config=sample_feature_config,
            performance_metrics=sample_performance_metrics
        )
        
        # Check that model directory was created
        expected_path = os.path.join(model_persistence.config.base_path, model_name)
        assert result_path == expected_path
        assert os.path.exists(expected_path)
        
        # Check that all files were created
        model_file = os.path.join(expected_path, model_persistence.config.model_filename)
        state_file = os.path.join(expected_path, model_persistence.config.state_filename)
        metadata_file = os.path.join(expected_path, model_persistence.config.metadata_filename)
        
        assert os.path.exists(model_file)
        assert os.path.exists(state_file)
        assert os.path.exists(metadata_file)
        
    def test_save_model_with_none_model_raises_error(self, model_persistence, sample_feature_names,
                                                    sample_hyperparameters, sample_training_config,
                                                    sample_feature_config):
        """Test that saving None model raises error"""
        with pytest.raises(ModelPersistenceError, match="Model cannot be None"):
            model_persistence.save_model(
                model=None,
                model_name="test_model",
                feature_names=sample_feature_names,
                hyperparameters=sample_hyperparameters,
                training_config=sample_training_config,
                feature_config=sample_feature_config
            )
            
    def test_save_model_with_empty_feature_names_raises_error(self, model_persistence, sample_model,
                                                             sample_hyperparameters, sample_training_config,
                                                             sample_feature_config):
        """Test that saving with empty feature names raises error"""
        with pytest.raises(ModelPersistenceError, match="Feature names cannot be empty"):
            model_persistence.save_model(
                model=sample_model,
                model_name="test_model",
                feature_names=[],
                hyperparameters=sample_hyperparameters,
                training_config=sample_training_config,
                feature_config=sample_feature_config
            )
            
    def test_load_model_success(self, model_persistence, sample_model, sample_feature_names,
                               sample_hyperparameters, sample_training_config,
                               sample_feature_config, sample_performance_metrics):
        """Test successful model loading"""
        model_name = "test_model"
        
        # First save the model
        model_persistence.save_model(
            model=sample_model,
            model_name=model_name,
            feature_names=sample_feature_names,
            hyperparameters=sample_hyperparameters,
            training_config=sample_training_config,
            feature_config=sample_feature_config,
            performance_metrics=sample_performance_metrics
        )
        
        # Then load it
        loaded_model, model_state, metadata = model_persistence.load_model(model_name)
        
        # Check that model was loaded correctly
        assert isinstance(loaded_model, XGBClassifier)
        assert isinstance(model_state, ModelState)
        assert isinstance(metadata, dict)
        
        # Check model state
        assert model_state.model_type == "XGBClassifier"
        assert model_state.feature_names == sample_feature_names
        assert model_state.hyperparameters == sample_hyperparameters
        assert model_state.training_config == sample_training_config
        assert model_state.feature_config == sample_feature_config
        
    def test_load_nonexistent_model_raises_error(self, model_persistence):
        """Test that loading nonexistent model raises error"""
        with pytest.raises(ModelPersistenceError, match="Model directory .* does not exist"):
            model_persistence.load_model("nonexistent_model")
            
    def test_list_saved_models_empty(self, model_persistence):
        """Test listing models when no models are saved"""
        models = model_persistence.list_saved_models()
        assert models == []
        
    def test_list_saved_models_with_models(self, model_persistence, sample_model, sample_feature_names,
                                          sample_hyperparameters, sample_training_config,
                                          sample_feature_config, sample_performance_metrics):
        """Test listing models when models are saved"""
        # Save multiple models
        model_names = ["model_1", "model_2", "model_3"]
        
        for model_name in model_names:
            model_persistence.save_model(
                model=sample_model,
                model_name=model_name,
                feature_names=sample_feature_names,
                hyperparameters=sample_hyperparameters,
                training_config=sample_training_config,
                feature_config=sample_feature_config,
                performance_metrics=sample_performance_metrics
            )
            
        models = model_persistence.list_saved_models()
        
        assert len(models) == 3
        saved_names = [model['name'] for model in models]
        assert all(name in saved_names for name in model_names)
        
        # Check model information
        for model_info in models:
            assert 'name' in model_info
            assert 'created_at' in model_info
            assert 'model_type' in model_info
            assert 'feature_count' in model_info
            assert model_info['model_type'] == "XGBClassifier"
            assert model_info['feature_count'] == len(sample_feature_names)
            
    def test_delete_model_success(self, model_persistence, sample_model, sample_feature_names,
                                 sample_hyperparameters, sample_training_config,
                                 sample_feature_config):
        """Test successful model deletion"""
        model_name = "test_model"
        
        # Save model first
        model_persistence.save_model(
            model=sample_model,
            model_name=model_name,
            feature_names=sample_feature_names,
            hyperparameters=sample_hyperparameters,
            training_config=sample_training_config,
            feature_config=sample_feature_config
        )
        
        # Verify model exists
        model_dir = os.path.join(model_persistence.config.base_path, model_name)
        assert os.path.exists(model_dir)
        
        # Delete model
        result = model_persistence.delete_model(model_name, confirm=True)
        
        assert result is True
        assert not os.path.exists(model_dir)
        
    def test_delete_model_without_confirmation_raises_error(self, model_persistence):
        """Test that deleting without confirmation raises error"""
        with pytest.raises(ModelPersistenceError, match="Model deletion requires explicit confirmation"):
            model_persistence.delete_model("test_model", confirm=False)
            
    def test_delete_nonexistent_model_raises_error(self, model_persistence):
        """Test that deleting nonexistent model raises error"""
        with pytest.raises(ModelPersistenceError, match="Model .* does not exist"):
            model_persistence.delete_model("nonexistent_model", confirm=True)
            
    def test_export_model_success(self, model_persistence, sample_model, sample_feature_names,
                                 sample_hyperparameters, sample_training_config,
                                 sample_feature_config, temp_dir):
        """Test successful model export"""
        model_name = "test_model"
        export_path = os.path.join(temp_dir, "exports")
        
        # Save model first
        model_persistence.save_model(
            model=sample_model,
            model_name=model_name,
            feature_names=sample_feature_names,
            hyperparameters=sample_hyperparameters,
            training_config=sample_training_config,
            feature_config=sample_feature_config
        )
        
        # Export model
        result_path = model_persistence.export_model(model_name, export_path)
        
        expected_path = os.path.join(export_path, model_name)
        assert result_path == expected_path
        assert os.path.exists(expected_path)
        
        # Check that all files were exported
        model_file = os.path.join(expected_path, model_persistence.config.model_filename)
        state_file = os.path.join(expected_path, model_persistence.config.state_filename)
        metadata_file = os.path.join(expected_path, model_persistence.config.metadata_filename)
        
        assert os.path.exists(model_file)
        assert os.path.exists(state_file)
        assert os.path.exists(metadata_file)
        
    def test_export_nonexistent_model_raises_error(self, model_persistence, temp_dir):
        """Test that exporting nonexistent model raises error"""
        export_path = os.path.join(temp_dir, "exports")
        
        with pytest.raises(ModelPersistenceError, match="Model .* does not exist"):
            model_persistence.export_model("nonexistent_model", export_path)
            
    def test_backup_existing_model(self, temp_dir):
        """Test backup functionality when saving over existing model"""
        config = PersistenceConfig(
            base_path=temp_dir,
            backup_existing=True
        )
        persistence = ModelPersistence(config)
        
        # Create sample model and data
        model = XGBClassifier(n_estimators=5, random_state=42)
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        feature_names = ['f1', 'f2', 'f3']
        hyperparameters = {'n_estimators': 5}
        training_config = {'test_size': 0.2}
        feature_config = {'indicators': ['sma']}
        
        model_name = "backup_test_model"
        
        # Save model first time
        persistence.save_model(
            model=model,
            model_name=model_name,
            feature_names=feature_names,
            hyperparameters=hyperparameters,
            training_config=training_config,
            feature_config=feature_config
        )
        
        # Save model second time (should create backup)
        persistence.save_model(
            model=model,
            model_name=model_name,
            feature_names=feature_names,
            hyperparameters=hyperparameters,
            training_config=training_config,
            feature_config=feature_config
        )
        
        # Check that backup was created
        backup_dirs = [d for d in os.listdir(temp_dir) if d.startswith(f"{model_name}_backup_")]
        assert len(backup_dirs) >= 1
        
    def test_model_validation_on_load(self, model_persistence, sample_model, sample_feature_names,
                                     sample_hyperparameters, sample_training_config,
                                     sample_feature_config):
        """Test model validation during loading"""
        model_name = "validation_test_model"
        
        # Save model
        model_persistence.save_model(
            model=sample_model,
            model_name=model_name,
            feature_names=sample_feature_names,
            hyperparameters=sample_hyperparameters,
            training_config=sample_training_config,
            feature_config=sample_feature_config
        )
        
        # Load model (validation should pass)
        loaded_model, model_state, metadata = model_persistence.load_model(model_name)
        
        assert isinstance(loaded_model, XGBClassifier)
        assert model_state.feature_names == sample_feature_names
        
    def test_model_state_serialization(self, temp_dir):
        """Test ModelState serialization and deserialization"""
        model_state = ModelState(
            model_type="XGBClassifier",
            model_version="1.0.0",
            feature_names=['f1', 'f2', 'f3'],
            hyperparameters={'n_estimators': 100},
            training_config={'test_size': 0.2},
            feature_config={'indicators': ['sma', 'rsi']},
            performance_metrics={'accuracy': 0.85},
            training_metadata={'duration': 120},
            created_at=datetime.now().isoformat(),
            model_hash="abc123"
        )
        
        # Test JSON serialization
        state_file = os.path.join(temp_dir, "test_state.json")
        
        with open(state_file, 'w') as f:
            json.dump(model_state.__dict__, f, indent=2, default=str)
            
        # Test deserialization
        with open(state_file, 'r') as f:
            loaded_dict = json.load(f)
            
        loaded_state = ModelState(**loaded_dict)
        
        assert loaded_state.model_type == model_state.model_type
        assert loaded_state.feature_names == model_state.feature_names
        assert loaded_state.hyperparameters == model_state.hyperparameters
        
    @patch('src.models.persistence.joblib.dump')
    def test_save_model_file_error_handling(self, mock_dump, model_persistence, sample_model):
        """Test error handling in model file saving"""
        mock_dump.side_effect = Exception("Disk full")
        
        with pytest.raises(ModelPersistenceError, match="Failed to save model file"):
            model_persistence._save_model_file(sample_model, "test_path.pkl")
            
    @patch('src.models.persistence.joblib.load')
    def test_load_model_file_error_handling(self, mock_load, model_persistence):
        """Test error handling in model file loading"""
        mock_load.side_effect = Exception("File corrupted")
        
        # Create a dummy file to pass the existence check
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            with pytest.raises(ModelPersistenceError, match="Failed to load model file"):
                model_persistence._load_model_file(tmp_path)
        finally:
            os.unlink(tmp_path)
            
    def test_calculate_model_hash(self, model_persistence, sample_model):
        """Test model hash calculation"""
        hash1 = model_persistence._calculate_model_hash(sample_model)
        hash2 = model_persistence._calculate_model_hash(sample_model)
        
        # Same model should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length
        
    def test_get_model_version(self, model_persistence):
        """Test model version extraction"""
        version = model_persistence._get_model_version(None)
        
        # Should return either actual version or "unknown"
        assert isinstance(version, str)
        assert len(version) > 0


class TestPersistenceConfig:
    """Test cases for PersistenceConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = PersistenceConfig()
        
        assert config.base_path == "models"
        assert config.model_filename == "model.pkl"
        assert config.state_filename == "model_state.json"
        assert config.metadata_filename == "metadata.pkl"
        assert config.compress_model is True
        assert config.backup_existing is True
        assert config.validate_on_load is True
        
    def test_custom_config(self):
        """Test custom configuration values"""
        config = PersistenceConfig(
            base_path="/custom/path",
            model_filename="custom_model.pkl",
            compress_model=False,
            backup_existing=False
        )
        
        assert config.base_path == "/custom/path"
        assert config.model_filename == "custom_model.pkl"
        assert config.compress_model is False
        assert config.backup_existing is False


class TestModelState:
    """Test cases for ModelState dataclass"""
    
    def test_model_state_creation(self):
        """Test ModelState creation with all fields"""
        state = ModelState(
            model_type="XGBClassifier",
            model_version="1.0.0",
            feature_names=['f1', 'f2'],
            hyperparameters={'n_estimators': 100},
            training_config={'test_size': 0.2},
            feature_config={'indicators': ['sma']},
            performance_metrics={'accuracy': 0.85},
            training_metadata={'duration': 120},
            created_at="2024-01-01T12:00:00",
            model_hash="abc123"
        )
        
        assert state.model_type == "XGBClassifier"
        assert state.feature_names == ['f1', 'f2']
        assert state.hyperparameters['n_estimators'] == 100
        assert state.performance_metrics['accuracy'] == 0.85


if __name__ == "__main__":
    pytest.main([__file__])