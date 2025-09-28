"""
Unit tests for ModelTrainer class
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from src.models.trainer import (
    ModelTrainer, TrainingConfig, HyperparameterGrid, 
    TrainingResult, ModelTrainingError
)
from src.models.data_splitter import DataSplit
from src.interfaces import ModelMetrics
from xgboost import XGBClassifier


class TestTrainingConfig:
    """Test TrainingConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = TrainingConfig()
        
        assert config.algorithm == "xgboost"
        assert config.objective == "binary:logistic"
        assert config.test_size == 0.2
        assert config.cv_splits == 5
        assert config.early_stopping_rounds == 10
        assert config.eval_metric == "logloss"
        assert config.random_state == 42
        assert config.n_jobs == -1
        assert config.verbose is False
        
    def test_custom_config(self):
        """Test custom configuration values"""
        config = TrainingConfig(
            cv_splits=3,
            early_stopping_rounds=5,
            random_state=123
        )
        
        assert config.cv_splits == 3
        assert config.early_stopping_rounds == 5
        assert config.random_state == 123


class TestHyperparameterGrid:
    """Test HyperparameterGrid dataclass"""
    
    def test_default_grid(self):
        """Test default hyperparameter grid"""
        grid = HyperparameterGrid()
        
        assert grid.max_depth == [3, 5, 7]
        assert grid.learning_rate == [0.01, 0.1, 0.2]
        assert grid.n_estimators == [100, 200, 300]
        assert grid.gamma == [0, 0.1, 0.2]
        
    def test_custom_grid(self):
        """Test custom hyperparameter grid"""
        grid = HyperparameterGrid(
            max_depth=[3, 5],
            learning_rate=[0.1, 0.2],
            n_estimators=[100, 200]
        )
        
        assert grid.max_depth == [3, 5]
        assert grid.learning_rate == [0.1, 0.2]
        assert grid.n_estimators == [100, 200]
        
    def test_to_dict(self):
        """Test conversion to dictionary"""
        grid = HyperparameterGrid(
            max_depth=[3, 5],
            learning_rate=[0.1, 0.2]
        )
        
        param_dict = grid.to_dict()
        
        assert isinstance(param_dict, dict)
        assert param_dict['max_depth'] == [3, 5]
        assert param_dict['learning_rate'] == [0.1, 0.2]
        assert 'n_estimators' in param_dict


class TestModelTrainer:
    """Test ModelTrainer class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 1000
        
        # Create datetime index
        dates = pd.date_range(
            start='2023-01-01', 
            periods=n_samples, 
            freq='1min'
        )
        
        # Create features
        data = pd.DataFrame({
            'datetime': dates,
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'feature_4': np.random.randn(n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        
        return data
        
    @pytest.fixture
    def sample_data_split(self, sample_data):
        """Create sample DataSplit object"""
        from src.models.data_splitter import DataSplitter
        
        splitter = DataSplitter(test_size=0.2)
        return splitter.split_data_chronologically(sample_data)
        
    @pytest.fixture
    def trainer(self):
        """Create ModelTrainer instance"""
        config = TrainingConfig(cv_splits=2, verbose=False)  # Reduce CV splits for faster testing
        grid = HyperparameterGrid(
            max_depth=[3, 5],
            learning_rate=[0.1, 0.2],
            n_estimators=[50, 100]  # Reduce for faster testing
        )
        return ModelTrainer(config=config, hyperparameter_grid=grid)
        
    def test_initialization(self):
        """Test ModelTrainer initialization"""
        trainer = ModelTrainer()
        
        assert isinstance(trainer.config, TrainingConfig)
        assert isinstance(trainer.hyperparameter_grid, HyperparameterGrid)
        assert trainer.model is None
        assert trainer.best_params is None
        assert trainer.feature_names is None
        
    def test_initialization_with_custom_config(self):
        """Test ModelTrainer initialization with custom config"""
        config = TrainingConfig(cv_splits=3)
        grid = HyperparameterGrid(max_depth=[3, 5])
        
        trainer = ModelTrainer(config=config, hyperparameter_grid=grid)
        
        assert trainer.config.cv_splits == 3
        assert trainer.hyperparameter_grid.max_depth == [3, 5]
        
    def test_prepare_data_success(self, trainer, sample_data):
        """Test successful data preparation"""
        X, y = trainer.prepare_data(sample_data)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == len(sample_data)
        assert X.shape[1] == 4  # 4 feature columns
        assert len(y) == len(sample_data)
        assert trainer.feature_names == ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        
    def test_prepare_data_empty_dataframe(self, trainer):
        """Test data preparation with empty DataFrame"""
        empty_data = pd.DataFrame()
        
        with pytest.raises(ModelTrainingError, match="Input data is empty"):
            trainer.prepare_data(empty_data)
            
    def test_prepare_data_no_target_column(self, trainer, sample_data):
        """Test data preparation without target column"""
        data_no_target = sample_data.drop('target', axis=1)
        
        with pytest.raises(ModelTrainingError, match="Target column 'target' not found"):
            trainer.prepare_data(data_no_target)
            
    def test_prepare_data_no_features(self, trainer):
        """Test data preparation with no feature columns"""
        data_no_features = pd.DataFrame({
            'target': [0, 1, 0, 1],
            'datetime': pd.date_range('2023-01-01', periods=4)
        })
        
        with pytest.raises(ModelTrainingError, match="No feature columns found"):
            trainer.prepare_data(data_no_features)
            
    def test_prepare_data_with_nan_features(self, trainer, sample_data):
        """Test data preparation with NaN in features"""
        sample_data.loc[0, 'feature_1'] = np.nan
        
        with pytest.raises(ModelTrainingError, match="NaN values found in features"):
            trainer.prepare_data(sample_data)
            
    def test_prepare_data_with_nan_target(self, trainer, sample_data):
        """Test data preparation with NaN in target"""
        sample_data.loc[0, 'target'] = np.nan
        
        with pytest.raises(ModelTrainingError, match="NaN values found in target"):
            trainer.prepare_data(sample_data)
            
    def test_prepare_data_invalid_target_values(self, trainer, sample_data):
        """Test data preparation with invalid target values"""
        sample_data['target'] = [0, 1, 2, 3] * (len(sample_data) // 4)
        
        with pytest.raises(ModelTrainingError, match="Target values must be 0 or 1"):
            trainer.prepare_data(sample_data)
            
    def test_create_base_model(self, trainer):
        """Test base model creation"""
        model = trainer.create_base_model()
        
        assert isinstance(model, XGBClassifier)
        assert model.objective == "binary:logistic"
        assert model.random_state == 42
        assert model.n_jobs == -1
        
    @patch('src.models.trainer.GridSearchCV')
    def test_tune_hyperparameters_success(self, mock_grid_search, trainer, sample_data):
        """Test successful hyperparameter tuning"""
        # Mock GridSearchCV
        mock_grid_instance = Mock()
        mock_grid_instance.best_params_ = {'max_depth': 5, 'learning_rate': 0.1}
        mock_grid_instance.best_score_ = 0.85
        mock_grid_search.return_value = mock_grid_instance
        
        X, y = trainer.prepare_data(sample_data)
        best_params = trainer.tune_hyperparameters(X, y)
        
        assert best_params == {'max_depth': 5, 'learning_rate': 0.1}
        assert trainer.best_params == {'max_depth': 5, 'learning_rate': 0.1}
        mock_grid_search.assert_called_once()
        mock_grid_instance.fit.assert_called_once_with(X, y)
        
    @patch('src.models.trainer.GridSearchCV')
    def test_tune_hyperparameters_failure(self, mock_grid_search, trainer, sample_data):
        """Test hyperparameter tuning failure"""
        # Mock GridSearchCV to raise exception
        mock_grid_instance = Mock()
        mock_grid_instance.fit.side_effect = Exception("Grid search failed")
        mock_grid_search.return_value = mock_grid_instance
        
        X, y = trainer.prepare_data(sample_data)
        
        with pytest.raises(ModelTrainingError, match="Hyperparameter tuning failed"):
            trainer.tune_hyperparameters(X, y)
            
    def test_train_model_with_hyperparameters(self, trainer, sample_data):
        """Test model training with provided hyperparameters"""
        X, y = trainer.prepare_data(sample_data)
        
        # Split data for training
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        hyperparams = {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 50}
        
        model = trainer.train_model(X_train, y_train, X_val, y_val, hyperparams)
        
        assert isinstance(model, XGBClassifier)
        assert model.max_depth == 3
        assert model.learning_rate == 0.1
        assert model.n_estimators == 50
        assert trainer.model is not None
        
    @patch.object(ModelTrainer, 'tune_hyperparameters')
    def test_train_model_with_tuning(self, mock_tune, trainer, sample_data):
        """Test model training with hyperparameter tuning"""
        mock_tune.return_value = {'max_depth': 5, 'learning_rate': 0.2}
        
        X, y = trainer.prepare_data(sample_data)
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        
        model = trainer.train_model(X_train, y_train)
        
        assert isinstance(model, XGBClassifier)
        mock_tune.assert_called_once_with(X_train, y_train)
        
    def test_train_model_failure(self, trainer, sample_data):
        """Test model training failure"""
        X, y = trainer.prepare_data(sample_data)
        
        # Create invalid hyperparameters to cause failure
        invalid_hyperparams = {'max_depth': -1}  # Invalid parameter
        
        with pytest.raises(ModelTrainingError, match="Model training failed"):
            trainer.train_model(X, y, hyperparameters=invalid_hyperparams)
            
    def test_evaluate_model_success(self, trainer, sample_data):
        """Test successful model evaluation"""
        X, y = trainer.prepare_data(sample_data)
        
        # Train a simple model
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        hyperparams = {'max_depth': 3, 'n_estimators': 10}  # Small model for testing
        model = trainer.train_model(X_train, y_train, hyperparameters=hyperparams)
        
        metrics = trainer.evaluate_model(model, X_test, y_test)
        
        assert isinstance(metrics, ModelMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1_score <= 1
        assert metrics.confusion_matrix.shape == (2, 2)
        assert isinstance(metrics.feature_importance, dict)
        assert len(metrics.feature_importance) == 4  # 4 features
        
    def test_evaluate_model_no_model(self, trainer, sample_data):
        """Test model evaluation without trained model"""
        X, y = trainer.prepare_data(sample_data)
        
        with pytest.raises(ModelTrainingError, match="Model is not trained"):
            trainer.evaluate_model(None, X, y)
            
    def test_train_and_evaluate_pipeline(self, trainer, sample_data_split):
        """Test complete training and evaluation pipeline"""
        # Use a trainer without early stopping for cross-validation
        config = TrainingConfig(cv_splits=2, verbose=False, early_stopping_rounds=None)
        grid = HyperparameterGrid(
            max_depth=[3, 5],
            learning_rate=[0.1, 0.2],
            n_estimators=[10, 20]  # Small values for faster testing
        )
        trainer_no_early_stop = ModelTrainer(config=config, hyperparameter_grid=grid)
        
        model, metrics = trainer_no_early_stop.train_and_evaluate(sample_data_split)
        
        assert isinstance(model, XGBClassifier)
        assert isinstance(metrics, ModelMetrics)
        assert 0 <= metrics.accuracy <= 1
        
    def test_save_and_load_model(self, trainer, sample_data):
        """Test model saving and loading"""
        X, y = trainer.prepare_data(sample_data)
        
        # Train a simple model
        hyperparams = {'max_depth': 3, 'n_estimators': 10}
        model = trainer.train_model(X, y, hyperparameters=hyperparams)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'test_model.pkl')
            
            # Save model
            metadata = {'test_key': 'test_value'}
            trainer.save_model(model, filepath, metadata)
            
            assert os.path.exists(filepath)
            assert os.path.exists(filepath.replace('.pkl', '_metadata.pkl'))
            
            # Load model
            loaded_model, loaded_metadata = trainer.load_model(filepath)
            
            assert isinstance(loaded_model, XGBClassifier)
            assert loaded_metadata['test_key'] == 'test_value'
            assert loaded_metadata['feature_names'] == trainer.feature_names
            
    def test_save_model_failure(self, trainer):
        """Test model saving failure"""
        model = Mock()
        invalid_filepath = '/invalid/path/model.pkl'
        
        with pytest.raises(ModelTrainingError, match="Failed to save model"):
            trainer.save_model(model, invalid_filepath)
            
    def test_load_model_failure(self, trainer):
        """Test model loading failure"""
        nonexistent_filepath = '/nonexistent/model.pkl'
        
        with pytest.raises(ModelTrainingError, match="Failed to load model"):
            trainer.load_model(nonexistent_filepath)


class TestModelTrainerIntegration:
    """Integration tests for ModelTrainer"""
    
    @pytest.fixture
    def large_sample_data(self):
        """Create larger sample dataset for integration testing"""
        np.random.seed(42)
        n_samples = 2000
        
        dates = pd.date_range(
            start='2023-01-01', 
            periods=n_samples, 
            freq='1min'
        )
        
        # Create more realistic features with some correlation
        feature_1 = np.random.randn(n_samples)
        feature_2 = feature_1 * 0.5 + np.random.randn(n_samples) * 0.5
        feature_3 = np.random.randn(n_samples)
        feature_4 = feature_3 * 0.3 + np.random.randn(n_samples) * 0.7
        
        # Create target with some relationship to features
        target_prob = 1 / (1 + np.exp(-(feature_1 + feature_2 * 0.5)))
        target = (np.random.random(n_samples) < target_prob).astype(int)
        
        data = pd.DataFrame({
            'datetime': dates,
            'feature_1': feature_1,
            'feature_2': feature_2,
            'feature_3': feature_3,
            'feature_4': feature_4,
            'target': target
        })
        
        return data
        
    def test_full_training_pipeline(self, large_sample_data):
        """Test complete training pipeline with realistic data"""
        from src.models.data_splitter import DataSplitter
        
        # Create data split
        splitter = DataSplitter(test_size=0.2)
        data_split = splitter.split_data_chronologically(large_sample_data)
        
        # Create trainer with reduced parameters for faster testing
        # Disable early stopping for cross-validation
        config = TrainingConfig(cv_splits=2, verbose=False, early_stopping_rounds=None)
        grid = HyperparameterGrid(
            max_depth=[3, 5],
            learning_rate=[0.1, 0.2],
            n_estimators=[10, 20]  # Reduced for faster testing
        )
        trainer = ModelTrainer(config=config, hyperparameter_grid=grid)
        
        # Train and evaluate
        model, metrics = trainer.train_and_evaluate(data_split)
        
        # Verify results
        assert isinstance(model, XGBClassifier)
        assert isinstance(metrics, ModelMetrics)
        assert metrics.accuracy > 0.4  # Should be reasonable (lowered threshold for small models)
        assert len(metrics.feature_importance) == 4
        
        # Verify feature importance makes sense
        importance_sum = sum(metrics.feature_importance.values())
        assert abs(importance_sum - 1.0) < 0.01  # Should sum to approximately 1
        
    def test_model_persistence_integration(self, large_sample_data):
        """Test model training, saving, and loading integration"""
        from src.models.data_splitter import DataSplitter
        
        # Train model
        splitter = DataSplitter(test_size=0.2)
        data_split = splitter.split_data_chronologically(large_sample_data)
        
        # Disable early stopping for cross-validation
        config = TrainingConfig(cv_splits=2, verbose=False, early_stopping_rounds=None)
        grid = HyperparameterGrid(
            max_depth=[3],
            learning_rate=[0.1],
            n_estimators=[10]  # Single configuration for faster testing
        )
        trainer = ModelTrainer(config=config, hyperparameter_grid=grid)
        
        model, metrics = trainer.train_and_evaluate(data_split)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'integration_model.pkl')
            
            # Save model
            trainer.save_model(model, filepath, {'accuracy': metrics.accuracy})
            
            # Create new trainer and load model
            new_trainer = ModelTrainer()
            loaded_model, metadata = new_trainer.load_model(filepath)
            
            # Verify loaded model produces same predictions
            test_predictions_original = model.predict(data_split.X_test)
            test_predictions_loaded = loaded_model.predict(data_split.X_test)
            
            np.testing.assert_array_equal(test_predictions_original, test_predictions_loaded)
            assert metadata['accuracy'] == metrics.accuracy


if __name__ == '__main__':
    pytest.main([__file__])