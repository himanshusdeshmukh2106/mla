"""
Unit tests for data splitting and preparation system
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.models.data_splitter import DataSplitter, DataSplit, SplitReport
from src.interfaces import ValidationResult


class TestDataSplitter:
    """Test cases for DataSplitter class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time-series data for testing"""
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'datetime': dates,
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.choice([0, 1], 100),
            'exclude_me': np.random.randn(100)
        })
        
        return data
        
    @pytest.fixture
    def splitter(self):
        """Create DataSplitter instance with default settings"""
        return DataSplitter(
            test_size=0.2,
            exclude_columns=['exclude_me'],
            target_column='target',
            datetime_column='datetime'
        )
        
    def test_init_valid_parameters(self):
        """Test DataSplitter initialization with valid parameters"""
        splitter = DataSplitter(test_size=0.3, exclude_columns=['col1', 'col2'])
        
        assert splitter.test_size == 0.3
        assert splitter.exclude_columns == ['col1', 'col2']
        assert splitter.target_column == 'target'
        assert splitter.datetime_column == 'datetime'
        
    def test_init_invalid_test_size(self):
        """Test DataSplitter initialization with invalid test_size"""
        with pytest.raises(ValueError, match="test_size must be between 0.0 and 1.0"):
            DataSplitter(test_size=0.0)
            
        with pytest.raises(ValueError, match="test_size must be between 0.0 and 1.0"):
            DataSplitter(test_size=1.0)
            
        with pytest.raises(ValueError, match="test_size must be between 0.0 and 1.0"):
            DataSplitter(test_size=-0.1)
            
    def test_prepare_data_success(self, splitter, sample_data):
        """Test successful data preparation"""
        X, y, feature_names = splitter.prepare_data(sample_data)
        
        # Check shapes
        assert X.shape[0] == 100
        assert X.shape[1] == 3  # feature1, feature2, feature3
        assert y.shape[0] == 100
        
        # Check feature names
        expected_features = ['feature1', 'feature2', 'feature3']
        assert feature_names == expected_features
        
        # Check data types
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        
    def test_prepare_data_empty_dataframe(self, splitter):
        """Test data preparation with empty DataFrame"""
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Input data is empty"):
            splitter.prepare_data(empty_data)
            
    def test_prepare_data_missing_target(self, splitter, sample_data):
        """Test data preparation with missing target column"""
        data_no_target = sample_data.drop('target', axis=1)
        
        with pytest.raises(ValueError, match="Target column 'target' not found"):
            splitter.prepare_data(data_no_target)
            
    def test_prepare_data_no_features(self, sample_data):
        """Test data preparation when all columns are excluded"""
        # Create splitter that excludes all feature columns
        splitter = DataSplitter(
            exclude_columns=['feature1', 'feature2', 'feature3', 'exclude_me']
        )
        
        with pytest.raises(ValueError, match="No feature columns found after exclusions"):
            splitter.prepare_data(sample_data)
            
    def test_prepare_data_with_nan_features(self, sample_data):
        """Test data preparation with NaN values in features"""
        sample_data.loc[0, 'feature1'] = np.nan
        
        # Create splitter with mocked logger
        with patch('src.models.data_splitter.logging.getLogger') as mock_logger:
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance
            
            splitter = DataSplitter(
                test_size=0.2,
                exclude_columns=['exclude_me'],
                target_column='target',
                datetime_column='datetime'
            )
            
            X, y, feature_names = splitter.prepare_data(sample_data)
            
            # Should log warning but not raise error
            mock_logger_instance.warning.assert_called_once()
            assert "NaN values found in features" in str(mock_logger_instance.warning.call_args)
            
    def test_prepare_data_with_nan_target(self, splitter, sample_data):
        """Test data preparation with NaN values in target"""
        sample_data.loc[0, 'target'] = np.nan
        
        with pytest.raises(ValueError, match="NaN values found in target variable"):
            splitter.prepare_data(sample_data)
            
    def test_split_data_chronologically_success(self, splitter, sample_data):
        """Test successful chronological data splitting"""
        data_split = splitter.split_data_chronologically(sample_data)
        
        # Check split object type
        assert isinstance(data_split, DataSplit)
        
        # Check split sizes
        assert len(data_split.X_train) == 80  # 80% of 100
        assert len(data_split.X_test) == 20   # 20% of 100
        assert len(data_split.y_train) == 80
        assert len(data_split.y_test) == 20
        
        # Check feature count
        assert data_split.X_train.shape[1] == 3
        assert data_split.X_test.shape[1] == 3
        
        # Check indices
        assert len(data_split.train_indices) == 80
        assert len(data_split.test_indices) == 20
        assert data_split.train_indices[-1] == 79
        assert data_split.test_indices[0] == 80
        
        # Check feature names
        assert data_split.feature_names == ['feature1', 'feature2', 'feature3']
        
        # Check split timestamp
        expected_timestamp = sample_data.iloc[80]['datetime']
        assert data_split.split_timestamp == expected_timestamp
        
    def test_split_data_empty_dataframe(self, splitter):
        """Test data splitting with empty DataFrame"""
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Input data is empty"):
            splitter.split_data_chronologically(empty_data)
            
    def test_split_data_insufficient_samples(self, splitter):
        """Test data splitting with insufficient samples"""
        # Create very small dataset
        small_data = pd.DataFrame({
            'datetime': [datetime.now()],
            'feature1': [1.0],
            'target': [1]
        })
        
        with pytest.raises(ValueError, match="Invalid split index"):
            splitter.split_data_chronologically(small_data)
            
    def test_split_data_with_datetime_index(self, splitter):
        """Test data splitting with datetime index instead of column"""
        # Create data with datetime index
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        }, index=dates)
        
        # Update splitter to not expect datetime column
        splitter.datetime_column = None
        
        data_split = splitter.split_data_chronologically(data)
        
        assert len(data_split.X_train) == 80
        assert len(data_split.X_test) == 20
        
    def test_split_data_no_datetime_info(self, sample_data):
        """Test data splitting without datetime information"""
        # Remove datetime column
        data_no_datetime = sample_data.drop('datetime', axis=1)
        
        with patch('src.models.data_splitter.logging.getLogger') as mock_logger:
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance
            
            splitter = DataSplitter(
                test_size=0.2,
                exclude_columns=['exclude_me'],
                target_column='target',
                datetime_column='datetime'
            )
            
            data_split = splitter.split_data_chronologically(data_no_datetime)
            
            # Should log warning
            mock_logger_instance.warning.assert_called_once()
            assert "No datetime column found" in str(mock_logger_instance.warning.call_args)
            
            # Should still work
            assert len(data_split.X_train) == 80
            assert len(data_split.X_test) == 20
            
    def test_validate_split_success(self, splitter, sample_data):
        """Test successful split validation"""
        data_split = splitter.split_data_chronologically(sample_data)
        validation_result = splitter.validate_split(data_split, sample_data)
        
        assert isinstance(validation_result, ValidationResult)
        assert validation_result.is_valid is True
        assert len(validation_result.errors) == 0
        
    def test_validate_split_sample_count_mismatch(self, splitter, sample_data):
        """Test split validation with sample count mismatch"""
        data_split = splitter.split_data_chronologically(sample_data)
        
        # Artificially modify split to create mismatch
        data_split.X_train = data_split.X_train[:-1]  # Remove one sample
        
        validation_result = splitter.validate_split(data_split, sample_data)
        
        assert validation_result.is_valid is False
        assert any("Split samples" in error for error in validation_result.errors)
        
    def test_validate_split_feature_count_mismatch(self, splitter, sample_data):
        """Test split validation with feature count mismatch"""
        data_split = splitter.split_data_chronologically(sample_data)
        
        # Artificially modify split to create feature mismatch
        data_split.X_test = data_split.X_test[:, :-1]  # Remove one feature
        
        validation_result = splitter.validate_split(data_split, sample_data)
        
        assert validation_result.is_valid is False
        assert any("Feature count mismatch" in error for error in validation_result.errors)
        
    def test_validate_split_empty_train_set(self, splitter, sample_data):
        """Test split validation with empty training set"""
        data_split = splitter.split_data_chronologically(sample_data)
        
        # Make training set empty
        data_split.X_train = np.array([]).reshape(0, 3)
        data_split.y_train = np.array([])
        
        validation_result = splitter.validate_split(data_split, sample_data)
        
        assert validation_result.is_valid is False
        assert any("Training set is empty" in error for error in validation_result.errors)
        
    def test_validate_split_empty_test_set(self, splitter, sample_data):
        """Test split validation with empty test set"""
        data_split = splitter.split_data_chronologically(sample_data)
        
        # Make test set empty
        data_split.X_test = np.array([]).reshape(0, 3)
        data_split.y_test = np.array([])
        
        validation_result = splitter.validate_split(data_split, sample_data)
        
        assert validation_result.is_valid is False
        assert any("Test set is empty" in error for error in validation_result.errors)
        
    def test_generate_split_report(self, splitter, sample_data):
        """Test split report generation"""
        data_split = splitter.split_data_chronologically(sample_data)
        report = splitter.generate_split_report(data_split, sample_data)
        
        assert isinstance(report, SplitReport)
        assert report.total_samples == 100
        assert report.train_samples == 80
        assert report.test_samples == 20
        assert report.train_ratio == 0.8
        assert report.test_ratio == 0.2
        assert report.feature_count == 3
        
        # Check target distributions
        assert isinstance(report.target_distribution_train, dict)
        assert isinstance(report.target_distribution_test, dict)
        
        # Check date ranges
        assert isinstance(report.date_range_train, tuple)
        assert isinstance(report.date_range_test, tuple)
        
    def test_different_test_sizes(self, sample_data):
        """Test splitting with different test sizes"""
        test_sizes = [0.1, 0.3, 0.5]
        
        for test_size in test_sizes:
            splitter = DataSplitter(test_size=test_size)
            data_split = splitter.split_data_chronologically(sample_data)
            
            expected_test_samples = int(100 * test_size)
            expected_train_samples = 100 - expected_test_samples
            
            assert len(data_split.X_test) == expected_test_samples
            assert len(data_split.X_train) == expected_train_samples
            
    def test_chronological_order_preservation(self, splitter, sample_data):
        """Test that chronological order is preserved in splits"""
        data_split = splitter.split_data_chronologically(sample_data)
        
        # Check that train indices come before test indices
        assert data_split.train_indices[-1] < data_split.test_indices[0]
        
        # Check that indices are sequential
        expected_train_indices = np.arange(80)
        expected_test_indices = np.arange(80, 100)
        
        np.testing.assert_array_equal(data_split.train_indices, expected_train_indices)
        np.testing.assert_array_equal(data_split.test_indices, expected_test_indices)
        
    def test_data_integrity_with_unsorted_data(self, splitter):
        """Test data splitting with unsorted datetime data"""
        # Create unsorted data
        dates = pd.date_range('2023-01-01', periods=50, freq='1h')
        shuffled_dates = dates.to_series().sample(frac=1, random_state=42).values
        
        data = pd.DataFrame({
            'datetime': shuffled_dates,
            'feature1': np.random.randn(50),
            'target': np.random.choice([0, 1], 50)
        })
        
        data_split = splitter.split_data_chronologically(data)
        
        # Should still work and sort the data
        assert len(data_split.X_train) == 40  # 80% of 50
        assert len(data_split.X_test) == 10   # 20% of 50
        
    def test_logging_behavior(self, sample_data):
        """Test that appropriate logging occurs"""
        with patch('src.models.data_splitter.logging.getLogger') as mock_logger:
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance
            
            splitter = DataSplitter(
                test_size=0.2,
                exclude_columns=['exclude_me'],
                target_column='target',
                datetime_column='datetime'
            )
            
            # Test prepare_data logging
            splitter.prepare_data(sample_data)
            mock_logger_instance.info.assert_called()
            
            # Test split_data_chronologically logging
            splitter.split_data_chronologically(sample_data)
            
            # Should have multiple info calls
            assert mock_logger_instance.info.call_count >= 2