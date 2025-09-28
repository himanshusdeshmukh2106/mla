"""
Data splitting and preparation system for time-series data
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from ..interfaces import ValidationResult
from ..exceptions import TradingSystemError


@dataclass
class DataSplit:
    """Container for split data"""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    feature_names: List[str]
    split_timestamp: datetime


@dataclass
class SplitReport:
    """Report on data split characteristics"""
    total_samples: int
    train_samples: int
    test_samples: int
    train_ratio: float
    test_ratio: float
    split_timestamp: datetime
    feature_count: int
    target_distribution_train: Dict[str, int]
    target_distribution_test: Dict[str, int]
    date_range_train: Tuple[datetime, datetime]
    date_range_test: Tuple[datetime, datetime]


class DataSplitter:
    """
    Handles chronological data splitting and preparation for time-series ML models
    """
    
    def __init__(self, 
                 test_size: float = 0.2,
                 exclude_columns: Optional[List[str]] = None,
                 target_column: str = 'target',
                 datetime_column: str = 'datetime'):
        """
        Initialize data splitter
        
        Args:
            test_size: Proportion of data to use for testing (0.0 to 1.0)
            exclude_columns: Columns to exclude from features (beyond target and datetime)
            target_column: Name of the target variable column
            datetime_column: Name of the datetime index column
        """
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0.0 and 1.0")
            
        self.test_size = test_size
        self.exclude_columns = exclude_columns or []
        self.target_column = target_column
        self.datetime_column = datetime_column
        self.logger = logging.getLogger(__name__)
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Separate features from target and excluded columns
        
        Args:
            data: DataFrame with features, target, and datetime
            
        Returns:
            Tuple of (features, target, feature_names)
        """
        if data.empty:
            raise ValueError("Input data is empty")
            
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
            
        # Define columns to exclude from features
        exclude_cols = set([self.target_column, self.datetime_column] + self.exclude_columns)
        
        # Get feature columns
        feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        if not feature_columns:
            raise ValueError("No feature columns found after exclusions")
            
        # Extract features and target
        X = data[feature_columns].values
        y = data[self.target_column].values
        
        # Check for NaN values
        if np.isnan(X).any():
            self.logger.warning("NaN values found in features, consider data cleaning")
            
        if np.isnan(y).any():
            raise ValueError("NaN values found in target variable")
            
        self.logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, feature_columns
        
    def split_data_chronologically(self, 
                                 data: pd.DataFrame) -> DataSplit:
        """
        Split data chronologically without shuffling
        
        Args:
            data: DataFrame with datetime index and features/target
            
        Returns:
            DataSplit object with train/test splits
        """
        if data.empty:
            raise ValueError("Input data is empty")
            
        # Ensure data is sorted by datetime
        if self.datetime_column in data.columns:
            data = data.sort_values(self.datetime_column)
        elif isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()
        else:
            self.logger.warning("No datetime column found, assuming data is already chronologically ordered")
            
        # Prepare features and target
        X, y, feature_names = self.prepare_data(data)
        
        # Calculate split point
        n_samples = len(data)
        split_idx = int(n_samples * (1 - self.test_size))
        
        if split_idx <= 0 or split_idx >= n_samples:
            raise ValueError(f"Invalid split index {split_idx} for {n_samples} samples")
            
        # Create chronological split
        train_indices = np.arange(split_idx)
        test_indices = np.arange(split_idx, n_samples)
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        # Get split timestamp
        if self.datetime_column in data.columns:
            split_timestamp = data.iloc[split_idx][self.datetime_column]
        elif isinstance(data.index, pd.DatetimeIndex):
            split_timestamp = data.index[split_idx]
        else:
            split_timestamp = datetime.now()
            
        self.logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        self.logger.info(f"Split timestamp: {split_timestamp}")
        
        return DataSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_indices=train_indices,
            test_indices=test_indices,
            feature_names=feature_names,
            split_timestamp=split_timestamp
        )
        
    def validate_split(self, data_split: DataSplit, original_data: pd.DataFrame) -> ValidationResult:
        """
        Validate the data split for time-series integrity
        
        Args:
            data_split: The split data to validate
            original_data: Original DataFrame for reference
            
        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        
        # Check split sizes
        total_samples = len(data_split.X_train) + len(data_split.X_test)
        expected_samples = len(original_data)
        
        if total_samples != expected_samples:
            errors.append(f"Split samples ({total_samples}) don't match original ({expected_samples})")
            
        # Check test size ratio
        actual_test_ratio = len(data_split.X_test) / total_samples
        expected_test_ratio = self.test_size
        
        if abs(actual_test_ratio - expected_test_ratio) > 0.05:  # 5% tolerance
            warnings.append(f"Test ratio {actual_test_ratio:.3f} differs from expected {expected_test_ratio:.3f}")
            
        # Check for data leakage (chronological order)
        if self.datetime_column in original_data.columns:
            train_end_idx = data_split.train_indices[-1]
            test_start_idx = data_split.test_indices[0]
            
            if test_start_idx <= train_end_idx:
                errors.append("Data leakage detected: test data overlaps with training data")
                
        # Check feature consistency
        if data_split.X_train.shape[1] != data_split.X_test.shape[1]:
            errors.append("Feature count mismatch between train and test sets")
            
        # Check for empty splits
        if len(data_split.X_train) == 0:
            errors.append("Training set is empty")
            
        if len(data_split.X_test) == 0:
            errors.append("Test set is empty")
            
        # Check target distribution
        unique_train = np.unique(data_split.y_train)
        unique_test = np.unique(data_split.y_test)
        
        if len(unique_train) != len(unique_test):
            warnings.append("Different number of target classes in train vs test sets")
            
        is_valid = len(errors) == 0
        
        if is_valid:
            self.logger.info("Data split validation passed")
        else:
            self.logger.error(f"Data split validation failed: {errors}")
            
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )
        
    def generate_split_report(self, 
                            data_split: DataSplit, 
                            original_data: pd.DataFrame) -> SplitReport:
        """
        Generate comprehensive report on data split
        
        Args:
            data_split: The split data
            original_data: Original DataFrame for reference
            
        Returns:
            SplitReport with detailed split information
        """
        total_samples = len(data_split.X_train) + len(data_split.X_test)
        train_samples = len(data_split.X_train)
        test_samples = len(data_split.X_test)
        
        # Target distribution
        train_unique, train_counts = np.unique(data_split.y_train, return_counts=True)
        test_unique, test_counts = np.unique(data_split.y_test, return_counts=True)
        
        target_dist_train = dict(zip(train_unique.astype(str), train_counts.tolist()))
        target_dist_test = dict(zip(test_unique.astype(str), test_counts.tolist()))
        
        # Date ranges
        if self.datetime_column in original_data.columns:
            train_dates = original_data.iloc[data_split.train_indices][self.datetime_column]
            test_dates = original_data.iloc[data_split.test_indices][self.datetime_column]
            
            date_range_train = (train_dates.min(), train_dates.max())
            date_range_test = (test_dates.min(), test_dates.max())
        else:
            # Use index if datetime column not available
            date_range_train = (datetime.min, datetime.min)
            date_range_test = (datetime.min, datetime.min)
            
        return SplitReport(
            total_samples=total_samples,
            train_samples=train_samples,
            test_samples=test_samples,
            train_ratio=train_samples / total_samples,
            test_ratio=test_samples / total_samples,
            split_timestamp=data_split.split_timestamp,
            feature_count=len(data_split.feature_names),
            target_distribution_train=target_dist_train,
            target_distribution_test=target_dist_test,
            date_range_train=date_range_train,
            date_range_test=date_range_test
        )