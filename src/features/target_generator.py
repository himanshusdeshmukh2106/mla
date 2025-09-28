"""
Target variable generation for binary classification trading signals
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from src.logger import get_logger

logger = get_logger(__name__)


class TargetGenerator:
    """
    Generate binary classification targets for trading signals
    """
    
    def __init__(self, config: Dict):
        """
        Initialize TargetGenerator with configuration
        
        Args:
            config: Dictionary containing target generation parameters
        """
        self.config = config
        self.lookahead_periods = config.get('lookahead_periods', 1)
        self.profit_threshold = config.get('profit_threshold', 0.001)  # 0.1% default
        self.loss_threshold = config.get('loss_threshold', -0.001)  # -0.1% default
        self.method = config.get('method', 'next_period_return')  # 'next_period_return' or 'fixed_horizon'
        
        logger.info(f"TargetGenerator initialized with method: {self.method}, "
                   f"lookahead: {self.lookahead_periods}, "
                   f"thresholds: profit={self.profit_threshold}, loss={self.loss_threshold}")
    
    def generate_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate binary classification targets from price data
        
        Args:
            data: DataFrame with OHLCV data including 'close' column
            
        Returns:
            DataFrame with original data plus target columns
        """
        logger.info(f"Generating targets for {len(data)} data points")
        
        df = data.copy()
        
        if self.method == 'next_period_return':
            df = self._generate_next_period_targets(df)
        elif self.method == 'fixed_horizon':
            df = self._generate_fixed_horizon_targets(df)
        else:
            raise ValueError(f"Unknown target generation method: {self.method}")
        
        # Clean undefined target values
        df = self._clean_undefined_targets(df)
        
        initial_rows = len(data)
        final_rows = len(df)
        logger.info(f"Target generation complete. Rows: {initial_rows} -> {final_rows} "
                   f"(removed {initial_rows - final_rows} undefined targets)")
        
        return df
    
    def _generate_next_period_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate targets based on next period price comparison
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with target columns added
        """
        df = data.copy()
        
        # Calculate future returns
        df['future_return'] = df['close'].shift(-self.lookahead_periods) / df['close'] - 1
        
        # Generate binary targets based on thresholds
        df['target'] = 0  # Default to hold (0)
        
        # Buy signal (1) if future return exceeds profit threshold
        df.loc[df['future_return'] >= self.profit_threshold, 'target'] = 1
        
        # Sell signal (-1) if future return falls below loss threshold
        df.loc[df['future_return'] <= self.loss_threshold, 'target'] = -1
        
        # For binary classification, convert to 0/1
        df['target_binary'] = (df['target'] == 1).astype(int)
        
        logger.debug(f"Next period targets generated. "
                    f"Buy signals: {(df['target'] == 1).sum()}, "
                    f"Sell signals: {(df['target'] == -1).sum()}, "
                    f"Hold signals: {(df['target'] == 0).sum()}")
        
        return df
    
    def _generate_fixed_horizon_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate targets based on fixed horizon maximum/minimum prices
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with target columns added
        """
        df = data.copy()
        
        # Calculate rolling maximum and minimum over lookahead period
        future_high = df['high'].shift(-self.lookahead_periods).rolling(
            window=self.lookahead_periods, min_periods=1
        ).max()
        
        future_low = df['low'].shift(-self.lookahead_periods).rolling(
            window=self.lookahead_periods, min_periods=1
        ).min()
        
        # Calculate potential returns
        max_return = future_high / df['close'] - 1
        min_return = future_low / df['close'] - 1
        
        # Generate targets based on which threshold is hit first
        df['target'] = 0  # Default to hold
        
        # Buy signal if max return exceeds profit threshold and occurs before loss threshold
        profit_condition = max_return >= self.profit_threshold
        loss_condition = min_return <= self.loss_threshold
        
        # Buy if profit threshold is hit and no significant loss
        df.loc[profit_condition & ~loss_condition, 'target'] = 1
        
        # Sell if loss threshold is hit and no significant profit
        df.loc[loss_condition & ~profit_condition, 'target'] = -1
        
        # If both thresholds are hit, choose based on which is more extreme
        both_condition = profit_condition & loss_condition
        profit_magnitude = np.abs(max_return)
        loss_magnitude = np.abs(min_return)
        
        df.loc[both_condition & (profit_magnitude > loss_magnitude), 'target'] = 1
        df.loc[both_condition & (loss_magnitude > profit_magnitude), 'target'] = -1
        
        # Store the actual returns for analysis
        df['future_max_return'] = max_return
        df['future_min_return'] = min_return
        
        # For binary classification, convert to 0/1
        df['target_binary'] = (df['target'] == 1).astype(int)
        
        logger.debug(f"Fixed horizon targets generated. "
                    f"Buy signals: {(df['target'] == 1).sum()}, "
                    f"Sell signals: {(df['target'] == -1).sum()}, "
                    f"Hold signals: {(df['target'] == 0).sum()}")
        
        return df
    
    def _clean_undefined_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with undefined target values (NaN or at the end of dataset)
        
        Args:
            data: DataFrame with target columns
            
        Returns:
            DataFrame with undefined targets removed
        """
        df = data.copy()
        
        # Remove rows where target is NaN
        before_cleaning = len(df)
        df = df.dropna(subset=['target'])
        after_cleaning = len(df)
        
        if before_cleaning != after_cleaning:
            logger.debug(f"Removed {before_cleaning - after_cleaning} rows with undefined targets")
        
        return df
    
    def get_target_distribution(self, data: pd.DataFrame) -> Dict[str, int]:
        """
        Get distribution of target values
        
        Args:
            data: DataFrame with target column
            
        Returns:
            Dictionary with target value counts
        """
        if 'target' not in data.columns:
            return {}
        
        distribution = data['target'].value_counts().to_dict()
        
        # Add percentage information
        total = len(data)
        distribution_pct = {
            f"{k}_pct": round(v / total * 100, 2) 
            for k, v in distribution.items()
        }
        
        result = {**distribution, **distribution_pct}
        logger.info(f"Target distribution: {result}")
        
        return result
    
    def validate_targets(self, data: pd.DataFrame) -> Tuple[bool, Dict[str, any]]:
        """
        Validate target generation results
        
        Args:
            data: DataFrame with target columns
            
        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {}
        is_valid = True
        
        # Check if target column exists
        if 'target' not in data.columns:
            validation_info['error'] = "Target column not found"
            return False, validation_info
        
        # Check for valid target values
        valid_targets = {-1, 0, 1}
        unique_targets = set(data['target'].dropna().unique())
        invalid_targets = unique_targets - valid_targets
        
        if invalid_targets:
            validation_info['error'] = f"Invalid target values found: {invalid_targets}"
            is_valid = False
        
        # Check target distribution
        distribution = self.get_target_distribution(data)
        validation_info['distribution'] = distribution
        
        # Check for class imbalance (warn if any class < 5%)
        for target_val in [-1, 0, 1]:
            pct_key = f"{target_val}_pct"
            if pct_key in distribution and distribution[pct_key] < 5.0:
                validation_info['warning'] = f"Class {target_val} has low representation: {distribution[pct_key]}%"
        
        # Check for sufficient data
        if len(data) < 100:
            validation_info['warning'] = f"Small dataset size: {len(data)} rows"
        
        validation_info['total_samples'] = len(data)
        validation_info['valid_targets'] = len(data.dropna(subset=['target']))
        
        logger.info(f"Target validation: {'PASSED' if is_valid else 'FAILED'}")
        
        return is_valid, validation_info
    
    def create_stratified_split_indices(self, data: pd.DataFrame, 
                                      train_ratio: float = 0.7, 
                                      val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create stratified train/validation/test split indices
        
        Args:
            data: DataFrame with target column
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set (test gets remainder)
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        if 'target' not in data.columns:
            raise ValueError("Target column not found in data")
        
        # Use time-based split to avoid lookahead bias
        n_samples = len(data)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        train_indices = np.arange(0, train_end)
        val_indices = np.arange(train_end, val_end)
        test_indices = np.arange(val_end, n_samples)
        
        logger.info(f"Created time-based split: "
                   f"train={len(train_indices)}, "
                   f"val={len(val_indices)}, "
                   f"test={len(test_indices)}")
        
        return train_indices, val_indices, test_indices