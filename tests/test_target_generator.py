"""
Unit tests for TargetGenerator class
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.target_generator import TargetGenerator


class TestTargetGenerator:
    """Test cases for TargetGenerator class"""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1min')
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic price data with trends
        base_price = 100.0
        trend = np.linspace(0, 0.1, 100)  # 10% upward trend over period
        noise = np.random.normal(0, 0.01, 100)  # 1% volatility
        
        prices = base_price * (1 + trend + noise)
        
        # Ensure realistic OHLCV relationships
        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'close': prices * (1 + np.random.normal(0, 0.005, 100)),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Generate high and low based on open and close
        data['high'] = np.maximum(data['open'], data['close']) * (1 + np.abs(np.random.normal(0, 0.003, 100)))
        data['low'] = np.minimum(data['open'], data['close']) * (1 - np.abs(np.random.normal(0, 0.003, 100)))
        
        return data.set_index('datetime')
    
    @pytest.fixture
    def target_config_next_period(self):
        """Configuration for next period target generation"""
        return {
            'method': 'next_period_return',
            'lookahead_periods': 1,
            'profit_threshold': 0.01,  # 1%
            'loss_threshold': -0.01    # -1%
        }
    
    @pytest.fixture
    def target_config_fixed_horizon(self):
        """Configuration for fixed horizon target generation"""
        return {
            'method': 'fixed_horizon',
            'lookahead_periods': 5,
            'profit_threshold': 0.02,  # 2%
            'loss_threshold': -0.02    # -2%
        }
    
    def test_target_generator_initialization(self, target_config_next_period):
        """Test TargetGenerator initialization"""
        generator = TargetGenerator(target_config_next_period)
        
        assert generator.config == target_config_next_period
        assert generator.method == 'next_period_return'
        assert generator.lookahead_periods == 1
        assert generator.profit_threshold == 0.01
        assert generator.loss_threshold == -0.01
    
    def test_next_period_target_generation(self, sample_ohlcv_data, target_config_next_period):
        """Test next period target generation"""
        generator = TargetGenerator(target_config_next_period)
        result = generator.generate_targets(sample_ohlcv_data)
        
        # Check that target columns are created
        assert 'target' in result.columns
        assert 'target_binary' in result.columns
        assert 'future_return' in result.columns
        
        # Check target values are valid
        unique_targets = result['target'].dropna().unique()
        assert all(target in [-1, 0, 1] for target in unique_targets)
        
        # Check binary targets are 0 or 1
        unique_binary = result['target_binary'].dropna().unique()
        assert all(target in [0, 1] for target in unique_binary)
        
        # Check that we have fewer rows due to lookahead
        assert len(result) <= len(sample_ohlcv_data)
        
        # Verify target logic
        profitable_rows = result[result['future_return'] >= target_config_next_period['profit_threshold']]
        if len(profitable_rows) > 0:
            assert all(profitable_rows['target'] == 1)
            assert all(profitable_rows['target_binary'] == 1)
        
        loss_rows = result[result['future_return'] <= target_config_next_period['loss_threshold']]
        if len(loss_rows) > 0:
            assert all(loss_rows['target'] == -1)
            assert all(loss_rows['target_binary'] == 0)
    
    def test_fixed_horizon_target_generation(self, sample_ohlcv_data, target_config_fixed_horizon):
        """Test fixed horizon target generation"""
        generator = TargetGenerator(target_config_fixed_horizon)
        result = generator.generate_targets(sample_ohlcv_data)
        
        # Check that target columns are created
        assert 'target' in result.columns
        assert 'target_binary' in result.columns
        assert 'future_max_return' in result.columns
        assert 'future_min_return' in result.columns
        
        # Check target values are valid
        unique_targets = result['target'].dropna().unique()
        assert all(target in [-1, 0, 1] for target in unique_targets)
        
        # Check that we have fewer rows due to lookahead
        assert len(result) <= len(sample_ohlcv_data)
        
        # Verify that max return >= min return (logical consistency)
        valid_rows = result.dropna(subset=['future_max_return', 'future_min_return'])
        if len(valid_rows) > 0:
            assert all(valid_rows['future_max_return'] >= valid_rows['future_min_return'])
    
    def test_target_distribution(self, sample_ohlcv_data, target_config_next_period):
        """Test target distribution calculation"""
        generator = TargetGenerator(target_config_next_period)
        result = generator.generate_targets(sample_ohlcv_data)
        
        distribution = generator.get_target_distribution(result)
        
        # Check that distribution contains expected keys
        assert isinstance(distribution, dict)
        
        # Check that percentages sum to approximately 100%
        pct_keys = [k for k in distribution.keys() if isinstance(k, str) and k.endswith('_pct')]
        if pct_keys:
            total_pct = sum(distribution[k] for k in pct_keys)
            assert abs(total_pct - 100.0) < 0.1  # Allow small rounding errors
    
    def test_target_validation(self, sample_ohlcv_data, target_config_next_period):
        """Test target validation functionality"""
        generator = TargetGenerator(target_config_next_period)
        result = generator.generate_targets(sample_ohlcv_data)
        
        is_valid, validation_info = generator.validate_targets(result)
        
        # Should be valid for properly generated targets
        assert is_valid is True
        assert 'distribution' in validation_info
        assert 'total_samples' in validation_info
        assert 'valid_targets' in validation_info
        
        # Test with invalid data (no target column)
        invalid_data = sample_ohlcv_data.copy()
        is_valid, validation_info = generator.validate_targets(invalid_data)
        
        assert is_valid is False
        assert 'error' in validation_info
    
    def test_stratified_split_indices(self, sample_ohlcv_data, target_config_next_period):
        """Test stratified split index generation"""
        generator = TargetGenerator(target_config_next_period)
        result = generator.generate_targets(sample_ohlcv_data)
        
        train_idx, val_idx, test_idx = generator.create_stratified_split_indices(
            result, train_ratio=0.6, val_ratio=0.2
        )
        
        # Check that indices don't overlap
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(set(val_idx) & set(test_idx)) == 0
        
        # Check that all indices are covered
        total_indices = len(train_idx) + len(val_idx) + len(test_idx)
        assert total_indices == len(result)
        
        # Check approximate ratios
        total_len = len(result)
        assert abs(len(train_idx) / total_len - 0.6) < 0.05
        assert abs(len(val_idx) / total_len - 0.2) < 0.05
        assert abs(len(test_idx) / total_len - 0.2) < 0.05
        
        # Check time-based ordering (train < val < test)
        if len(train_idx) > 0 and len(val_idx) > 0:
            assert max(train_idx) < min(val_idx)
        if len(val_idx) > 0 and len(test_idx) > 0:
            assert max(val_idx) < min(test_idx)
    
    def test_edge_cases(self, target_config_next_period):
        """Test edge cases and error handling"""
        generator = TargetGenerator(target_config_next_period)
        
        # Test with very small dataset
        small_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 1100]
        })
        
        result = generator.generate_targets(small_data)
        
        # Should handle gracefully
        assert isinstance(result, pd.DataFrame)
        assert 'target' in result.columns
    
    def test_different_thresholds(self, sample_ohlcv_data):
        """Test target generation with different threshold configurations"""
        # Test with very tight thresholds (should generate mostly hold signals)
        tight_config = {
            'method': 'next_period_return',
            'lookahead_periods': 1,
            'profit_threshold': 0.1,   # 10% - very high
            'loss_threshold': -0.1     # -10% - very low
        }
        
        generator = TargetGenerator(tight_config)
        result = generator.generate_targets(sample_ohlcv_data)
        
        # Should have mostly hold signals (0)
        distribution = generator.get_target_distribution(result)
        if 0 in distribution:
            assert distribution[0] > distribution.get(1, 0)
            assert distribution[0] > distribution.get(-1, 0)
        
        # Test with very loose thresholds (should generate more buy/sell signals)
        loose_config = {
            'method': 'next_period_return',
            'lookahead_periods': 1,
            'profit_threshold': 0.001,  # 0.1% - very low
            'loss_threshold': -0.001    # -0.1% - very high
        }
        
        generator_loose = TargetGenerator(loose_config)
        result_loose = generator_loose.generate_targets(sample_ohlcv_data)
        
        distribution_loose = generator_loose.get_target_distribution(result_loose)
        
        # Should have more buy/sell signals than tight thresholds
        buy_sell_signals_loose = distribution_loose.get(1, 0) + distribution_loose.get(-1, 0)
        buy_sell_signals_tight = distribution.get(1, 0) + distribution.get(-1, 0)
        
        assert buy_sell_signals_loose >= buy_sell_signals_tight
    
    def test_invalid_method(self, sample_ohlcv_data):
        """Test handling of invalid target generation method"""
        invalid_config = {
            'method': 'invalid_method',
            'lookahead_periods': 1,
            'profit_threshold': 0.01,
            'loss_threshold': -0.01
        }
        
        generator = TargetGenerator(invalid_config)
        
        with pytest.raises(ValueError, match="Unknown target generation method"):
            generator.generate_targets(sample_ohlcv_data)
    
    def test_missing_close_column(self, target_config_next_period):
        """Test handling of missing close column"""
        invalid_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'volume': [1000, 1100]
            # Missing 'close' column
        })
        
        generator = TargetGenerator(target_config_next_period)
        
        with pytest.raises(KeyError):
            generator.generate_targets(invalid_data)
    
    def test_target_consistency(self, sample_ohlcv_data, target_config_next_period):
        """Test that target generation is consistent across multiple runs"""
        generator = TargetGenerator(target_config_next_period)
        
        result1 = generator.generate_targets(sample_ohlcv_data)
        result2 = generator.generate_targets(sample_ohlcv_data)
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)


if __name__ == "__main__":
    pytest.main([__file__])