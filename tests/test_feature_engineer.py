"""
Unit tests for FeatureEngineer class
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.engineer import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class"""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1min')
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic price data
        base_price = 100.0
        returns = np.random.normal(0, 0.01, 100)  # 1% volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Ensure high >= close >= low and high >= open >= low
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        return data.set_index('datetime')
    
    @pytest.fixture
    def feature_config(self):
        """Sample feature configuration"""
        return {
            'trend': {
                'sma_periods': [10, 20],
                'ema_periods': [12, 26]
            },
            'momentum': {
                'rsi_period': 14,
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'stochastic': {'k': 14, 'd': 3},
                'roc_period': 10
            },
            'volatility': {
                'bollinger_bands': {'period': 20, 'std': 2},
                'atr_period': 14,
                'volatility_period': 20
            },
            'volume': {
                'periods': [20],
                'roc_period': 10
            }
        }
    
    def test_feature_engineer_initialization(self, feature_config):
        """Test FeatureEngineer initialization"""
        engineer = FeatureEngineer(feature_config)
        
        assert engineer.config == feature_config
        assert engineer.trend_config == feature_config['trend']
        assert engineer.momentum_config == feature_config['momentum']
        assert engineer.volatility_config == feature_config['volatility']
        assert engineer.volume_config == feature_config['volume']
    
    def test_add_trend_indicators(self, sample_ohlcv_data, feature_config):
        """Test trend indicator calculations"""
        engineer = FeatureEngineer(feature_config)
        result = engineer.add_trend_indicators(sample_ohlcv_data)
        
        # Check SMA indicators
        assert 'SMA_10' in result.columns
        assert 'SMA_20' in result.columns
        
        # Check EMA indicators
        assert 'EMA_12' in result.columns
        assert 'EMA_26' in result.columns
        
        # Check ratio indicators
        assert 'Price_SMA_10_Ratio' in result.columns
        assert 'Price_EMA_12_Ratio' in result.columns
        
        # Verify SMA calculation for a known period
        manual_sma_10 = sample_ohlcv_data['close'].rolling(window=10).mean()
        pd.testing.assert_series_equal(
            result['SMA_10'].dropna(), 
            manual_sma_10.dropna(), 
            check_names=False,
            rtol=1e-10
        )
        
        # Verify price ratio calculation
        expected_ratio = sample_ohlcv_data['close'] / manual_sma_10
        pd.testing.assert_series_equal(
            result['Price_SMA_10_Ratio'].dropna(),
            expected_ratio.dropna(),
            check_names=False,
            rtol=1e-10
        )
    
    def test_add_momentum_indicators(self, sample_ohlcv_data, feature_config):
        """Test momentum indicator calculations"""
        engineer = FeatureEngineer(feature_config)
        result = engineer.add_momentum_indicators(sample_ohlcv_data)
        
        # Check RSI
        assert 'RSI' in result.columns
        assert result['RSI'].min() >= 0
        assert result['RSI'].max() <= 100
        
        # Check MACD components
        assert 'MACD' in result.columns
        assert 'MACD_Signal' in result.columns
        assert 'MACD_Histogram' in result.columns
        
        # Check Stochastic
        assert 'Stoch_K' in result.columns
        assert 'Stoch_D' in result.columns
        
        # Check ROC
        assert 'ROC' in result.columns
        
        # Verify RSI bounds
        rsi_values = result['RSI'].dropna()
        assert all(rsi_values >= 0) and all(rsi_values <= 100)
    
    def test_add_volatility_indicators(self, sample_ohlcv_data, feature_config):
        """Test volatility indicator calculations"""
        engineer = FeatureEngineer(feature_config)
        result = engineer.add_volatility_indicators(sample_ohlcv_data)
        
        # Check Bollinger Bands
        assert 'BB_Upper' in result.columns
        assert 'BB_Middle' in result.columns
        assert 'BB_Lower' in result.columns
        assert 'BB_Width' in result.columns
        assert 'BB_Position' in result.columns
        
        # Check ATR
        assert 'ATR' in result.columns
        assert all(result['ATR'].dropna() >= 0)  # ATR should be non-negative
        
        # Check volatility measures
        assert 'Volatility' in result.columns
        assert 'Volatility_Ratio' in result.columns
        
        # Verify Bollinger Band relationships
        bb_data = result[['BB_Upper', 'BB_Middle', 'BB_Lower']].dropna()
        assert all(bb_data['BB_Upper'] >= bb_data['BB_Middle'])
        assert all(bb_data['BB_Middle'] >= bb_data['BB_Lower'])
        
        # Verify BB_Position is between 0 and 1 (mostly)
        bb_position = result['BB_Position'].dropna()
        # Allow some values outside 0-1 range as prices can break out of bands
        assert bb_position.quantile(0.1) >= -0.5
        assert bb_position.quantile(0.9) <= 1.5
    
    def test_add_volume_indicators(self, sample_ohlcv_data, feature_config):
        """Test volume indicator calculations"""
        engineer = FeatureEngineer(feature_config)
        result = engineer.add_volume_indicators(sample_ohlcv_data)
        
        # Check OBV
        assert 'OBV' in result.columns
        
        # Check Volume SMA and ratios
        assert 'Volume_SMA_20' in result.columns
        assert 'Volume_Ratio_20' in result.columns
        
        # Check Volume ROC
        assert 'Volume_ROC' in result.columns
        
        # Check PVT
        assert 'PVT' in result.columns
        
        # Verify volume ratio calculation
        manual_vol_sma = sample_ohlcv_data['volume'].rolling(window=20).mean()
        expected_ratio = sample_ohlcv_data['volume'] / manual_vol_sma
        pd.testing.assert_series_equal(
            result['Volume_Ratio_20'].dropna(),
            expected_ratio.dropna(),
            check_names=False,
            rtol=1e-10
        )
    
    def test_create_features_integration(self, sample_ohlcv_data, feature_config):
        """Test complete feature creation process"""
        engineer = FeatureEngineer(feature_config)
        result = engineer.create_features(sample_ohlcv_data)
        
        # Check that original columns are preserved
        for col in sample_ohlcv_data.columns:
            assert col in result.columns
        
        # Check that no NaN rows remain
        assert result.isnull().sum().sum() == 0
        
        # Check that we have fewer rows due to NaN removal
        assert len(result) < len(sample_ohlcv_data)
        
        # Check that all expected feature types are present
        feature_names = result.columns.tolist()
        
        # Trend features
        assert any('SMA_' in name for name in feature_names)
        assert any('EMA_' in name for name in feature_names)
        
        # Momentum features
        assert 'RSI' in feature_names
        assert 'MACD' in feature_names
        
        # Volatility features
        assert 'ATR' in feature_names
        assert any('BB_' in name for name in feature_names)
        
        # Volume features
        assert 'OBV' in feature_names
        assert any('Volume_' in name for name in feature_names)
    
    def test_get_feature_names(self, feature_config):
        """Test feature name generation"""
        engineer = FeatureEngineer(feature_config)
        feature_names = engineer.get_feature_names()
        
        # Check that we get a list of strings
        assert isinstance(feature_names, list)
        assert all(isinstance(name, str) for name in feature_names)
        
        # Check for expected feature categories
        assert any('SMA_' in name for name in feature_names)
        assert any('EMA_' in name for name in feature_names)
        assert 'RSI' in feature_names
        assert 'MACD' in feature_names
        assert 'ATR' in feature_names
        assert 'OBV' in feature_names
    
    def test_empty_config_handling(self, sample_ohlcv_data):
        """Test handling of empty or minimal configuration"""
        minimal_config = {}
        engineer = FeatureEngineer(minimal_config)
        
        # Should not raise an exception
        result = engineer.create_features(sample_ohlcv_data)
        
        # Should still create some features with default parameters
        assert len(result.columns) > len(sample_ohlcv_data.columns)
    
    def test_invalid_data_handling(self, feature_config):
        """Test handling of invalid or insufficient data"""
        # Test with very small dataset
        small_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 1100]
        })
        
        engineer = FeatureEngineer(feature_config)
        
        # Should handle gracefully and return empty or minimal result
        result = engineer.create_features(small_data)
        
        # Result might be empty due to NaN removal, which is expected
        assert isinstance(result, pd.DataFrame)
    
    def test_feature_consistency(self, sample_ohlcv_data, feature_config):
        """Test that features are calculated consistently across multiple runs"""
        engineer = FeatureEngineer(feature_config)
        
        result1 = engineer.create_features(sample_ohlcv_data)
        result2 = engineer.create_features(sample_ohlcv_data)
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_select_features(self, sample_ohlcv_data, feature_config):
        """Test feature selection functionality"""
        engineer = FeatureEngineer(feature_config)
        full_data = engineer.create_features(sample_ohlcv_data)
        
        # Test selecting all features (excluding OHLCV)
        all_features = engineer.select_features(full_data)
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Should not contain OHLCV columns
        for col in ohlcv_cols:
            assert col not in all_features.columns
        
        # Should contain feature columns
        assert len(all_features.columns) > 0
        
        # Test selecting specific features
        specific_features = ['RSI', 'SMA_10', 'ATR']
        selected = engineer.select_features(full_data, specific_features)
        
        # Should only contain requested features that exist
        available_features = [col for col in specific_features if col in full_data.columns]
        assert list(selected.columns) == available_features
    
    def test_create_feature_combinations(self, sample_ohlcv_data, feature_config):
        """Test feature combination creation"""
        engineer = FeatureEngineer(feature_config)
        basic_features = engineer.create_features(sample_ohlcv_data)
        
        # Create combinations
        combined_features = engineer.create_feature_combinations(basic_features)
        
        # Should have more columns than basic features
        assert len(combined_features.columns) >= len(basic_features.columns)
        
        # Check for specific combination features
        expected_combinations = ['RSI_MACD_Signal', 'Trend_Alignment', 'BB_Breakout', 
                               'Volume_Confirmation', 'EMA_Cross']
        
        # At least some combinations should be present (depending on available base features)
        combination_count = sum(1 for col in expected_combinations if col in combined_features.columns)
        assert combination_count > 0
        
        # Combination features should be binary (0 or 1)
        for col in expected_combinations:
            if col in combined_features.columns:
                unique_values = combined_features[col].dropna().unique()
                assert all(val in [0, 1] for val in unique_values)


if __name__ == "__main__":
    pytest.main([__file__])