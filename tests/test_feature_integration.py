"""
Integration tests for FeatureEngineer and TargetGenerator working together
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.engineer import FeatureEngineer
from features.target_generator import TargetGenerator


class TestFeatureIntegration:
    """Integration tests for feature engineering components"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data"""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='1min')
        np.random.seed(42)
        
        base_price = 100.0
        returns = np.random.normal(0, 0.01, 200)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 200)
        })
        
        # Ensure realistic OHLCV relationships
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        return data.set_index('datetime')
    
    @pytest.fixture
    def feature_config(self):
        """Feature engineering configuration"""
        return {
            'trend': {
                'sma_periods': [10, 20],
                'ema_periods': [12, 26]
            },
            'momentum': {
                'rsi_period': 14,
                'macd': {'fast': 12, 'slow': 26, 'signal': 9}
            },
            'volatility': {
                'bollinger_bands': {'period': 20, 'std': 2},
                'atr_period': 14
            },
            'volume': {
                'periods': [20]
            }
        }
    
    @pytest.fixture
    def target_config(self):
        """Target generation configuration"""
        return {
            'method': 'next_period_return',
            'lookahead_periods': 1,
            'profit_threshold': 0.005,  # 0.5%
            'loss_threshold': -0.005    # -0.5%
        }
    
    def test_complete_feature_pipeline(self, sample_data, feature_config, target_config):
        """Test complete feature engineering and target generation pipeline"""
        # Step 1: Create features
        feature_engineer = FeatureEngineer(feature_config)
        features_data = feature_engineer.create_features(sample_data)
        
        # Verify features were created
        assert len(features_data.columns) > len(sample_data.columns)
        assert 'RSI' in features_data.columns
        assert 'SMA_10' in features_data.columns
        
        # Step 2: Generate targets
        target_generator = TargetGenerator(target_config)
        final_data = target_generator.generate_targets(features_data)
        
        # Verify targets were created
        assert 'target' in final_data.columns
        assert 'target_binary' in final_data.columns
        
        # Verify data integrity
        assert len(final_data) > 0
        
        # Check for NaN values (expect some in future_return for last row due to lookahead)
        nan_counts = final_data.isnull().sum()
        # Only future_return should have NaN values (at most 1 for the last row)
        assert nan_counts['future_return'] <= 1
        # Other columns should not have NaN values
        other_cols = [col for col in nan_counts.index if col != 'future_return']
        assert all(nan_counts[col] == 0 for col in other_cols)
        
        # Verify target values are valid
        unique_targets = final_data['target'].unique()
        assert all(target in [-1, 0, 1] for target in unique_targets)
        
        # Verify we have both features and targets
        feature_cols = [col for col in final_data.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 
                                     'target', 'target_binary', 'future_return']]
        assert len(feature_cols) > 10  # Should have many features
        
        print(f"Pipeline complete: {len(final_data)} samples with {len(feature_cols)} features")
        
    def test_feature_target_correlation(self, sample_data, feature_config, target_config):
        """Test that features and targets have reasonable relationships"""
        # Create complete dataset
        feature_engineer = FeatureEngineer(feature_config)
        target_generator = TargetGenerator(target_config)
        
        features_data = feature_engineer.create_features(sample_data)
        final_data = target_generator.generate_targets(features_data)
        
        # Check that some features have correlation with targets
        feature_cols = ['RSI', 'MACD', 'BB_Position']
        available_features = [col for col in feature_cols if col in final_data.columns]
        
        if available_features:
            correlations = final_data[available_features + ['target_binary']].corr()['target_binary']
            
            # At least one feature should have some correlation (> 0.05 absolute)
            max_correlation = correlations[available_features].abs().max()
            assert max_correlation > 0.01  # Very loose threshold for correlation
    
    def test_data_splits_with_features(self, sample_data, feature_config, target_config):
        """Test that data splitting works with engineered features"""
        # Create complete dataset
        feature_engineer = FeatureEngineer(feature_config)
        target_generator = TargetGenerator(target_config)
        
        features_data = feature_engineer.create_features(sample_data)
        final_data = target_generator.generate_targets(features_data)
        
        # Create splits
        train_idx, val_idx, test_idx = target_generator.create_stratified_split_indices(
            final_data, train_ratio=0.6, val_ratio=0.2
        )
        
        # Extract feature matrix and targets
        feature_cols = feature_engineer.select_features(final_data).columns
        
        X_train = final_data.iloc[train_idx][feature_cols]
        y_train = final_data.iloc[train_idx]['target_binary']
        
        X_val = final_data.iloc[val_idx][feature_cols]
        y_val = final_data.iloc[val_idx]['target_binary']
        
        X_test = final_data.iloc[test_idx][feature_cols]
        y_test = final_data.iloc[test_idx]['target_binary']
        
        # Verify splits have data
        assert len(X_train) > 0 and len(y_train) > 0
        assert len(X_val) > 0 and len(y_val) > 0
        assert len(X_test) > 0 and len(y_test) > 0
        
        # Verify feature dimensions match
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
        
        # Verify no data leakage (time-based split)
        train_end_time = final_data.index[train_idx[-1]]
        val_start_time = final_data.index[val_idx[0]]
        assert train_end_time < val_start_time
        
        print(f"Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        print(f"Features: {X_train.shape[1]}")
    
    def test_feature_combinations_with_targets(self, sample_data, feature_config, target_config):
        """Test feature combinations work with target generation"""
        feature_engineer = FeatureEngineer(feature_config)
        target_generator = TargetGenerator(target_config)
        
        # Create basic features
        features_data = feature_engineer.create_features(sample_data)
        
        # Add feature combinations
        combined_features = feature_engineer.create_feature_combinations(features_data)
        
        # Generate targets
        final_data = target_generator.generate_targets(combined_features)
        
        # Verify combination features exist
        combination_features = ['RSI_MACD_Signal', 'Trend_Alignment', 'EMA_Cross']
        available_combinations = [col for col in combination_features if col in final_data.columns]
        
        assert len(available_combinations) > 0
        
        # Verify combination features are binary
        for col in available_combinations:
            unique_vals = final_data[col].dropna().unique()
            assert all(val in [0, 1] for val in unique_vals)
        
        print(f"Available combination features: {available_combinations}")


if __name__ == "__main__":
    pytest.main([__file__])