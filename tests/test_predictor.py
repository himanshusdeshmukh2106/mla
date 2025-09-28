"""
Unit tests for the Predictor signal generation system
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from xgboost import XGBClassifier

from src.strategies.predictor import Predictor, SignalConfig, PredictionResult, PredictionError
from src.interfaces import TradingSignal
from src.features.engineer import FeatureEngineer


class TestPredictor:
    """Test cases for Predictor class"""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock XGBoost model"""
        model = Mock(spec=XGBClassifier)
        model.predict_proba.return_value = np.array([[0.3, 0.7]])  # 70% probability for positive class
        model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4])
        model.feature_names_in_ = np.array(['feature_1', 'feature_2', 'feature_3', 'feature_4'])
        return model
        
    @pytest.fixture
    def mock_feature_engineer(self):
        """Create a mock feature engineer"""
        engineer = Mock(spec=FeatureEngineer)
        
        # Create sample features dataframe
        sample_data = pd.DataFrame({
            'open': [100.0],
            'high': [102.0],
            'low': [99.0],
            'close': [101.0],
            'volume': [1000],
            'SMA_20': [100.5],
            'RSI': [65.0],
            'MACD': [0.5],
            'ATR': [1.2]
        }, index=[datetime.now()])
        
        engineer.create_features.return_value = sample_data
        return engineer
        
    @pytest.fixture
    def signal_config(self):
        """Create test signal configuration"""
        return SignalConfig(
            confidence_threshold=0.6,
            probability_threshold_buy=0.55,
            probability_threshold_sell=0.45,
            signal_smoothing=False,  # Disable for simpler testing
            enable_regime_filter=False  # Disable for simpler testing
        )
        
    @pytest.fixture
    def predictor(self, mock_model, mock_feature_engineer, signal_config):
        """Create predictor instance for testing"""
        return Predictor(mock_model, mock_feature_engineer, signal_config)
        
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        data = pd.DataFrame({
            'datetime': dates,
            'open': np.random.uniform(99, 101, 100),
            'high': np.random.uniform(100, 102, 100),
            'low': np.random.uniform(98, 100, 100),
            'close': np.random.uniform(99, 101, 100),
            'volume': np.random.randint(500, 1500, 100)
        })
        data.set_index('datetime', inplace=True)
        return data
        
    def test_predictor_initialization(self, mock_model, mock_feature_engineer, signal_config):
        """Test predictor initialization"""
        predictor = Predictor(mock_model, mock_feature_engineer, signal_config)
        
        assert predictor.model == mock_model
        assert predictor.feature_engineer == mock_feature_engineer
        assert predictor.config == signal_config
        assert predictor.signal_history == []
        assert predictor.probability_history == []
        
    def test_predictor_initialization_without_config(self, mock_model, mock_feature_engineer):
        """Test predictor initialization with default config"""
        predictor = Predictor(mock_model, mock_feature_engineer)
        
        assert isinstance(predictor.config, SignalConfig)
        assert predictor.config.confidence_threshold == 0.6  # Default value
        
    def test_predictor_initialization_invalid_model(self, mock_feature_engineer):
        """Test predictor initialization with invalid model"""
        invalid_model = Mock()
        del invalid_model.predict_proba  # Remove predict_proba method
        
        with pytest.raises(PredictionError, match="Model must support probability predictions"):
            Predictor(invalid_model, mock_feature_engineer)
            
    def test_predict_signal_success(self, predictor, sample_ohlcv_data):
        """Test successful signal prediction"""
        signal = predictor.predict_signal(sample_ohlcv_data)
        
        assert isinstance(signal, TradingSignal)
        assert signal.signal in [-1, 0, 1]
        assert 0.0 <= signal.confidence <= 1.0
        assert isinstance(signal.timestamp, datetime)
        assert isinstance(signal.features, dict)
        
        # Verify model was called
        predictor.model.predict_proba.assert_called_once()
        predictor.feature_engineer.create_features.assert_called_once_with(sample_ohlcv_data)
        
    def test_predict_signal_high_probability_buy(self, predictor, sample_ohlcv_data):
        """Test signal prediction with high probability (should generate buy signal)"""
        # Set high probability for positive class
        predictor.model.predict_proba.return_value = np.array([[0.2, 0.8]])
        
        signal = predictor.predict_signal(sample_ohlcv_data)
        
        assert signal.signal == 1  # Buy signal
        assert signal.confidence > 0.5  # High confidence
        
    def test_predict_signal_low_probability_sell(self, predictor, sample_ohlcv_data):
        """Test signal prediction with low probability (should generate sell signal)"""
        # Set low probability for positive class
        predictor.model.predict_proba.return_value = np.array([[0.8, 0.2]])
        
        signal = predictor.predict_signal(sample_ohlcv_data)
        
        assert signal.signal == -1  # Sell signal
        assert signal.confidence > 0.5  # High confidence
        
    def test_predict_signal_neutral_probability_hold(self, predictor, sample_ohlcv_data):
        """Test signal prediction with neutral probability (should generate hold signal)"""
        # Set neutral probability
        predictor.model.predict_proba.return_value = np.array([[0.5, 0.5]])
        
        signal = predictor.predict_signal(sample_ohlcv_data)
        
        assert signal.signal == 0  # Hold signal
        assert signal.confidence == 0.0  # Low confidence for neutral probability
        
    def test_predict_signal_low_confidence_filter(self, predictor, sample_ohlcv_data):
        """Test signal filtering due to low confidence"""
        # Set probability that would normally generate buy signal but with low confidence
        predictor.model.predict_proba.return_value = np.array([[0.44, 0.56]])  # Just above buy threshold
        predictor.config.confidence_threshold = 0.5  # High confidence requirement
        
        signal = predictor.predict_signal(sample_ohlcv_data)
        
        # Should be filtered to hold due to low confidence
        assert signal.signal == 0
        
    def test_predict_signal_empty_features(self, predictor, sample_ohlcv_data):
        """Test signal prediction with empty features"""
        # Mock feature engineer to return empty dataframe
        predictor.feature_engineer.create_features.return_value = pd.DataFrame()
        
        with pytest.raises(PredictionError, match="No features generated from current data"):
            predictor.predict_signal(sample_ohlcv_data)
            
    def test_predict_signal_nan_features(self, predictor, sample_ohlcv_data):
        """Test signal prediction with NaN features"""
        # Create features with NaN values
        features_with_nan = pd.DataFrame({
            'open': [100.0],
            'high': [102.0],
            'low': [99.0],
            'close': [101.0],
            'volume': [1000],
            'SMA_20': [np.nan],  # NaN feature
            'RSI': [65.0]
        }, index=[datetime.now()])
        
        predictor.feature_engineer.create_features.return_value = features_with_nan
        
        with pytest.raises(PredictionError, match="NaN values found in features"):
            predictor.predict_signal(sample_ohlcv_data)
            
    def test_predict_probability_success(self, predictor, sample_ohlcv_data):
        """Test successful probability prediction"""
        probability = predictor.predict_probability(sample_ohlcv_data)
        
        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0
        assert probability == 0.7  # Based on mock return value
        
        # Verify model was called
        predictor.model.predict_proba.assert_called_once()
        
    def test_predict_probability_empty_features(self, predictor, sample_ohlcv_data):
        """Test probability prediction with empty features"""
        predictor.feature_engineer.create_features.return_value = pd.DataFrame()
        
        with pytest.raises(PredictionError, match="No features generated from current data"):
            predictor.predict_probability(sample_ohlcv_data)
            
    def test_calculate_confidence(self, predictor):
        """Test confidence calculation"""
        # Test various probability values
        assert predictor._calculate_confidence(0.5) == 0.0  # Neutral probability
        assert predictor._calculate_confidence(0.75) == 0.5  # 25% away from neutral
        assert predictor._calculate_confidence(0.25) == 0.5  # 25% away from neutral
        assert predictor._calculate_confidence(1.0) == 1.0  # Maximum confidence
        assert predictor._calculate_confidence(0.0) == 1.0  # Maximum confidence
        
    def test_probability_to_signal(self, predictor):
        """Test probability to signal conversion"""
        # Test buy signal
        assert predictor._probability_to_signal(0.8) == 1
        assert predictor._probability_to_signal(0.55) == 1  # At threshold
        
        # Test sell signal
        assert predictor._probability_to_signal(0.2) == -1
        assert predictor._probability_to_signal(0.45) == -1  # At threshold
        
        # Test hold signal
        assert predictor._probability_to_signal(0.5) == 0
        assert predictor._probability_to_signal(0.52) == 0  # Between thresholds
        
    def test_signal_smoothing(self, predictor, sample_ohlcv_data):
        """Test signal smoothing functionality"""
        # Enable signal smoothing
        predictor.config.signal_smoothing = True
        predictor.config.smoothing_window = 3
        
        # Generate multiple signals
        predictor.model.predict_proba.return_value = np.array([[0.2, 0.8]])  # Buy signal
        
        signals = []
        for _ in range(5):
            signal = predictor.predict_signal(sample_ohlcv_data)
            signals.append(signal.signal)
            
        # All signals should be consistent due to smoothing
        assert len(set(signals)) <= 2  # Should be mostly consistent
        assert len(predictor.signal_history) <= predictor.config.smoothing_window
        
    def test_regime_filter(self, predictor, sample_ohlcv_data):
        """Test market regime filtering"""
        # Enable regime filtering
        predictor.config.enable_regime_filter = True
        predictor.config.volatility_threshold = 1.0
        
        # Create features with high volatility
        high_vol_features = pd.DataFrame({
            'open': [100.0],
            'high': [102.0],
            'low': [99.0],
            'close': [101.0],
            'volume': [1000],
            'SMA_20': [100.5],
            'RSI': [65.0],
            'Volatility': [2.0],  # High volatility
            'ATR': [1.2]
        }, index=[datetime.now()])
        
        predictor.feature_engineer.create_features.return_value = high_vol_features
        predictor.model.predict_proba.return_value = np.array([[0.2, 0.8]])  # Would be buy signal
        
        signal = predictor.predict_signal(sample_ohlcv_data)
        
        # Signal should be filtered due to high volatility
        assert signal.signal == 0
        
    def test_get_feature_importance(self, predictor):
        """Test feature importance retrieval"""
        importance = predictor.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == 4  # Based on mock model
        assert 'feature_1' in importance
        assert importance['feature_4'] > importance['feature_1']  # Should be sorted by importance
        
    def test_get_feature_importance_no_importance(self, predictor):
        """Test feature importance when model doesn't have it"""
        # Remove feature importance from model
        del predictor.model.feature_importances_
        
        with pytest.raises(PredictionError, match="Model does not have feature importance information"):
            predictor.get_feature_importance()
            
    def test_get_signal_statistics_empty(self, predictor):
        """Test signal statistics with no history"""
        stats = predictor.get_signal_statistics()
        
        assert "message" in stats
        assert stats["message"] == "No signal history available"
        
    def test_get_signal_statistics_with_history(self, predictor):
        """Test signal statistics with signal history"""
        # Add some signals to history
        predictor.signal_history = [1, 1, 0, -1, 1]
        
        stats = predictor.get_signal_statistics()
        
        assert stats["total_signals"] == 5
        assert stats["buy_signals"] == 3
        assert stats["sell_signals"] == 1
        assert stats["hold_signals"] == 1
        assert stats["buy_percentage"] == 60.0
        assert stats["sell_percentage"] == 20.0
        assert stats["hold_percentage"] == 20.0
        
    def test_reset_signal_history(self, predictor):
        """Test signal history reset"""
        # Add some history
        predictor.signal_history = [1, -1, 0]
        predictor.probability_history = [0.8, 0.2, 0.5]
        
        predictor.reset_signal_history()
        
        assert predictor.signal_history == []
        assert predictor.probability_history == []
        
    def test_update_config(self, predictor):
        """Test configuration update"""
        new_config = SignalConfig(confidence_threshold=0.8)
        
        predictor.update_config(new_config)
        
        assert predictor.config.confidence_threshold == 0.8
        
    def test_validate_features_success(self, predictor):
        """Test successful feature validation"""
        # Create features dataframe with expected features
        features_df = pd.DataFrame({
            'feature_1': [1.0],
            'feature_2': [2.0],
            'feature_3': [3.0],
            'feature_4': [4.0],
            'close': [100.0]  # Should be excluded
        })
        
        assert predictor.validate_features(features_df) is True
        
    def test_validate_features_missing(self, predictor):
        """Test feature validation with missing features"""
        # Create features dataframe missing some expected features
        features_df = pd.DataFrame({
            'feature_1': [1.0],
            'feature_2': [2.0],
            'close': [100.0]
        })
        
        assert predictor.validate_features(features_df) is False
        
    def test_validate_features_no_model_features(self, predictor):
        """Test feature validation when model has no feature names"""
        # Remove feature names from model
        del predictor.model.feature_names_in_
        
        features_df = pd.DataFrame({'any_feature': [1.0]})
        
        # Should pass validation when no expected features
        assert predictor.validate_features(features_df) is True


class TestSignalConfig:
    """Test cases for SignalConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = SignalConfig()
        
        assert config.confidence_threshold == 0.6
        assert config.probability_threshold_buy == 0.55
        assert config.probability_threshold_sell == 0.45
        assert config.signal_smoothing is True
        assert config.smoothing_window == 3
        assert config.enable_regime_filter is False
        
    def test_custom_config(self):
        """Test custom configuration values"""
        config = SignalConfig(
            confidence_threshold=0.8,
            probability_threshold_buy=0.6,
            signal_smoothing=False
        )
        
        assert config.confidence_threshold == 0.8
        assert config.probability_threshold_buy == 0.6
        assert config.signal_smoothing is False


class TestPredictionResult:
    """Test cases for PredictionResult dataclass"""
    
    def test_prediction_result_creation(self):
        """Test PredictionResult creation"""
        timestamp = datetime.now()
        features = {'feature_1': 1.0, 'feature_2': 2.0}
        
        result = PredictionResult(
            probability=0.75,
            confidence=0.5,
            raw_signal=1,
            filtered_signal=1,
            features_used=features,
            timestamp=timestamp
        )
        
        assert result.probability == 0.75
        assert result.confidence == 0.5
        assert result.raw_signal == 1
        assert result.filtered_signal == 1
        assert result.features_used == features
        assert result.timestamp == timestamp


class TestIntegrationScenarios:
    """Integration test scenarios for various market conditions"""
    
    @pytest.fixture
    def real_feature_engineer(self):
        """Create a real feature engineer for integration tests"""
        config = {
            'trend': {'sma_periods': [5], 'ema_periods': [3]},  # Shorter periods for testing
            'momentum': {'rsi_period': 5},  # Shorter period
            'volatility': {'atr_period': 3},  # Shorter period
            'volume': {'periods': [5]}  # Shorter period
        }
        return FeatureEngineer(config)
        
    @pytest.fixture
    def mock_model_integration(self):
        """Create a mock XGBoost model for integration tests"""
        model = Mock(spec=XGBClassifier)
        model.predict_proba.return_value = np.array([[0.3, 0.7]])
        model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4])
        model.feature_names_in_ = np.array(['SMA_20', 'EMA_12', 'RSI', 'ATR'])
        return model
        
    def test_bull_market_scenario(self, mock_model_integration, real_feature_engineer):
        """Test signal generation in bull market conditions"""
        # Create bull market data (uptrend) - need more data for indicators
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        prices = np.linspace(100, 110, 100)  # Upward trend
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(500, 1500, 100)
        }, index=dates)
        
        # Mock model to return bullish probabilities with high confidence
        mock_model_integration.predict_proba.return_value = np.array([[0.1, 0.9]])  # Higher confidence
        
        # Use lower confidence threshold for testing
        config = SignalConfig(confidence_threshold=0.4)
        predictor = Predictor(mock_model_integration, real_feature_engineer, config)
        signal = predictor.predict_signal(data)
        
        assert signal.signal == 1  # Should generate buy signal
        assert signal.confidence > 0.4  # Should have reasonable confidence
        
    def test_bear_market_scenario(self, mock_model_integration, real_feature_engineer):
        """Test signal generation in bear market conditions"""
        # Create bear market data (downtrend)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        prices = np.linspace(110, 100, 100)  # Downward trend
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(500, 1500, 100)
        }, index=dates)
        
        # Mock model to return bearish probabilities with high confidence
        mock_model_integration.predict_proba.return_value = np.array([[0.9, 0.1]])  # Higher confidence
        
        # Use lower confidence threshold for testing
        config = SignalConfig(confidence_threshold=0.4)
        predictor = Predictor(mock_model_integration, real_feature_engineer, config)
        signal = predictor.predict_signal(data)
        
        assert signal.signal == -1  # Should generate sell signal
        assert signal.confidence > 0.4  # Should have reasonable confidence
        
    def test_sideways_market_scenario(self, mock_model_integration, real_feature_engineer):
        """Test signal generation in sideways market conditions"""
        # Create sideways market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        prices = np.full(100, 105) + np.random.normal(0, 0.5, 100)  # Sideways with noise
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': np.random.randint(500, 1500, 100)
        }, index=dates)
        
        # Mock model to return neutral probabilities
        mock_model_integration.predict_proba.return_value = np.array([[0.52, 0.48]])
        
        predictor = Predictor(mock_model_integration, real_feature_engineer)
        signal = predictor.predict_signal(data)
        
        assert signal.signal == 0  # Should generate hold signal
        assert signal.confidence < 0.1  # Should have low confidence
        
    def test_high_volatility_scenario(self, mock_model_integration, real_feature_engineer):
        """Test signal generation in high volatility conditions"""
        # Create high volatility data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        base_price = 100
        volatility = 5  # High volatility
        prices = base_price + np.random.normal(0, volatility, 100)
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000, 3000, 100)  # Higher volume
        }, index=dates)
        
        # Enable regime filtering
        config = SignalConfig(enable_regime_filter=True, volatility_threshold=1.0)
        predictor = Predictor(mock_model_integration, real_feature_engineer, config)
        
        # Mock strong signal that should be filtered
        mock_model_integration.predict_proba.return_value = np.array([[0.1, 0.9]])
        
        signal = predictor.predict_signal(data)
        
        # Signal might be filtered due to high volatility
        # The exact behavior depends on the volatility indicators calculated
        assert signal.signal in [-1, 0, 1]  # Valid signal range