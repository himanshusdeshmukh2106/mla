"""
Test suite for base interfaces and core functionality
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from interfaces import TradingSignal, ValidationResult, ModelMetrics
from config_manager import ConfigManager, DataConfig, FeatureConfig
from logger import TradingLogger
from exceptions import TradingSystemError, DataError, ModelError


class TestDataStructures:
    """Test core data structures"""
    
    def test_trading_signal_creation(self):
        """Test TradingSignal dataclass"""
        signal = TradingSignal(
            signal=1,
            confidence=0.85,
            timestamp=datetime.now(),
            features={"rsi": 30.5, "macd": 0.02}
        )
        
        assert signal.signal == 1
        assert signal.confidence == 0.85
        assert isinstance(signal.timestamp, datetime)
        assert signal.features["rsi"] == 30.5
    
    def test_validation_result_creation(self):
        """Test ValidationResult dataclass"""
        result = ValidationResult(
            is_valid=False,
            errors=["Missing data"],
            warnings=["Low quality data"]
        )
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert len(result.warnings) == 1


class TestConfigManager:
    """Test configuration management"""
    
    def test_config_manager_initialization(self):
        """Test ConfigManager creates default configs"""
        config_manager = ConfigManager("test_config")
        
        # Test that default configs are created
        data_config = config_manager.get_data_config()
        assert isinstance(data_config, DataConfig)
        assert data_config.timeframe == "1min"
        assert "AAPL" in data_config.symbols
    
    def test_feature_config_retrieval(self):
        """Test feature configuration retrieval"""
        config_manager = ConfigManager("test_config")
        feature_config = config_manager.get_feature_config()
        
        assert isinstance(feature_config, FeatureConfig)
        assert 20 in feature_config.trend_periods
        assert feature_config.momentum_periods["rsi"] == 14


class TestLogger:
    """Test logging functionality"""
    
    def test_logger_creation(self):
        """Test logger initialization"""
        logger = TradingLogger("test_logger", "test_logs")
        
        assert logger.name == "test_logger"
        assert logger.log_dir.name == "test_logs"
    
    def test_logging_methods(self):
        """Test different logging methods"""
        logger = TradingLogger("test_logger", "test_logs")
        
        # These should not raise exceptions
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        logger.debug("Test debug message")


class TestExceptions:
    """Test custom exception hierarchy"""
    
    def test_base_exception(self):
        """Test base TradingSystemError"""
        error = TradingSystemError("Test error", {"detail": "test"})
        
        assert str(error) == "Test error - Details: {'detail': 'test'}"
        assert error.message == "Test error"
        assert error.details == {"detail": "test"}
    
    def test_specific_exceptions(self):
        """Test specific exception types"""
        data_error = DataError("Data problem")
        model_error = ModelError("Model problem")
        
        assert isinstance(data_error, TradingSystemError)
        assert isinstance(model_error, TradingSystemError)
        assert data_error.message == "Data problem"
        assert model_error.message == "Model problem"


if __name__ == "__main__":
    pytest.main([__file__])