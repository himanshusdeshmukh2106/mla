"""
Custom exception hierarchy for trading system
"""


class TradingSystemError(Exception):
    """Base exception for trading system errors"""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class DataError(TradingSystemError):
    """Raised when data quality issues are detected"""
    pass


class ValidationError(DataError):
    """Raised when data validation fails"""
    pass


class DataLoadingError(DataError):
    """Raised when data loading fails"""
    pass


class FeatureEngineeringError(TradingSystemError):
    """Raised when feature engineering fails"""
    pass


class ModelError(TradingSystemError):
    """Raised when model training or prediction fails"""
    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails"""
    pass


class PredictionError(ModelError):
    """Raised when model prediction fails"""
    pass


class RiskError(TradingSystemError):
    """Raised when risk limits are violated"""
    pass


class PositionSizeError(RiskError):
    """Raised when position sizing fails"""
    pass


class RiskLimitError(RiskError):
    """Raised when risk limits are exceeded"""
    pass


class BacktestError(TradingSystemError):
    """Raised when backtesting encounters issues"""
    pass


class StrategyError(TradingSystemError):
    """Raised when strategy execution fails"""
    pass


class ConfigurationError(TradingSystemError):
    """Raised when configuration is invalid"""
    pass