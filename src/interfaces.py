"""
Base interfaces and abstract classes for all major components
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class TradingSignal:
    """Trading signal with confidence and metadata"""
    signal: int  # -1: sell, 0: hold, 1: buy
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    features: Dict[str, float]


@dataclass
class ValidationResult:
    """Data validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


@dataclass
class ModelMetrics:
    """Model evaluation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    feature_importance: Dict[str, float]


class IDataLoader(ABC):
    """Interface for data loading components"""
    
    @abstractmethod
    def load_ohlcv_data(self, symbol: str, timeframe: str, 
                       start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load OHLCV data for given parameters"""
        pass
    
    @abstractmethod
    def validate_data_integrity(self, data: pd.DataFrame) -> ValidationResult:
        """Validate data completeness and consistency"""
        pass
    
    @abstractmethod
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and outliers"""
        pass


class IFeatureEngineer(ABC):
    """Interface for feature engineering components"""
    
    @abstractmethod
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create all features from raw OHLCV data"""
        pass
    
    @abstractmethod
    def add_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based technical indicators"""
        pass
    
    @abstractmethod
    def add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based technical indicators"""
        pass


class IModelTrainer(ABC):
    """Interface for model training components"""
    
    @abstractmethod
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        pass
    
    @abstractmethod
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train the machine learning model"""
        pass
    
    @abstractmethod
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
        """Evaluate model performance"""
        pass


class IPredictor(ABC):
    """Interface for prediction components"""
    
    @abstractmethod
    def predict_signal(self, current_data: pd.DataFrame) -> TradingSignal:
        """Generate trading signal from current market data"""
        pass
    
    @abstractmethod
    def predict_probability(self, current_data: pd.DataFrame) -> float:
        """Get prediction probability"""
        pass


class IRiskManager(ABC):
    """Interface for risk management components"""
    
    @abstractmethod
    def calculate_position_size(self, signal: TradingSignal, portfolio_value: float) -> float:
        """Calculate appropriate position size"""
        pass
    
    @abstractmethod
    def check_risk_limits(self, current_positions: Dict, new_signal: TradingSignal) -> bool:
        """Check if new signal violates risk limits"""
        pass
    
    @abstractmethod
    def apply_stop_loss(self, position: Dict, current_price: float) -> bool:
        """Check if stop loss should be triggered"""
        pass


class IStrategyEngine(ABC):
    """Interface for trading strategy engines"""
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """Generate trading signal from market data"""
        pass
    
    @abstractmethod
    def init(self):
        """Initialize strategy (backtesting.py compatibility)"""
        pass
    
    @abstractmethod
    def next(self):
        """Process next data point (backtesting.py compatibility)"""
        pass


class IBacktestEngine(ABC):
    """Interface for backtesting engines"""
    
    @abstractmethod
    def run_backtest(self, initial_cash: float) -> Dict[str, Any]:
        """Run backtest simulation"""
        pass
    
    @abstractmethod
    def generate_performance_report(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        pass