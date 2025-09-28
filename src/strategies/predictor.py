"""
Signal generation system with model-based predictions and confidence scoring
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

from xgboost import XGBClassifier
from ..interfaces import IPredictor, TradingSignal
from ..features.engineer import FeatureEngineer
from ..exceptions import TradingSystemError


@dataclass
class SignalConfig:
    """Configuration for signal generation"""
    confidence_threshold: float = 0.6  # Minimum confidence for signal generation
    probability_threshold_buy: float = 0.55  # Probability threshold for buy signals
    probability_threshold_sell: float = 0.45  # Probability threshold for sell signals
    signal_smoothing: bool = True  # Apply signal smoothing
    smoothing_window: int = 3  # Window size for signal smoothing
    max_signal_strength: float = 1.0  # Maximum signal strength
    min_signal_strength: float = 0.1  # Minimum signal strength
    enable_regime_filter: bool = False  # Enable market regime filtering
    volatility_threshold: float = 2.0  # Volatility threshold for regime filtering


@dataclass
class PredictionResult:
    """Result of model prediction"""
    probability: float  # Probability of positive class (price increase)
    confidence: float  # Confidence score (0.0 to 1.0)
    raw_signal: int  # Raw signal before filtering (-1, 0, 1)
    filtered_signal: int  # Final signal after filtering (-1, 0, 1)
    features_used: Dict[str, float]  # Features used for prediction
    timestamp: datetime  # Prediction timestamp


class PredictionError(TradingSystemError):
    """Raised when prediction fails"""
    pass


class Predictor(IPredictor):
    """
    Signal generation system with model-based predictions and confidence scoring
    """
    
    def __init__(self, 
                 model: XGBClassifier,
                 feature_engineer: FeatureEngineer,
                 config: Optional[SignalConfig] = None):
        """
        Initialize predictor with trained model and feature engineer
        
        Args:
            model: Trained XGBoost model
            feature_engineer: Feature engineering instance
            config: Signal generation configuration
        """
        self.model = model
        self.feature_engineer = feature_engineer
        self.config = config or SignalConfig()
        self.logger = logging.getLogger(__name__)
        
        # Signal history for smoothing
        self.signal_history: List[int] = []
        self.probability_history: List[float] = []
        
        # Validate model
        if not hasattr(model, 'predict_proba'):
            raise PredictionError("Model must support probability predictions")
            
        self.logger.info("Predictor initialized with signal generation system")
        
    def predict_signal(self, current_data: pd.DataFrame) -> TradingSignal:
        """
        Generate trading signal from current market data
        
        Args:
            current_data: DataFrame with current OHLCV data
            
        Returns:
            TradingSignal with signal, confidence, and metadata
        """
        try:
            # Generate features for current data
            features_df = self.feature_engineer.create_features(current_data)
            
            if features_df.empty:
                raise PredictionError("No features generated from current data")
                
            # Get the latest row with features
            latest_features = features_df.iloc[-1:].copy()
            
            # Remove non-feature columns
            exclude_columns = ['open', 'high', 'low', 'close', 'volume', 'datetime', 'timestamp', 'target']
            feature_columns = [col for col in latest_features.columns if col not in exclude_columns]
            
            if not feature_columns:
                raise PredictionError("No feature columns available for prediction")
                
            # Extract feature values
            X = latest_features[feature_columns].values
            
            # Check for NaN values
            if np.isnan(X).any():
                raise PredictionError("NaN values found in features")
                
            # Generate prediction
            prediction_result = self._generate_prediction(X[0], feature_columns, latest_features.index[0])
            
            # Create features dictionary for signal
            features_dict = dict(zip(feature_columns, X[0]))
            
            # Create trading signal
            signal = TradingSignal(
                signal=prediction_result.filtered_signal,
                confidence=prediction_result.confidence,
                timestamp=prediction_result.timestamp,
                features=features_dict
            )
            
            self.logger.debug(f"Generated signal: {signal.signal} with confidence {signal.confidence:.3f}")
            
            return signal
            
        except Exception as e:
            raise PredictionError(f"Signal prediction failed: {str(e)}")
            
    def predict_probability(self, current_data: pd.DataFrame) -> float:
        """
        Get prediction probability for current market data
        
        Args:
            current_data: DataFrame with current OHLCV data
            
        Returns:
            Probability of positive class (price increase)
        """
        try:
            # Generate features
            features_df = self.feature_engineer.create_features(current_data)
            
            if features_df.empty:
                raise PredictionError("No features generated from current data")
                
            # Get latest features
            latest_features = features_df.iloc[-1:].copy()
            
            # Remove non-feature columns
            exclude_columns = ['open', 'high', 'low', 'close', 'volume', 'datetime', 'timestamp', 'target']
            feature_columns = [col for col in latest_features.columns if col not in exclude_columns]
            
            X = latest_features[feature_columns].values
            
            # Generate probability
            probabilities = self.model.predict_proba(X)
            
            # Return probability of positive class (index 1)
            return float(probabilities[0][1])
            
        except Exception as e:
            raise PredictionError(f"Probability prediction failed: {str(e)}")
            
    def _generate_prediction(self, 
                           features: np.ndarray, 
                           feature_names: List[str],
                           timestamp_index: Any) -> PredictionResult:
        """
        Generate prediction with confidence scoring and filtering
        
        Args:
            features: Feature array for prediction
            feature_names: Names of features
            timestamp_index: Timestamp index from data
            
        Returns:
            PredictionResult with detailed prediction information
        """
        # Get model predictions
        probabilities = self.model.predict_proba(features.reshape(1, -1))
        probability = float(probabilities[0][1])  # Probability of positive class
        
        # Calculate confidence score
        confidence = self._calculate_confidence(probability)
        
        # Generate raw signal based on probability thresholds
        raw_signal = self._probability_to_signal(probability)
        
        # Apply signal filtering
        filtered_signal = self._apply_signal_filters(raw_signal, probability, features, feature_names)
        
        # Create features dictionary
        features_dict = dict(zip(feature_names, features))
        
        # Create timestamp
        if hasattr(timestamp_index, 'to_pydatetime'):
            timestamp = timestamp_index.to_pydatetime()
        else:
            timestamp = datetime.now()
            
        return PredictionResult(
            probability=probability,
            confidence=confidence,
            raw_signal=raw_signal,
            filtered_signal=filtered_signal,
            features_used=features_dict,
            timestamp=timestamp
        )
        
    def _calculate_confidence(self, probability: float) -> float:
        """
        Calculate confidence score based on probability
        
        Args:
            probability: Model probability output
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Confidence is based on how far the probability is from 0.5 (neutral)
        # Scale to 0-1 range where 1.0 means maximum confidence
        distance_from_neutral = abs(probability - 0.5)
        confidence = min(distance_from_neutral * 2.0, 1.0)
        
        return confidence
        
    def _probability_to_signal(self, probability: float) -> int:
        """
        Convert probability to raw trading signal
        
        Args:
            probability: Model probability output
            
        Returns:
            Raw signal: 1 (buy), 0 (hold), -1 (sell)
        """
        if probability >= self.config.probability_threshold_buy:
            return 1  # Buy signal
        elif probability <= self.config.probability_threshold_sell:
            return -1  # Sell signal
        else:
            return 0  # Hold signal
            
    def _apply_signal_filters(self, 
                            raw_signal: int, 
                            probability: float,
                            features: np.ndarray,
                            feature_names: List[str]) -> int:
        """
        Apply various filters to the raw signal
        
        Args:
            raw_signal: Raw signal from probability thresholds
            probability: Model probability
            features: Feature array
            feature_names: Feature names
            
        Returns:
            Filtered signal
        """
        # Start with raw signal
        filtered_signal = raw_signal
        
        # Apply confidence threshold filter
        confidence = self._calculate_confidence(probability)
        if confidence < self.config.confidence_threshold:
            filtered_signal = 0  # No signal if confidence too low
            
        # Apply market regime filter if enabled
        if self.config.enable_regime_filter:
            filtered_signal = self._apply_regime_filter(filtered_signal, features, feature_names)
            
        # Apply signal smoothing if enabled
        if self.config.signal_smoothing:
            filtered_signal = self._apply_signal_smoothing(filtered_signal)
            
        return filtered_signal
        
    def _apply_regime_filter(self, 
                           signal: int, 
                           features: np.ndarray,
                           feature_names: List[str]) -> int:
        """
        Apply market regime filtering based on volatility and trend
        
        Args:
            signal: Current signal
            features: Feature array
            feature_names: Feature names
            
        Returns:
            Regime-filtered signal
        """
        try:
            # Find volatility-related features
            volatility_features = [name for name in feature_names if 'Volatility' in name or 'ATR' in name]
            
            if volatility_features:
                # Get volatility feature index
                vol_idx = feature_names.index(volatility_features[0])
                current_volatility = features[vol_idx]
                
                # If volatility is too high, reduce signal strength or eliminate
                if current_volatility > self.config.volatility_threshold:
                    if abs(signal) > 0:
                        # Reduce signal strength in high volatility
                        signal = 0
                        self.logger.debug(f"Signal filtered due to high volatility: {current_volatility}")
                        
            return signal
            
        except (IndexError, ValueError):
            # If regime filtering fails, return original signal
            self.logger.warning("Regime filtering failed, using original signal")
            return signal
            
    def _apply_signal_smoothing(self, signal: int) -> int:
        """
        Apply signal smoothing using historical signals
        
        Args:
            signal: Current signal
            
        Returns:
            Smoothed signal
        """
        # Add current signal to history
        self.signal_history.append(signal)
        
        # Keep only recent history
        if len(self.signal_history) > self.config.smoothing_window:
            self.signal_history = self.signal_history[-self.config.smoothing_window:]
            
        # If we don't have enough history, return current signal
        if len(self.signal_history) < self.config.smoothing_window:
            return signal
            
        # Calculate smoothed signal as majority vote
        signal_sum = sum(self.signal_history)
        window_size = len(self.signal_history)
        
        # Determine smoothed signal based on average
        if signal_sum > window_size * 0.3:
            smoothed_signal = 1
        elif signal_sum < -window_size * 0.3:
            smoothed_signal = -1
        else:
            smoothed_signal = 0
            
        return smoothed_signal
        
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise PredictionError("Model does not have feature importance information")
            
        # Get feature names from the model or use generic names
        if hasattr(self.model, 'feature_names_in_'):
            feature_names = self.model.feature_names_in_.tolist()
        else:
            # Use generic feature names
            n_features = len(self.model.feature_importances_)
            feature_names = [f'feature_{i}' for i in range(n_features)]
            
        # Create importance dictionary
        importance_dict = dict(zip(feature_names, self.model.feature_importances_.tolist()))
        
        # Sort by importance (descending)
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
        
    def get_signal_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about recent signal generation
        
        Returns:
            Dictionary with signal statistics
        """
        if not self.signal_history:
            return {"message": "No signal history available"}
            
        total_signals = len(self.signal_history)
        buy_signals = sum(1 for s in self.signal_history if s == 1)
        sell_signals = sum(1 for s in self.signal_history if s == -1)
        hold_signals = sum(1 for s in self.signal_history if s == 0)
        
        return {
            "total_signals": total_signals,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "hold_signals": hold_signals,
            "buy_percentage": (buy_signals / total_signals) * 100 if total_signals > 0 else 0,
            "sell_percentage": (sell_signals / total_signals) * 100 if total_signals > 0 else 0,
            "hold_percentage": (hold_signals / total_signals) * 100 if total_signals > 0 else 0,
            "recent_signals": self.signal_history[-10:] if len(self.signal_history) >= 10 else self.signal_history
        }
        
    def reset_signal_history(self):
        """Reset signal history for fresh start"""
        self.signal_history.clear()
        self.probability_history.clear()
        self.logger.info("Signal history reset")
        
    def update_config(self, new_config: SignalConfig):
        """
        Update signal generation configuration
        
        Args:
            new_config: New configuration to apply
        """
        self.config = new_config
        self.logger.info("Signal generation configuration updated")
        
    def validate_features(self, features_df: pd.DataFrame) -> bool:
        """
        Validate that required features are present in the dataframe
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            True if all required features are present
        """
        try:
            # Get expected feature names from model
            if hasattr(self.model, 'feature_names_in_'):
                expected_features = set(self.model.feature_names_in_)
            else:
                # If no feature names available, assume validation passes
                return True
                
            # Get available features (excluding OHLCV and metadata columns)
            exclude_columns = ['open', 'high', 'low', 'close', 'volume', 'datetime', 'timestamp', 'target']
            available_features = set(col for col in features_df.columns if col not in exclude_columns)
            
            # Check if all expected features are available
            missing_features = expected_features - available_features
            
            if missing_features:
                self.logger.error(f"Missing required features: {missing_features}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Feature validation failed: {str(e)}")
            return False