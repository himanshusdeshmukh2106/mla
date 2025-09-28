"""
Feature engineering implementation with technical indicators
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional
from src.interfaces import IFeatureEngineer
from src.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer(IFeatureEngineer):
    """
    Feature engineering class that creates technical indicators from OHLCV data
    """
    
    def __init__(self, config: Dict):
        """
        Initialize FeatureEngineer with configuration
        
        Args:
            config: Dictionary containing feature configuration parameters
        """
        self.config = config
        self.trend_config = config.get('trend', {})
        self.momentum_config = config.get('momentum', {})
        self.volatility_config = config.get('volatility', {})
        self.volume_config = config.get('volume', {})
        
        logger.info("FeatureEngineer initialized with config")
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from raw OHLCV data
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with original data plus all technical indicators
        """
        logger.info(f"Creating features for {len(data)} data points")
        
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Add all indicator types
        df = self.add_trend_indicators(df)
        df = self.add_momentum_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_volume_indicators(df)
        
        # Remove rows with NaN values created by indicators
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        logger.info(f"Features created. Rows: {initial_rows} -> {final_rows} (removed {initial_rows - final_rows} NaN rows)")
        
        return df
    
    def add_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend-based technical indicators (SMA, EMA)
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with trend indicators added
        """
        df = data.copy()
        
        # Simple Moving Averages
        sma_periods = self.trend_config.get('sma_periods', [20, 50, 200])
        for period in sma_periods:
            df[f'SMA_{period}'] = ta.sma(df['close'], length=period)
            logger.debug(f"Added SMA_{period}")
        
        # Exponential Moving Averages
        ema_periods = self.trend_config.get('ema_periods', [12, 26, 50])
        for period in ema_periods:
            df[f'EMA_{period}'] = ta.ema(df['close'], length=period)
            logger.debug(f"Added EMA_{period}")
        
        # Price relative to moving averages
        if sma_periods:
            primary_sma = sma_periods[0]
            df[f'Price_SMA_{primary_sma}_Ratio'] = df['close'] / df[f'SMA_{primary_sma}']
        
        if ema_periods:
            primary_ema = ema_periods[0]
            df[f'Price_EMA_{primary_ema}_Ratio'] = df['close'] / df[f'EMA_{primary_ema}']
        
        logger.info(f"Added trend indicators: SMA periods {sma_periods}, EMA periods {ema_periods}")
        
        return df
    
    def add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum-based technical indicators (RSI, MACD, Stochastic)
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum indicators added
        """
        df = data.copy()
        
        # RSI (Relative Strength Index)
        rsi_period = self.momentum_config.get('rsi_period', 14)
        df['RSI'] = ta.rsi(df['close'], length=rsi_period)
        logger.debug(f"Added RSI with period {rsi_period}")
        
        # MACD (Moving Average Convergence Divergence)
        macd_config = self.momentum_config.get('macd', {'fast': 12, 'slow': 26, 'signal': 9})
        macd_result = ta.macd(
            df['close'], 
            fast=macd_config['fast'], 
            slow=macd_config['slow'], 
            signal=macd_config['signal']
        )
        
        if macd_result is not None:
            df['MACD'] = macd_result[f"MACD_{macd_config['fast']}_{macd_config['slow']}_{macd_config['signal']}"]
            df['MACD_Signal'] = macd_result[f"MACDs_{macd_config['fast']}_{macd_config['slow']}_{macd_config['signal']}"]
            df['MACD_Histogram'] = macd_result[f"MACDh_{macd_config['fast']}_{macd_config['slow']}_{macd_config['signal']}"]
            logger.debug(f"Added MACD with config {macd_config}")
        
        # Stochastic Oscillator
        stoch_config = self.momentum_config.get('stochastic', {'k': 14, 'd': 3})
        stoch_result = ta.stoch(
            df['high'], 
            df['low'], 
            df['close'], 
            k=stoch_config['k'], 
            d=stoch_config['d']
        )
        
        if stoch_result is not None:
            df['Stoch_K'] = stoch_result[f"STOCHk_{stoch_config['k']}_{stoch_config['d']}_3"]
            df['Stoch_D'] = stoch_result[f"STOCHd_{stoch_config['k']}_{stoch_config['d']}_3"]
            logger.debug(f"Added Stochastic with config {stoch_config}")
        
        # Rate of Change
        roc_period = self.momentum_config.get('roc_period', 10)
        df['ROC'] = ta.roc(df['close'], length=roc_period)
        logger.debug(f"Added ROC with period {roc_period}")
        
        logger.info("Added momentum indicators: RSI, MACD, Stochastic, ROC")
        
        return df
    
    def add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based technical indicators (Bollinger Bands, ATR, Volatility Ratio)
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility indicators added
        """
        df = data.copy()
        
        # Bollinger Bands
        bb_config = self.volatility_config.get('bollinger_bands', {'period': 20, 'std': 2})
        bb_result = ta.bbands(
            df['close'], 
            length=bb_config['period'], 
            std=bb_config['std']
        )
        
        if bb_result is not None:
            # Get the actual column names from the result
            bb_columns = bb_result.columns.tolist()
            
            # Find columns by pattern matching
            upper_col = [col for col in bb_columns if 'BBU_' in col][0] if any('BBU_' in col for col in bb_columns) else None
            middle_col = [col for col in bb_columns if 'BBM_' in col][0] if any('BBM_' in col for col in bb_columns) else None
            lower_col = [col for col in bb_columns if 'BBL_' in col][0] if any('BBL_' in col for col in bb_columns) else None
            
            if upper_col and middle_col and lower_col:
                df['BB_Upper'] = bb_result[upper_col]
                df['BB_Middle'] = bb_result[middle_col]
                df['BB_Lower'] = bb_result[lower_col]
                df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
                df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
                logger.debug(f"Added Bollinger Bands with config {bb_config}")
            else:
                logger.warning(f"Could not find expected Bollinger Band columns in result: {bb_columns}")
        
        # Average True Range (ATR)
        atr_period = self.volatility_config.get('atr_period', 14)
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
        logger.debug(f"Added ATR with period {atr_period}")
        
        # Volatility Ratio (current volatility vs historical average)
        volatility_period = self.volatility_config.get('volatility_period', 20)
        df['Returns'] = df['close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=volatility_period).std()
        df['Volatility_Ratio'] = df['Volatility'] / df['Volatility'].rolling(window=volatility_period*2).mean()
        
        # Clean up temporary columns
        df = df.drop(['Returns'], axis=1)
        
        logger.info("Added volatility indicators: Bollinger Bands, ATR, Volatility Ratio")
        
        return df
    
    def add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based technical indicators (OBV, Volume SMA, Volume ROC)
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume indicators added
        """
        df = data.copy()
        
        # On-Balance Volume (OBV)
        df['OBV'] = ta.obv(df['close'], df['volume'])
        logger.debug("Added OBV")
        
        # Volume Simple Moving Averages
        volume_periods = self.volume_config.get('periods', [20])
        for period in volume_periods:
            df[f'Volume_SMA_{period}'] = ta.sma(df['volume'], length=period)
            df[f'Volume_Ratio_{period}'] = df['volume'] / df[f'Volume_SMA_{period}']
            logger.debug(f"Added Volume_SMA_{period} and Volume_Ratio_{period}")
        
        # Volume Rate of Change
        volume_roc_period = self.volume_config.get('roc_period', 10)
        df['Volume_ROC'] = ta.roc(df['volume'], length=volume_roc_period)
        logger.debug(f"Added Volume_ROC with period {volume_roc_period}")
        
        # Price-Volume Trend (PVT)
        df['PVT'] = ta.pvt(df['close'], df['volume'])
        logger.debug("Added PVT")
        
        logger.info("Added volume indicators: OBV, Volume SMA, Volume ROC, PVT")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names that will be created
        
        Returns:
            List of feature column names
        """
        features = []
        
        # Trend features
        sma_periods = self.trend_config.get('sma_periods', [20, 50, 200])
        ema_periods = self.trend_config.get('ema_periods', [12, 26, 50])
        
        for period in sma_periods:
            features.append(f'SMA_{period}')
        for period in ema_periods:
            features.append(f'EMA_{period}')
        
        if sma_periods:
            features.append(f'Price_SMA_{sma_periods[0]}_Ratio')
        if ema_periods:
            features.append(f'Price_EMA_{ema_periods[0]}_Ratio')
        
        # Momentum features
        features.extend(['RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 
                        'Stoch_K', 'Stoch_D', 'ROC'])
        
        # Volatility features
        features.extend(['BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 
                        'BB_Position', 'ATR', 'Volatility', 'Volatility_Ratio'])
        
        # Volume features
        volume_periods = self.volume_config.get('periods', [20])
        features.extend(['OBV', 'Volume_ROC', 'PVT'])
        for period in volume_periods:
            features.extend([f'Volume_SMA_{period}', f'Volume_Ratio_{period}'])
        
        return features
    
    def select_features(self, data: pd.DataFrame, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Select specific features from the dataframe
        
        Args:
            data: DataFrame with all features
            feature_names: List of feature names to select. If None, returns all features
            
        Returns:
            DataFrame with selected features only
        """
        if feature_names is None:
            # Return all non-OHLCV columns as features
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in data.columns if col not in ohlcv_cols]
            return data[feature_cols]
        else:
            # Return only specified features that exist in the data
            available_features = [col for col in feature_names if col in data.columns]
            if len(available_features) != len(feature_names):
                missing = set(feature_names) - set(available_features)
                logger.warning(f"Some requested features not found in data: {missing}")
            return data[available_features]
    
    def create_feature_combinations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features by combining existing indicators
        
        Args:
            data: DataFrame with basic technical indicators
            
        Returns:
            DataFrame with additional combination features
        """
        df = data.copy()
        
        # Price momentum combinations
        if 'RSI' in df.columns and 'MACD' in df.columns:
            df['RSI_MACD_Signal'] = ((df['RSI'] > 50) & (df['MACD'] > 0)).astype(int)
        
        # Trend strength combinations
        if 'SMA_20' in df.columns and 'EMA_12' in df.columns:
            df['Trend_Alignment'] = ((df['close'] > df['SMA_20']) & (df['close'] > df['EMA_12'])).astype(int)
        
        # Volatility breakout signals
        if 'BB_Position' in df.columns and 'ATR' in df.columns:
            df['BB_Breakout'] = ((df['BB_Position'] > 1.0) | (df['BB_Position'] < 0.0)).astype(int)
        
        # Volume confirmation
        if 'Volume_Ratio_20' in df.columns and 'OBV' in df.columns:
            df['Volume_Confirmation'] = (df['Volume_Ratio_20'] > 1.5).astype(int)
        
        # Multi-timeframe momentum
        if 'EMA_12' in df.columns and 'EMA_26' in df.columns:
            df['EMA_Cross'] = (df['EMA_12'] > df['EMA_26']).astype(int)
        
        logger.info("Added feature combinations")
        return df