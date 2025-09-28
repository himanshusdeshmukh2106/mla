"""
Data validation system with completeness and consistency checks
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from scipy import stats

from ..interfaces import ValidationResult
from ..exceptions import ValidationError


class DataValidator:
    """
    DataValidator class for comprehensive data quality assessment
    """
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Default validation configuration
        self.config = config or {
            'max_missing_pct': 0.05,  # 5% maximum missing values
            'outlier_threshold': 3.0,  # Z-score threshold for outliers
            'min_volume': 0,  # Minimum volume threshold
            'max_price_change_pct': 0.20,  # 20% maximum price change between periods
            'min_data_points': 100  # Minimum number of data points required
        }
    
    def check_completeness(self, data: pd.DataFrame) -> bool:
        """
        Check data completeness against configured thresholds
        
        Args:
            data: DataFrame to check
            
        Returns:
            True if data meets completeness requirements
        """
        try:
            if data.empty:
                self.logger.error("DataFrame is empty")
                return False
            
            # Check minimum data points
            if len(data) < self.config['min_data_points']:
                self.logger.error(f"Insufficient data points: {len(data)} < {self.config['min_data_points']}")
                return False
            
            # Check missing value percentages
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col in data.columns:
                    missing_pct = data[col].isnull().sum() / len(data)
                    if missing_pct > self.config['max_missing_pct']:
                        self.logger.error(f"Column '{col}' exceeds missing value threshold: "
                                        f"{missing_pct:.2%} > {self.config['max_missing_pct']:.2%}")
                        return False
            
            self.logger.info("Data completeness check passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in completeness check: {str(e)}")
            return False
    
    def detect_outliers(self, data: pd.DataFrame, method: str = 'zscore') -> List[int]:
        """
        Detect outliers in OHLCV data using specified method
        
        Args:
            data: DataFrame to analyze
            method: Outlier detection method ('zscore', 'iqr', 'isolation')
            
        Returns:
            List of row indices containing outliers
        """
        try:
            outlier_indices = []
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            
            for col in numeric_cols:
                if col not in data.columns:
                    continue
                
                col_data = data[col].dropna()
                if len(col_data) == 0:
                    continue
                
                if method == 'zscore':
                    outliers = self._detect_zscore_outliers(col_data, col)
                elif method == 'iqr':
                    outliers = self._detect_iqr_outliers(col_data, col)
                elif method == 'isolation':
                    outliers = self._detect_isolation_outliers(col_data, col)
                else:
                    self.logger.warning(f"Unknown outlier detection method: {method}")
                    continue
                
                outlier_indices.extend(outliers)
            
            # Remove duplicates and sort
            outlier_indices = sorted(list(set(outlier_indices)))
            
            self.logger.info(f"Detected {len(outlier_indices)} outlier rows using {method} method")
            return outlier_indices
            
        except Exception as e:
            self.logger.error(f"Error in outlier detection: {str(e)}")
            return []
    
    def _detect_zscore_outliers(self, series: pd.Series, column_name: str) -> List[int]:
        """Detect outliers using Z-score method"""
        try:
            z_scores = np.abs(stats.zscore(series))
            outliers = series.index[z_scores > self.config['outlier_threshold']].tolist()
            
            if outliers:
                self.logger.debug(f"Found {len(outliers)} Z-score outliers in {column_name}")
            
            return outliers
            
        except Exception as e:
            self.logger.error(f"Error in Z-score outlier detection for {column_name}: {str(e)}")
            return []
    
    def _detect_iqr_outliers(self, series: pd.Series, column_name: str) -> List[int]:
        """Detect outliers using Interquartile Range method"""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series.index[(series < lower_bound) | (series > upper_bound)].tolist()
            
            if outliers:
                self.logger.debug(f"Found {len(outliers)} IQR outliers in {column_name}")
            
            return outliers
            
        except Exception as e:
            self.logger.error(f"Error in IQR outlier detection for {column_name}: {str(e)}")
            return []
    
    def _detect_isolation_outliers(self, series: pd.Series, column_name: str) -> List[int]:
        """Detect outliers using Isolation Forest method"""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Reshape for sklearn
            data_reshaped = series.values.reshape(-1, 1)
            
            # Fit isolation forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(data_reshaped)
            
            # Get outlier indices
            outliers = series.index[outlier_labels == -1].tolist()
            
            if outliers:
                self.logger.debug(f"Found {len(outliers)} isolation forest outliers in {column_name}")
            
            return outliers
            
        except ImportError:
            self.logger.warning("sklearn not available for isolation forest outlier detection")
            return []
        except Exception as e:
            self.logger.error(f"Error in isolation forest outlier detection for {column_name}: {str(e)}")
            return []
    
    def validate_ohlcv_consistency(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate OHLCV data consistency and logical constraints
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            ValidationResult with detailed validation information
        """
        errors = []
        warnings = []
        
        try:
            # Check required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                errors.append(f"Missing required columns: {missing_cols}")
                return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
            
            # OHLC logical consistency checks
            ohlc_errors = self._validate_ohlc_logic(data)
            errors.extend(ohlc_errors)
            
            # Price change validation
            price_change_errors = self._validate_price_changes(data)
            errors.extend(price_change_errors)
            
            # Volume validation
            volume_errors = self._validate_volume(data)
            errors.extend(volume_errors)
            
            # Temporal consistency
            temporal_errors = self._validate_temporal_consistency(data)
            errors.extend(temporal_errors)
            
            # Data quality assessment
            quality_warnings = self._assess_data_quality(data)
            warnings.extend(quality_warnings)
            
            is_valid = len(errors) == 0
            
            if is_valid:
                self.logger.info("OHLCV consistency validation passed")
            else:
                self.logger.warning(f"OHLCV consistency validation failed with {len(errors)} errors")
            
            return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
    
    def _validate_ohlc_logic(self, data: pd.DataFrame) -> List[str]:
        """Validate OHLC logical constraints"""
        errors = []
        
        try:
            # High should be >= max(open, close)
            high_violations = (data['high'] < data[['open', 'close']].max(axis=1)).sum()
            if high_violations > 0:
                errors.append(f"Found {high_violations} rows where high < max(open, close)")
            
            # Low should be <= min(open, close)
            low_violations = (data['low'] > data[['open', 'close']].min(axis=1)).sum()
            if low_violations > 0:
                errors.append(f"Found {low_violations} rows where low > min(open, close)")
            
            # High should be >= low
            high_low_violations = (data['high'] < data['low']).sum()
            if high_low_violations > 0:
                errors.append(f"Found {high_low_violations} rows where high < low")
            
            # Check for non-positive prices
            for col in ['open', 'high', 'low', 'close']:
                non_positive = (data[col] <= 0).sum()
                if non_positive > 0:
                    errors.append(f"Found {non_positive} non-positive values in {col}")
            
        except Exception as e:
            errors.append(f"OHLC logic validation error: {str(e)}")
        
        return errors
    
    def _validate_price_changes(self, data: pd.DataFrame) -> List[str]:
        """Validate price changes between periods"""
        errors = []
        
        try:
            if len(data) < 2:
                return errors
            
            # Calculate price changes
            price_changes = data['close'].pct_change().abs()
            
            # Find extreme price changes
            extreme_changes = price_changes > self.config['max_price_change_pct']
            extreme_count = extreme_changes.sum()
            
            if extreme_count > 0:
                max_change = price_changes.max()
                errors.append(f"Found {extreme_count} extreme price changes "
                            f"(max: {max_change:.2%}, threshold: {self.config['max_price_change_pct']:.2%})")
        
        except Exception as e:
            errors.append(f"Price change validation error: {str(e)}")
        
        return errors
    
    def _validate_volume(self, data: pd.DataFrame) -> List[str]:
        """Validate volume data"""
        errors = []
        
        try:
            # Check for negative volume
            negative_volume = (data['volume'] < 0).sum()
            if negative_volume > 0:
                errors.append(f"Found {negative_volume} negative volume values")
            
            # Check for zero volume (warning, not error)
            zero_volume = (data['volume'] == 0).sum()
            if zero_volume > len(data) * 0.1:  # More than 10% zero volume
                errors.append(f"High proportion of zero volume: {zero_volume} rows ({zero_volume/len(data):.1%})")
        
        except Exception as e:
            errors.append(f"Volume validation error: {str(e)}")
        
        return errors
    
    def _validate_temporal_consistency(self, data: pd.DataFrame) -> List[str]:
        """Validate temporal consistency of data"""
        errors = []
        
        try:
            if 'datetime' not in data.columns:
                return errors
            
            # Check for duplicate timestamps
            duplicates = data['datetime'].duplicated().sum()
            if duplicates > 0:
                errors.append(f"Found {duplicates} duplicate timestamps")
            
            # Check for proper time ordering
            if not data['datetime'].is_monotonic_increasing:
                errors.append("Timestamps are not in chronological order")
            
            # Check for large time gaps (more than expected)
            if len(data) > 1:
                time_diffs = data['datetime'].diff().dropna()
                median_diff = time_diffs.median()
                large_gaps = (time_diffs > median_diff * 10).sum()
                
                if large_gaps > 0:
                    errors.append(f"Found {large_gaps} unusually large time gaps")
        
        except Exception as e:
            errors.append(f"Temporal consistency validation error: {str(e)}")
        
        return errors
    
    def _assess_data_quality(self, data: pd.DataFrame) -> List[str]:
        """Assess overall data quality and generate warnings"""
        warnings = []
        
        try:
            # Check data density
            if len(data) < 1000:
                warnings.append(f"Limited data points: {len(data)} rows")
            
            # Check for repeated values (potential data issues)
            for col in ['open', 'high', 'low', 'close']:
                if col in data.columns:
                    repeated_pct = (data[col].value_counts().iloc[0] / len(data)) * 100
                    if repeated_pct > 10:  # More than 10% repeated values
                        warnings.append(f"High proportion of repeated values in {col}: {repeated_pct:.1f}%")
            
            # Check volume patterns
            if 'volume' in data.columns:
                zero_volume_pct = (data['volume'] == 0).sum() / len(data) * 100
                if zero_volume_pct > 5:
                    warnings.append(f"High proportion of zero volume: {zero_volume_pct:.1f}%")
        
        except Exception as e:
            warnings.append(f"Data quality assessment error: {str(e)}")
        
        return warnings
    
    def generate_data_quality_report(self, data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data quality report
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with detailed quality metrics
        """
        try:
            report = {
                'basic_stats': {
                    'total_rows': len(data),
                    'total_columns': len(data.columns),
                    'date_range': None,
                    'memory_usage': data.memory_usage(deep=True).sum()
                },
                'completeness': {},
                'consistency': {},
                'outliers': {},
                'quality_score': 0.0
            }
            
            # Date range
            if 'datetime' in data.columns:
                report['basic_stats']['date_range'] = {
                    'start': data['datetime'].min(),
                    'end': data['datetime'].max(),
                    'duration': data['datetime'].max() - data['datetime'].min()
                }
            
            # Completeness metrics
            for col in data.columns:
                missing_count = data[col].isnull().sum()
                report['completeness'][col] = {
                    'missing_count': missing_count,
                    'missing_percentage': (missing_count / len(data)) * 100,
                    'complete': missing_count == 0
                }
            
            # Consistency validation
            validation_result = self.validate_ohlcv_consistency(data)
            report['consistency'] = {
                'is_valid': validation_result.is_valid,
                'errors': validation_result.errors,
                'warnings': validation_result.warnings
            }
            
            # Outlier detection
            outlier_indices = self.detect_outliers(data)
            report['outliers'] = {
                'count': len(outlier_indices),
                'percentage': (len(outlier_indices) / len(data)) * 100,
                'indices': outlier_indices[:100]  # Limit to first 100
            }
            
            # Calculate quality score (0-100)
            score = 100.0
            
            # Deduct for missing data
            avg_missing_pct = np.mean([info['missing_percentage'] for info in report['completeness'].values()])
            score -= min(avg_missing_pct * 2, 30)  # Max 30 points deduction
            
            # Deduct for consistency errors
            score -= min(len(validation_result.errors) * 10, 40)  # Max 40 points deduction
            
            # Deduct for outliers
            outlier_pct = report['outliers']['percentage']
            score -= min(outlier_pct, 20)  # Max 20 points deduction
            
            report['quality_score'] = max(score, 0.0)
            
            self.logger.info(f"Generated data quality report. Quality score: {report['quality_score']:.1f}/100")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating data quality report: {str(e)}")
            return {'error': str(e)}