"""
Data cleaning and preprocessing system for time-series financial data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging

from ..exceptions import DataError, ValidationError


class DataPreprocessor:
    """
    Data preprocessing class for cleaning and preparing OHLCV data
    """
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Default preprocessing configuration
        self.config = config or {
            'outlier_method': 'iqr',  # 'iqr', 'zscore', 'isolation'
            'outlier_threshold': 3.0,
            'fill_method': 'forward',  # 'forward', 'backward', 'interpolate', 'drop'
            'max_gap_minutes': 60,  # Maximum gap to fill in minutes
            'min_volume_threshold': 0,
            'price_change_cap': 0.20,  # Cap extreme price changes at 20%
            'resample_method': 'last'  # For resampling: 'last', 'ohlc'
        }
    
    def clean_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in OHLCV data using configured method
        
        Args:
            data: Input DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        try:
            df_cleaned = data.copy()
            initial_rows = len(df_cleaned)
            
            # Identify critical columns that cannot have missing values
            critical_cols = ['open', 'high', 'low', 'close']
            non_critical_cols = ['volume']
            
            # Handle critical columns based on method
            if self.config['fill_method'] == 'drop':
                df_cleaned = df_cleaned.dropna(subset=critical_cols)
                dropped_rows = initial_rows - len(df_cleaned)
                if dropped_rows > 0:
                    self.logger.info(f"Dropped {dropped_rows} rows with missing critical values")
            
            elif self.config['fill_method'] == 'forward':
                df_cleaned[critical_cols] = df_cleaned[critical_cols].ffill()
                # Drop any remaining NaN rows at the beginning
                df_cleaned = df_cleaned.dropna(subset=critical_cols)
                
            elif self.config['fill_method'] == 'backward':
                df_cleaned[critical_cols] = df_cleaned[critical_cols].bfill()
                # Drop any remaining NaN rows at the end
                df_cleaned = df_cleaned.dropna(subset=critical_cols)
                
            elif self.config['fill_method'] == 'interpolate':
                df_cleaned[critical_cols] = df_cleaned[critical_cols].interpolate(method='linear')
                # Drop any remaining NaN rows
                df_cleaned = df_cleaned.dropna(subset=critical_cols)
            
            # Handle non-critical columns (volume)
            for col in non_critical_cols:
                if col in df_cleaned.columns:
                    if self.config['fill_method'] == 'drop':
                        # For volume, fill with 0 instead of dropping
                        df_cleaned[col] = df_cleaned[col].fillna(0)
                    else:
                        # Use forward fill for volume, then fill remaining with 0
                        df_cleaned[col] = df_cleaned[col].ffill().fillna(0)
            
            final_rows = len(df_cleaned)
            if final_rows != initial_rows:
                self.logger.info(f"Missing value handling: {initial_rows} -> {final_rows} rows")
            
            return df_cleaned
            
        except Exception as e:
            raise DataError(f"Error handling missing values: {str(e)}")
    
    def handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers in price and volume data
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with outliers handled
        """
        try:
            df_cleaned = data.copy()
            outliers_handled = 0
            
            # Handle price outliers
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col not in df_cleaned.columns:
                    continue
                
                outlier_mask = self._detect_outliers(df_cleaned[col])
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    # Cap outliers using median-based approach
                    median_val = df_cleaned[col].median()
                    mad = (df_cleaned[col] - median_val).abs().median()
                    
                    # Set reasonable bounds
                    lower_bound = median_val - 5 * mad
                    upper_bound = median_val + 5 * mad
                    
                    # Cap extreme values
                    df_cleaned.loc[outlier_mask, col] = np.clip(
                        df_cleaned.loc[outlier_mask, col], 
                        lower_bound, 
                        upper_bound
                    )
                    
                    outliers_handled += outlier_count
                    self.logger.info(f"Capped {outlier_count} outliers in {col}")
            
            # Handle volume outliers separately
            if 'volume' in df_cleaned.columns:
                volume_outliers = self._detect_outliers(df_cleaned['volume'])
                volume_outlier_count = volume_outliers.sum()
                
                if volume_outlier_count > 0:
                    # For volume, use percentile-based capping
                    p99 = df_cleaned['volume'].quantile(0.99)
                    df_cleaned.loc[volume_outliers, 'volume'] = np.minimum(
                        df_cleaned.loc[volume_outliers, 'volume'].astype(float), p99
                    ).astype(df_cleaned['volume'].dtype)
                    
                    outliers_handled += volume_outlier_count
                    self.logger.info(f"Capped {volume_outlier_count} volume outliers")
            
            if outliers_handled > 0:
                self.logger.info(f"Total outliers handled: {outliers_handled}")
            
            return df_cleaned
            
        except Exception as e:
            raise DataError(f"Error handling outliers: {str(e)}")
    
    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """
        Detect outliers using configured method
        
        Args:
            series: Data series to analyze
            
        Returns:
            Boolean mask indicating outliers
        """
        if self.config['outlier_method'] == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(series.dropna()))
            # Create boolean mask for original series length
            outlier_mask = pd.Series(False, index=series.index)
            outlier_mask.loc[series.dropna().index] = z_scores > self.config['outlier_threshold']
            return outlier_mask
            
        elif self.config['outlier_method'] == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (series < lower_bound) | (series > upper_bound)
            
        elif self.config['outlier_method'] == 'isolation':
            try:
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(series.values.reshape(-1, 1))
                return pd.Series(outlier_labels == -1, index=series.index)
            except ImportError:
                self.logger.warning("sklearn not available, falling back to IQR method")
                return self._detect_outliers_iqr(series)
        
        else:
            raise ValueError(f"Unknown outlier detection method: {self.config['outlier_method']}")
    
    def setup_datetime_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Set up proper datetime indexing for time-series data
        
        Args:
            data: Input DataFrame with datetime column
            
        Returns:
            DataFrame with datetime index
        """
        try:
            df_indexed = data.copy()
            
            # Ensure datetime column exists
            if 'datetime' not in df_indexed.columns:
                raise DataError("No datetime column found for indexing")
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df_indexed['datetime']):
                df_indexed['datetime'] = pd.to_datetime(df_indexed['datetime'])
            
            # Set as index
            df_indexed = df_indexed.set_index('datetime')
            
            # Sort by index
            df_indexed = df_indexed.sort_index()
            
            # Remove duplicate indices
            if df_indexed.index.duplicated().any():
                duplicates = df_indexed.index.duplicated().sum()
                df_indexed = df_indexed[~df_indexed.index.duplicated(keep='first')]
                self.logger.info(f"Removed {duplicates} duplicate datetime indices")
            
            self.logger.info(f"Set up datetime index: {df_indexed.index.min()} to {df_indexed.index.max()}")
            
            return df_indexed
            
        except Exception as e:
            raise DataError(f"Error setting up datetime index: {str(e)}")
    
    def fill_time_gaps(self, data: pd.DataFrame, expected_freq: str = '1min') -> pd.DataFrame:
        """
        Fill gaps in time-series data with appropriate values
        
        Args:
            data: Input DataFrame with datetime index
            expected_freq: Expected frequency of data (e.g., '1min', '5min')
            
        Returns:
            DataFrame with time gaps filled
        """
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                raise DataError("DataFrame must have datetime index for gap filling")
            
            # Create complete time range
            start_time = data.index.min()
            end_time = data.index.max()
            complete_range = pd.date_range(start=start_time, end=end_time, freq=expected_freq)
            
            # Reindex to complete range
            df_filled = data.reindex(complete_range)
            
            # Calculate gaps
            original_count = len(data)
            filled_count = len(df_filled)
            gaps_filled = filled_count - original_count
            
            if gaps_filled > 0:
                # Fill gaps using forward fill, but limit the fill distance
                max_gap_periods = self._freq_to_periods(expected_freq, self.config['max_gap_minutes'])
                
                # Forward fill with limit
                df_filled = df_filled.ffill(limit=max_gap_periods)
                
                # For remaining gaps, use interpolation
                numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
                df_filled[numeric_cols] = df_filled[numeric_cols].interpolate(method='linear', limit=max_gap_periods)
                
                # Drop any remaining NaN rows (gaps too large to fill)
                df_filled = df_filled.dropna()
                
                final_gaps = filled_count - len(df_filled)
                self.logger.info(f"Filled {gaps_filled - final_gaps} time gaps, "
                               f"dropped {final_gaps} rows with gaps too large to fill")
            
            return df_filled
            
        except Exception as e:
            raise DataError(f"Error filling time gaps: {str(e)}")
    
    def _freq_to_periods(self, freq: str, minutes: int) -> int:
        """Convert frequency string and minutes to number of periods"""
        if freq == '1min':
            return minutes
        elif freq == '5min':
            return minutes // 5
        elif freq == '15min':
            return minutes // 15
        elif freq == '1H':
            return minutes // 60
        else:
            # Default to 1-minute periods
            return minutes
    
    def validate_data_integrity(self, data: pd.DataFrame) -> Dict:
        """
        Perform comprehensive data integrity checks
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with integrity check results
        """
        try:
            integrity_report = {
                'ohlc_consistency': True,
                'temporal_consistency': True,
                'volume_consistency': True,
                'issues': [],
                'warnings': []
            }
            
            # Check OHLC consistency
            ohlc_issues = self._check_ohlc_consistency(data)
            if ohlc_issues:
                integrity_report['ohlc_consistency'] = False
                integrity_report['issues'].extend(ohlc_issues)
            
            # Check temporal consistency
            temporal_issues = self._check_temporal_consistency(data)
            if temporal_issues:
                integrity_report['temporal_consistency'] = False
                integrity_report['issues'].extend(temporal_issues)
            
            # Check volume consistency
            volume_issues = self._check_volume_consistency(data)
            if volume_issues:
                integrity_report['volume_consistency'] = False
                integrity_report['issues'].extend(volume_issues)
            
            # Generate warnings for potential issues
            warnings = self._generate_data_warnings(data)
            integrity_report['warnings'].extend(warnings)
            
            is_valid = all([
                integrity_report['ohlc_consistency'],
                integrity_report['temporal_consistency'],
                integrity_report['volume_consistency']
            ])
            
            integrity_report['is_valid'] = is_valid
            
            if is_valid:
                self.logger.info("Data integrity validation passed")
            else:
                self.logger.warning(f"Data integrity issues found: {len(integrity_report['issues'])} errors")
            
            return integrity_report
            
        except Exception as e:
            return {
                'is_valid': False,
                'error': f"Integrity validation failed: {str(e)}",
                'issues': [str(e)],
                'warnings': []
            }
    
    def _check_ohlc_consistency(self, data: pd.DataFrame) -> List[str]:
        """Check OHLC price consistency"""
        issues = []
        
        try:
            ohlc_cols = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in ohlc_cols):
                return issues
            
            # High >= max(open, close)
            high_violations = (data['high'] < data[['open', 'close']].max(axis=1)).sum()
            if high_violations > 0:
                issues.append(f"High price violations: {high_violations} rows")
            
            # Low <= min(open, close)
            low_violations = (data['low'] > data[['open', 'close']].min(axis=1)).sum()
            if low_violations > 0:
                issues.append(f"Low price violations: {low_violations} rows")
            
            # High >= low
            high_low_violations = (data['high'] < data['low']).sum()
            if high_low_violations > 0:
                issues.append(f"High < Low violations: {high_low_violations} rows")
            
            # Non-positive prices
            for col in ohlc_cols:
                non_positive = (data[col] <= 0).sum()
                if non_positive > 0:
                    issues.append(f"Non-positive {col} prices: {non_positive} rows")
            
        except Exception as e:
            issues.append(f"OHLC consistency check error: {str(e)}")
        
        return issues
    
    def _check_temporal_consistency(self, data: pd.DataFrame) -> List[str]:
        """Check temporal data consistency"""
        issues = []
        
        try:
            if isinstance(data.index, pd.DatetimeIndex):
                # Check for proper ordering
                if not data.index.is_monotonic_increasing:
                    issues.append("Timestamps not in chronological order")
                
                # Check for duplicates
                duplicates = data.index.duplicated().sum()
                if duplicates > 0:
                    issues.append(f"Duplicate timestamps: {duplicates}")
            
            elif 'datetime' in data.columns:
                # Check datetime column if not using datetime index
                if not data['datetime'].is_monotonic_increasing:
                    issues.append("Datetime column not in chronological order")
                
                duplicates = data['datetime'].duplicated().sum()
                if duplicates > 0:
                    issues.append(f"Duplicate datetime values: {duplicates}")
            
        except Exception as e:
            issues.append(f"Temporal consistency check error: {str(e)}")
        
        return issues
    
    def _check_volume_consistency(self, data: pd.DataFrame) -> List[str]:
        """Check volume data consistency"""
        issues = []
        
        try:
            if 'volume' not in data.columns:
                return issues
            
            # Negative volume
            negative_volume = (data['volume'] < 0).sum()
            if negative_volume > 0:
                issues.append(f"Negative volume values: {negative_volume} rows")
            
            # Extremely high volume (potential data error)
            volume_q99 = data['volume'].quantile(0.99)
            volume_q50 = data['volume'].quantile(0.50)
            
            if volume_q99 > volume_q50 * 100:  # 99th percentile > 100x median
                extreme_volume = (data['volume'] > volume_q99).sum()
                issues.append(f"Potentially erroneous high volume: {extreme_volume} rows")
            
        except Exception as e:
            issues.append(f"Volume consistency check error: {str(e)}")
        
        return issues
    
    def _generate_data_warnings(self, data: pd.DataFrame) -> List[str]:
        """Generate warnings for potential data quality issues"""
        warnings = []
        
        try:
            # Check data density
            if len(data) < 1000:
                warnings.append(f"Limited data points: {len(data)} rows")
            
            # Check for excessive zero volume
            if 'volume' in data.columns:
                zero_volume_pct = (data['volume'] == 0).sum() / len(data) * 100
                if zero_volume_pct > 10:
                    warnings.append(f"High zero volume percentage: {zero_volume_pct:.1f}%")
            
            # Check for price stagnation
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in data.columns:
                    unique_pct = data[col].nunique() / len(data) * 100
                    if unique_pct < 50:  # Less than 50% unique values
                        warnings.append(f"Low price variation in {col}: {unique_pct:.1f}% unique values")
        
        except Exception as e:
            warnings.append(f"Warning generation error: {str(e)}")
        
        return warnings
    
    def repair_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Attempt to repair common data issues
        
        Args:
            data: Input DataFrame with potential issues
            
        Returns:
            DataFrame with repairs applied
        """
        try:
            df_repaired = data.copy()
            repairs_made = []
            
            # Repair OHLC inconsistencies
            ohlc_cols = ['open', 'high', 'low', 'close']
            if all(col in df_repaired.columns for col in ohlc_cols):
                
                # Fix high < max(open, close)
                max_oc = df_repaired[['open', 'close']].max(axis=1)
                high_fixes = (df_repaired['high'] < max_oc).sum()
                df_repaired['high'] = np.maximum(df_repaired['high'], max_oc)
                
                if high_fixes > 0:
                    repairs_made.append(f"Fixed {high_fixes} high price inconsistencies")
                
                # Fix low > min(open, close)
                min_oc = df_repaired[['open', 'close']].min(axis=1)
                low_fixes = (df_repaired['low'] > min_oc).sum()
                df_repaired['low'] = np.minimum(df_repaired['low'], min_oc)
                
                if low_fixes > 0:
                    repairs_made.append(f"Fixed {low_fixes} low price inconsistencies")
                
                # Ensure high >= low
                high_low_fixes = (df_repaired['high'] < df_repaired['low']).sum()
                if high_low_fixes > 0:
                    # Average high and low for these cases
                    mask = df_repaired['high'] < df_repaired['low']
                    avg_price = (df_repaired.loc[mask, 'high'] + df_repaired.loc[mask, 'low']) / 2
                    df_repaired.loc[mask, 'high'] = avg_price
                    df_repaired.loc[mask, 'low'] = avg_price
                    repairs_made.append(f"Fixed {high_low_fixes} high < low inconsistencies")
            
            # Repair negative volume
            if 'volume' in df_repaired.columns:
                negative_volume = (df_repaired['volume'] < 0).sum()
                if negative_volume > 0:
                    df_repaired['volume'] = df_repaired['volume'].abs()
                    repairs_made.append(f"Fixed {negative_volume} negative volume values")
            
            # Cap extreme price changes
            if 'close' in df_repaired.columns and len(df_repaired) > 1:
                price_changes = df_repaired['close'].pct_change().abs()
                extreme_changes = price_changes > self.config['price_change_cap']
                extreme_count = extreme_changes.sum()
                
                if extreme_count > 0:
                    # Cap extreme changes
                    for idx in df_repaired.index[extreme_changes]:
                        if idx == df_repaired.index[0]:
                            continue  # Skip first row
                        
                        prev_idx = df_repaired.index[df_repaired.index.get_loc(idx) - 1]
                        prev_close = df_repaired.loc[prev_idx, 'close']
                        
                        # Cap the change
                        max_change = prev_close * self.config['price_change_cap']
                        current_close = df_repaired.loc[idx, 'close']
                        
                        if current_close > prev_close + max_change:
                            df_repaired.loc[idx, 'close'] = prev_close + max_change
                        elif current_close < prev_close - max_change:
                            df_repaired.loc[idx, 'close'] = prev_close - max_change
                    
                    repairs_made.append(f"Capped {extreme_count} extreme price changes")
            
            if repairs_made:
                self.logger.info(f"Data repairs completed: {'; '.join(repairs_made)}")
            else:
                self.logger.info("No data repairs needed")
            
            return df_repaired
            
        except Exception as e:
            raise DataError(f"Error repairing data: {str(e)}")
    
    def preprocess_pipeline(self, data: pd.DataFrame, expected_freq: str = '1min') -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        
        Args:
            data: Raw input DataFrame
            expected_freq: Expected data frequency
            
        Returns:
            Fully preprocessed DataFrame
        """
        try:
            self.logger.info("Starting data preprocessing pipeline")
            
            # Step 1: Handle missing values
            df_processed = self.clean_missing_values(data)
            self.logger.info(f"Step 1 complete: {len(df_processed)} rows after missing value handling")
            
            # Step 2: Set up datetime indexing
            df_processed = self.setup_datetime_index(df_processed)
            self.logger.info(f"Step 2 complete: datetime index established")
            
            # Step 3: Fill time gaps
            df_processed = self.fill_time_gaps(df_processed, expected_freq)
            self.logger.info(f"Step 3 complete: {len(df_processed)} rows after gap filling")
            
            # Step 4: Handle outliers
            df_processed = self.handle_outliers(df_processed)
            self.logger.info(f"Step 4 complete: outliers handled")
            
            # Step 5: Repair data inconsistencies
            df_processed = self.repair_data(df_processed)
            self.logger.info(f"Step 5 complete: data repairs applied")
            
            # Step 6: Final integrity check
            integrity_report = self.validate_data_integrity(df_processed)
            if not integrity_report['is_valid']:
                self.logger.warning(f"Final integrity check found issues: {integrity_report['issues']}")
            
            self.logger.info(f"Preprocessing pipeline complete: {len(df_processed)} rows")
            
            return df_processed
            
        except Exception as e:
            raise DataError(f"Preprocessing pipeline failed: {str(e)}")