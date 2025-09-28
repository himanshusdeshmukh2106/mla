"""
Data loading and validation system for OHLCV data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

from ..interfaces import IDataLoader, ValidationResult
from ..exceptions import DataLoadingError, ValidationError


class DataLoader(IDataLoader):
    """
    DataLoader class for CSV file reading and OHLCV data handling
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Standard OHLCV column mappings
        self.column_mappings = {
            'datetime': ['datetime', 'timestamp', 'time', 'date'],
            'open': ['open', 'Open', 'OPEN', 'o'],
            'high': ['high', 'High', 'HIGH', 'h'],
            'low': ['low', 'Low', 'LOW', 'l'],
            'close': ['close', 'Close', 'CLOSE', 'c'],
            'volume': ['volume', 'Volume', 'VOLUME', 'vol', 'v']
        }
        
        # Required columns for OHLCV data
        self.required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    
    def load_csv_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load OHLCV data from CSV file with automatic column mapping
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with standardized OHLCV columns
            
        Raises:
            DataLoadingError: If file cannot be loaded or columns cannot be mapped
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise DataLoadingError(f"File not found: {file_path}")
            
            # Try different separators and encodings
            separators = [',', ';', '\t']
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            df = None
            for sep in separators:
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, sep=sep, encoding=encoding)
                        if len(df.columns) >= 5:  # Minimum columns for OHLCV
                            break
                    except Exception:
                        continue
                if df is not None and len(df.columns) >= 5:
                    break
            
            if df is None or df.empty:
                raise DataLoadingError(f"Could not read CSV file or file is empty: {file_path}")
            
            self.logger.info(f"Loaded {len(df)} rows from {file_path}")
            
            # Map columns to standard names
            df = self._map_columns(df)
            
            # Validate required columns exist
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                raise DataLoadingError(f"Missing required columns: {missing_cols}")
            
            return df
            
        except Exception as e:
            if isinstance(e, DataLoadingError):
                raise
            raise DataLoadingError(f"Error loading CSV file {file_path}: {str(e)}")
    
    def load_ohlcv_data(self, symbol: str, timeframe: str, 
                       start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load OHLCV data for given parameters (CSV implementation)
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe (e.g., '1min', '5min')
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with OHLCV data filtered by date range
        """
        try:
            # Construct expected CSV filename - handle different naming conventions
            possible_filenames = [
                f"{symbol}_{timeframe}.csv",
                f"{symbol.lower()}_{timeframe}.csv", 
                f"{symbol}_data_{timeframe}_full_year.csv",
                f"{symbol.lower()}_data_{timeframe}_full_year.csv"
            ]
            
            csv_path = None
            for filename in possible_filenames:
                potential_path = Path("data") / filename
                if potential_path.exists():
                    csv_path = potential_path
                    break
            
            if csv_path is None:
                raise DataLoadingError(f"No data file found for {symbol} with timeframe {timeframe}. Tried: {possible_filenames}")
            
            # Load the CSV data
            df = self.load_csv_data(csv_path)
            
            # Convert datetime column
            df = self._process_datetime(df)
            
            # Filter by date range
            df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
            
            if df.empty:
                raise DataLoadingError(f"No data found for {symbol} between {start_date} and {end_date}")
            
            self.logger.info(f"Loaded {len(df)} rows for {symbol} from {start_date} to {end_date}")
            return df
            
        except Exception as e:
            if isinstance(e, DataLoadingError):
                raise
            raise DataLoadingError(f"Error loading OHLCV data for {symbol}: {str(e)}")
    
    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map CSV columns to standard OHLCV column names
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        df_copy = df.copy()
        column_mapping = {}
        
        for standard_name, possible_names in self.column_mappings.items():
            for col in df_copy.columns:
                if col.lower() in [name.lower() for name in possible_names]:
                    column_mapping[col] = standard_name
                    break
        
        if column_mapping:
            df_copy = df_copy.rename(columns=column_mapping)
            self.logger.info(f"Mapped columns: {column_mapping}")
        
        return df_copy
    
    def _process_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process datetime column to ensure proper formatting
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with processed datetime column
        """
        df_copy = df.copy()
        
        try:
            # Try to parse datetime column
            df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
            
            # Sort by datetime
            df_copy = df_copy.sort_values('datetime').reset_index(drop=True)
            
            self.logger.info("Successfully processed datetime column")
            
        except Exception as e:
            raise DataLoadingError(f"Error processing datetime column: {str(e)}")
        
        return df_copy
    
    def validate_data_integrity(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate data completeness and consistency
        
        Args:
            data: DataFrame to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        
        try:
            # Check if DataFrame is empty
            if data.empty:
                errors.append("DataFrame is empty")
                return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
            
            # Check required columns
            missing_cols = [col for col in self.required_columns if col not in data.columns]
            if missing_cols:
                errors.append(f"Missing required columns: {missing_cols}")
            
            # Check for missing values
            missing_counts = data[self.required_columns].isnull().sum()
            for col, count in missing_counts.items():
                if count > 0:
                    pct = (count / len(data)) * 100
                    if pct > 5:  # More than 5% missing
                        errors.append(f"Column '{col}' has {count} missing values ({pct:.1f}%)")
                    else:
                        warnings.append(f"Column '{col}' has {count} missing values ({pct:.1f}%)")
            
            # Check OHLC consistency
            ohlc_errors = self._validate_ohlc_consistency(data)
            errors.extend(ohlc_errors)
            
            # Check for duplicate timestamps
            if 'datetime' in data.columns:
                duplicates = data['datetime'].duplicated().sum()
                if duplicates > 0:
                    warnings.append(f"Found {duplicates} duplicate timestamps")
            
            # Check data types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                    errors.append(f"Column '{col}' is not numeric")
            
            is_valid = len(errors) == 0
            
            if is_valid:
                self.logger.info("Data validation passed")
            else:
                self.logger.warning(f"Data validation failed with {len(errors)} errors")
            
            return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
    
    def _validate_ohlc_consistency(self, data: pd.DataFrame) -> List[str]:
        """
        Validate OHLC price consistency
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of error messages
        """
        errors = []
        
        try:
            ohlc_cols = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in ohlc_cols):
                return errors
            
            # Check that high >= max(open, close) and low <= min(open, close)
            high_errors = (data['high'] < data[['open', 'close']].max(axis=1)).sum()
            low_errors = (data['low'] > data[['open', 'close']].min(axis=1)).sum()
            
            if high_errors > 0:
                errors.append(f"Found {high_errors} rows where high < max(open, close)")
            
            if low_errors > 0:
                errors.append(f"Found {low_errors} rows where low > min(open, close)")
            
            # Check for negative prices
            for col in ohlc_cols:
                negative_count = (data[col] <= 0).sum()
                if negative_count > 0:
                    errors.append(f"Found {negative_count} non-positive values in {col}")
            
            # Check volume
            if 'volume' in data.columns:
                negative_volume = (data['volume'] < 0).sum()
                if negative_volume > 0:
                    errors.append(f"Found {negative_volume} negative volume values")
            
        except Exception as e:
            errors.append(f"OHLC consistency check failed: {str(e)}")
        
        return errors
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers
        
        Args:
            data: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        try:
            df_cleaned = data.copy()
            
            # Remove rows with missing critical values
            critical_cols = ['open', 'high', 'low', 'close']
            initial_rows = len(df_cleaned)
            df_cleaned = df_cleaned.dropna(subset=critical_cols)
            removed_rows = initial_rows - len(df_cleaned)
            
            if removed_rows > 0:
                self.logger.info(f"Removed {removed_rows} rows with missing OHLC values")
            
            # Fill missing volume with 0
            if 'volume' in df_cleaned.columns:
                df_cleaned['volume'] = df_cleaned['volume'].fillna(0)
            
            # Remove duplicate timestamps
            if 'datetime' in df_cleaned.columns:
                initial_rows = len(df_cleaned)
                df_cleaned = df_cleaned.drop_duplicates(subset=['datetime'], keep='first')
                removed_duplicates = initial_rows - len(df_cleaned)
                
                if removed_duplicates > 0:
                    self.logger.info(f"Removed {removed_duplicates} duplicate timestamps")
            
            self.logger.info(f"Data cleaning completed. Final dataset: {len(df_cleaned)} rows")
            
            return df_cleaned
            
        except Exception as e:
            raise DataLoadingError(f"Error cleaning data: {str(e)}")