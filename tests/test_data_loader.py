"""
Unit tests for data loading and validation functionality
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.interfaces import ValidationResult
from src.exceptions import DataLoadingError, ValidationError


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.loader = DataLoader()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample OHLCV data
        self.sample_data = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01 09:30:00', periods=100, freq='1min'),
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Ensure OHLC consistency
        for i in range(len(self.sample_data)):
            high = max(self.sample_data.loc[i, 'open'], self.sample_data.loc[i, 'close'], 
                      self.sample_data.loc[i, 'high'])
            low = min(self.sample_data.loc[i, 'open'], self.sample_data.loc[i, 'close'], 
                     self.sample_data.loc[i, 'low'])
            self.sample_data.loc[i, 'high'] = high
            self.sample_data.loc[i, 'low'] = low
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_csv_data_success(self):
        """Test successful CSV data loading"""
        # Create temporary CSV file
        csv_path = Path(self.temp_dir) / "test_data.csv"
        self.sample_data.to_csv(csv_path, index=False)
        
        # Load data
        result = self.loader.load_csv_data(csv_path)
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 100)
        self.assertIn('datetime', result.columns)
        self.assertIn('open', result.columns)
        self.assertIn('high', result.columns)
        self.assertIn('low', result.columns)
        self.assertIn('close', result.columns)
        self.assertIn('volume', result.columns)
    
    def test_load_csv_data_file_not_found(self):
        """Test CSV loading with non-existent file"""
        with self.assertRaises(DataLoadingError):
            self.loader.load_csv_data("non_existent_file.csv")
    
    def test_column_mapping(self):
        """Test automatic column mapping"""
        # Create data with different column names
        mapped_data = self.sample_data.copy()
        mapped_data = mapped_data.rename(columns={
            'datetime': 'timestamp',
            'open': 'Open',
            'high': 'HIGH',
            'low': 'Low',
            'close': 'CLOSE',
            'volume': 'vol'
        })
        
        # Save and load
        csv_path = Path(self.temp_dir) / "mapped_data.csv"
        mapped_data.to_csv(csv_path, index=False)
        
        result = self.loader.load_csv_data(csv_path)
        
        # Check that columns are mapped correctly
        expected_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        for col in expected_cols:
            self.assertIn(col, result.columns)
    
    def test_load_ohlcv_data_date_filtering(self):
        """Test OHLCV data loading with date filtering"""
        # Create CSV file
        csv_path = Path(self.temp_dir) / "AAPL_1min.csv"
        self.sample_data.to_csv(csv_path, index=False)
        
        # Create data directory and move file
        data_dir = Path(self.temp_dir) / "data"
        data_dir.mkdir(exist_ok=True)
        final_path = data_dir / "AAPL_1min.csv"
        self.sample_data.to_csv(final_path, index=False)
        
        # Change working directory temporarily
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            start_date = datetime(2023, 1, 1, 9, 30)
            end_date = datetime(2023, 1, 1, 10, 30)
            
            result = self.loader.load_ohlcv_data("AAPL", "1min", start_date, end_date)
            
            # Check date filtering
            self.assertTrue(all(result['datetime'] >= start_date))
            self.assertTrue(all(result['datetime'] <= end_date))
            
        finally:
            os.chdir(original_cwd)
    
    def test_validate_data_integrity_success(self):
        """Test successful data integrity validation"""
        result = self.loader.validate_data_integrity(self.sample_data)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
    
    def test_validate_data_integrity_missing_columns(self):
        """Test validation with missing columns"""
        incomplete_data = self.sample_data.drop(columns=['volume'])
        result = self.loader.validate_data_integrity(incomplete_data)
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any('Missing required columns' in error for error in result.errors))
    
    def test_validate_data_integrity_missing_values(self):
        """Test validation with missing values"""
        data_with_nulls = self.sample_data.copy()
        data_with_nulls.loc[0:10, 'close'] = np.nan
        
        result = self.loader.validate_data_integrity(data_with_nulls)
        
        # Should have warnings about missing values
        self.assertTrue(len(result.warnings) > 0 or len(result.errors) > 0)
    
    def test_clean_data(self):
        """Test data cleaning functionality"""
        # Create dirty data
        dirty_data = self.sample_data.copy()
        dirty_data.loc[0:5, 'close'] = np.nan
        dirty_data.loc[10, 'volume'] = np.nan
        
        # Add duplicate timestamps
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[[0]]], ignore_index=True)
        
        cleaned_data = self.loader.clean_data(dirty_data)
        
        # Check that missing OHLC rows are removed
        self.assertFalse(cleaned_data['close'].isnull().any())
        
        # Check that volume nulls are filled with 0
        self.assertFalse(cleaned_data['volume'].isnull().any())
        
        # Check that duplicates are removed
        self.assertFalse(cleaned_data['datetime'].duplicated().any())


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = DataValidator()
        
        # Create sample OHLCV data
        self.sample_data = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01 09:30:00', periods=200, freq='1min'),
            'open': np.random.uniform(100, 110, 200),
            'high': np.random.uniform(110, 120, 200),
            'low': np.random.uniform(90, 100, 200),
            'close': np.random.uniform(100, 110, 200),
            'volume': np.random.randint(1000, 10000, 200)
        })
        
        # Ensure OHLC consistency
        for i in range(len(self.sample_data)):
            high = max(self.sample_data.loc[i, 'open'], self.sample_data.loc[i, 'close'], 
                      self.sample_data.loc[i, 'high'])
            low = min(self.sample_data.loc[i, 'open'], self.sample_data.loc[i, 'close'], 
                     self.sample_data.loc[i, 'low'])
            self.sample_data.loc[i, 'high'] = high
            self.sample_data.loc[i, 'low'] = low
    
    def test_check_completeness_success(self):
        """Test successful completeness check"""
        result = self.validator.check_completeness(self.sample_data)
        self.assertTrue(result)
    
    def test_check_completeness_insufficient_data(self):
        """Test completeness check with insufficient data"""
        small_data = self.sample_data.head(50)  # Less than minimum 100 points
        result = self.validator.check_completeness(small_data)
        self.assertFalse(result)
    
    def test_check_completeness_too_many_missing(self):
        """Test completeness check with too many missing values"""
        data_with_missing = self.sample_data.copy()
        # Make 10% of close values missing (exceeds 5% threshold)
        missing_indices = np.random.choice(len(data_with_missing), size=20, replace=False)
        data_with_missing.loc[missing_indices, 'close'] = np.nan
        
        result = self.validator.check_completeness(data_with_missing)
        self.assertFalse(result)
    
    def test_detect_outliers_zscore(self):
        """Test Z-score outlier detection"""
        # Add some outliers
        data_with_outliers = self.sample_data.copy()
        data_with_outliers.loc[0, 'close'] = 1000  # Extreme outlier
        data_with_outliers.loc[1, 'volume'] = 1000000  # Volume outlier
        
        outliers = self.validator.detect_outliers(data_with_outliers, method='zscore')
        
        self.assertIsInstance(outliers, list)
        self.assertTrue(len(outliers) > 0)
        self.assertIn(0, outliers)  # Should detect the extreme price outlier
    
    def test_detect_outliers_iqr(self):
        """Test IQR outlier detection"""
        # Add some outliers
        data_with_outliers = self.sample_data.copy()
        data_with_outliers.loc[0, 'close'] = 1000  # Extreme outlier
        
        outliers = self.validator.detect_outliers(data_with_outliers, method='iqr')
        
        self.assertIsInstance(outliers, list)
        self.assertTrue(len(outliers) > 0)
    
    def test_validate_ohlcv_consistency_success(self):
        """Test successful OHLCV consistency validation"""
        result = self.validator.validate_ohlcv_consistency(self.sample_data)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
    
    def test_validate_ohlcv_consistency_logic_errors(self):
        """Test OHLCV validation with logic errors"""
        invalid_data = self.sample_data.copy()
        
        # Create OHLC inconsistencies
        invalid_data.loc[0, 'high'] = 50  # High less than open/close
        invalid_data.loc[1, 'low'] = 150  # Low greater than open/close
        invalid_data.loc[2, 'close'] = -10  # Negative price
        
        result = self.validator.validate_ohlcv_consistency(invalid_data)
        
        self.assertFalse(result.is_valid)
        self.assertTrue(len(result.errors) > 0)
    
    def test_validate_ohlcv_consistency_missing_columns(self):
        """Test OHLCV validation with missing columns"""
        incomplete_data = self.sample_data.drop(columns=['volume'])
        result = self.validator.validate_ohlcv_consistency(incomplete_data)
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any('Missing required columns' in error for error in result.errors))
    
    def test_generate_data_quality_report(self):
        """Test data quality report generation"""
        report = self.validator.generate_data_quality_report(self.sample_data)
        
        self.assertIsInstance(report, dict)
        self.assertIn('basic_stats', report)
        self.assertIn('completeness', report)
        self.assertIn('consistency', report)
        self.assertIn('outliers', report)
        self.assertIn('quality_score', report)
        
        # Check basic stats
        self.assertEqual(report['basic_stats']['total_rows'], 200)
        self.assertGreater(report['quality_score'], 0)
        self.assertLessEqual(report['quality_score'], 100)
    
    def test_price_change_validation(self):
        """Test extreme price change validation"""
        data_with_extreme_changes = self.sample_data.copy()
        
        # Create extreme price change (50% jump)
        data_with_extreme_changes.loc[1, 'close'] = data_with_extreme_changes.loc[0, 'close'] * 1.5
        
        result = self.validator.validate_ohlcv_consistency(data_with_extreme_changes)
        
        # Should detect extreme price changes
        self.assertTrue(any('extreme price changes' in error.lower() for error in result.errors))
    
    def test_volume_validation(self):
        """Test volume validation"""
        data_with_bad_volume = self.sample_data.copy()
        
        # Add negative volume
        data_with_bad_volume.loc[0, 'volume'] = -1000
        
        result = self.validator.validate_ohlcv_consistency(data_with_bad_volume)
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any('negative volume' in error.lower() for error in result.errors))


if __name__ == '__main__':
    unittest.main()