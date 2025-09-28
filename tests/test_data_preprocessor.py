"""
Unit tests for data cleaning and preprocessing functionality
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.preprocessor import DataPreprocessor
from src.exceptions import DataError


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = DataPreprocessor()
        
        # Create sample OHLCV data
        dates = pd.date_range('2023-01-01 09:30:00', periods=100, freq='1min')
        self.sample_data = pd.DataFrame({
            'datetime': dates,
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
    
    def test_clean_missing_values_drop_method(self):
        """Test missing value cleaning with drop method"""
        # Create data with missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0:5, 'close'] = np.nan
        data_with_missing.loc[10, 'volume'] = np.nan
        
        # Configure to drop missing values
        config = {'fill_method': 'drop'}
        preprocessor = DataPreprocessor(config=config)
        
        cleaned_data = preprocessor.clean_missing_values(data_with_missing)
        
        # Should have fewer rows (dropped missing close values)
        self.assertLess(len(cleaned_data), len(data_with_missing))
        
        # Should not have missing values in critical columns
        critical_cols = ['open', 'high', 'low', 'close']
        for col in critical_cols:
            self.assertFalse(cleaned_data[col].isnull().any())
        
        # Volume should be filled with 0, not dropped
        self.assertFalse(cleaned_data['volume'].isnull().any())
    
    def test_clean_missing_values_forward_fill(self):
        """Test missing value cleaning with forward fill method"""
        # Create data with missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[5:7, 'close'] = np.nan
        data_with_missing.loc[10, 'volume'] = np.nan
        
        # Configure to forward fill
        config = {'fill_method': 'forward'}
        preprocessor = DataPreprocessor(config=config)
        
        cleaned_data = preprocessor.clean_missing_values(data_with_missing)
        
        # Should have same number of rows (or fewer if first rows had NaN)
        self.assertLessEqual(len(cleaned_data), len(data_with_missing))
        
        # Should not have missing values
        self.assertFalse(cleaned_data.isnull().any().any())
    
    def test_clean_missing_values_interpolate(self):
        """Test missing value cleaning with interpolation method"""
        # Create data with missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[5:7, 'close'] = np.nan
        
        # Configure to interpolate
        config = {'fill_method': 'interpolate'}
        preprocessor = DataPreprocessor(config=config)
        
        cleaned_data = preprocessor.clean_missing_values(data_with_missing)
        
        # Should not have missing values in critical columns
        critical_cols = ['open', 'high', 'low', 'close']
        for col in critical_cols:
            self.assertFalse(cleaned_data[col].isnull().any())
    
    def test_handle_outliers_iqr_method(self):
        """Test outlier handling with IQR method"""
        # Create data with outliers
        data_with_outliers = self.sample_data.copy()
        data_with_outliers.loc[0, 'close'] = 1000  # Extreme outlier
        data_with_outliers.loc[1, 'volume'] = 1000000  # Volume outlier
        
        # Configure IQR outlier detection
        config = {'outlier_method': 'iqr'}
        preprocessor = DataPreprocessor(config=config)
        
        cleaned_data = preprocessor.handle_outliers(data_with_outliers)
        
        # Outliers should be capped, not the original extreme values
        self.assertNotEqual(cleaned_data.loc[0, 'close'], 1000)
        self.assertNotEqual(cleaned_data.loc[1, 'volume'], 1000000)
        
        # Values should be reasonable (not negative or zero)
        self.assertGreater(cleaned_data.loc[0, 'close'], 0)
        self.assertGreater(cleaned_data.loc[1, 'volume'], 0)
    
    def test_handle_outliers_zscore_method(self):
        """Test outlier handling with Z-score method"""
        # Create data with outliers
        data_with_outliers = self.sample_data.copy()
        data_with_outliers.loc[0, 'close'] = 1000  # Extreme outlier
        
        # Configure Z-score outlier detection
        config = {'outlier_method': 'zscore', 'outlier_threshold': 2.0}
        preprocessor = DataPreprocessor(config=config)
        
        cleaned_data = preprocessor.handle_outliers(data_with_outliers)
        
        # Outlier should be capped
        self.assertNotEqual(cleaned_data.loc[0, 'close'], 1000)
        self.assertGreater(cleaned_data.loc[0, 'close'], 0)
    
    def test_setup_datetime_index(self):
        """Test datetime index setup"""
        indexed_data = self.preprocessor.setup_datetime_index(self.sample_data)
        
        # Should have datetime index
        self.assertIsInstance(indexed_data.index, pd.DatetimeIndex)
        
        # Should be sorted
        self.assertTrue(indexed_data.index.is_monotonic_increasing)
        
        # Should not have datetime column anymore
        self.assertNotIn('datetime', indexed_data.columns)
    
    def test_setup_datetime_index_with_duplicates(self):
        """Test datetime index setup with duplicate timestamps"""
        # Create data with duplicate timestamps
        data_with_duplicates = self.sample_data.copy()
        duplicate_row = data_with_duplicates.iloc[0:1].copy()
        data_with_duplicates = pd.concat([data_with_duplicates, duplicate_row], ignore_index=True)
        
        indexed_data = self.preprocessor.setup_datetime_index(data_with_duplicates)
        
        # Should not have duplicate indices
        self.assertFalse(indexed_data.index.duplicated().any())
    
    def test_fill_time_gaps(self):
        """Test time gap filling"""
        # Create data with gaps
        data_with_gaps = self.sample_data.copy()
        data_with_gaps = self.preprocessor.setup_datetime_index(data_with_gaps)
        
        # Remove some rows to create gaps
        data_with_gaps = data_with_gaps.drop(data_with_gaps.index[10:15])
        
        filled_data = self.preprocessor.fill_time_gaps(data_with_gaps, '1min')
        
        # Should have more rows than the gapped data
        self.assertGreater(len(filled_data), len(data_with_gaps))
        
        # Should have continuous time index
        expected_length = (filled_data.index.max() - filled_data.index.min()).total_seconds() / 60 + 1
        self.assertAlmostEqual(len(filled_data), expected_length, delta=5)  # Allow small tolerance
    
    def test_validate_data_integrity_success(self):
        """Test successful data integrity validation"""
        indexed_data = self.preprocessor.setup_datetime_index(self.sample_data)
        integrity_report = self.preprocessor.validate_data_integrity(indexed_data)
        
        self.assertIsInstance(integrity_report, dict)
        self.assertTrue(integrity_report['is_valid'])
        self.assertTrue(integrity_report['ohlc_consistency'])
        self.assertTrue(integrity_report['temporal_consistency'])
        self.assertTrue(integrity_report['volume_consistency'])
    
    def test_validate_data_integrity_ohlc_issues(self):
        """Test data integrity validation with OHLC issues"""
        # Create data with OHLC inconsistencies
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'high'] = 50  # High less than open/close
        invalid_data.loc[1, 'low'] = 150  # Low greater than open/close
        
        indexed_data = self.preprocessor.setup_datetime_index(invalid_data)
        integrity_report = self.preprocessor.validate_data_integrity(indexed_data)
        
        self.assertFalse(integrity_report['is_valid'])
        self.assertFalse(integrity_report['ohlc_consistency'])
        self.assertTrue(len(integrity_report['issues']) > 0)
    
    def test_validate_data_integrity_volume_issues(self):
        """Test data integrity validation with volume issues"""
        # Create data with volume issues
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'volume'] = -1000  # Negative volume
        
        indexed_data = self.preprocessor.setup_datetime_index(invalid_data)
        integrity_report = self.preprocessor.validate_data_integrity(indexed_data)
        
        self.assertFalse(integrity_report['is_valid'])
        self.assertFalse(integrity_report['volume_consistency'])
        self.assertTrue(any('negative volume' in issue.lower() for issue in integrity_report['issues']))
    
    def test_repair_data_ohlc_fixes(self):
        """Test data repair for OHLC inconsistencies"""
        # Create data with OHLC issues
        broken_data = self.sample_data.copy()
        broken_data.loc[0, 'high'] = 50  # High less than open/close
        broken_data.loc[1, 'low'] = 150  # Low greater than open/close
        broken_data.loc[2, 'high'] = 80   # High less than low
        broken_data.loc[2, 'low'] = 90
        
        repaired_data = self.preprocessor.repair_data(broken_data)
        
        # Check that repairs were made
        self.assertGreaterEqual(repaired_data.loc[0, 'high'], 
                               max(broken_data.loc[0, 'open'], broken_data.loc[0, 'close']))
        self.assertLessEqual(repaired_data.loc[1, 'low'], 
                            min(broken_data.loc[1, 'open'], broken_data.loc[1, 'close']))
        self.assertGreaterEqual(repaired_data.loc[2, 'high'], repaired_data.loc[2, 'low'])
    
    def test_repair_data_volume_fixes(self):
        """Test data repair for volume issues"""
        # Create data with negative volume
        broken_data = self.sample_data.copy()
        broken_data.loc[0, 'volume'] = -1000
        
        repaired_data = self.preprocessor.repair_data(broken_data)
        
        # Volume should be positive now
        self.assertGreater(repaired_data.loc[0, 'volume'], 0)
        self.assertEqual(repaired_data.loc[0, 'volume'], 1000)  # Should be absolute value
    
    def test_repair_data_extreme_price_changes(self):
        """Test data repair for extreme price changes"""
        # Create data with extreme price change
        broken_data = self.sample_data.copy()
        broken_data.loc[1, 'close'] = broken_data.loc[0, 'close'] * 2  # 100% increase
        
        # Configure with 20% max change
        config = {'price_change_cap': 0.20}
        preprocessor = DataPreprocessor(config=config)
        
        repaired_data = preprocessor.repair_data(broken_data)
        
        # Price change should be capped
        price_change = (repaired_data.loc[1, 'close'] / broken_data.loc[0, 'close']) - 1
        self.assertLessEqual(price_change, 0.20)
    
    def test_preprocess_pipeline_complete(self):
        """Test complete preprocessing pipeline"""
        # Create messy data
        messy_data = self.sample_data.copy()
        
        # Add various issues
        messy_data.loc[0:2, 'close'] = np.nan  # Missing values
        messy_data.loc[5, 'high'] = 50  # OHLC inconsistency
        messy_data.loc[10, 'volume'] = -1000  # Negative volume
        messy_data.loc[15, 'close'] = 1000  # Outlier
        
        # Run complete pipeline
        processed_data = self.preprocessor.preprocess_pipeline(messy_data, '1min')
        
        # Should be a clean DataFrame
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertIsInstance(processed_data.index, pd.DatetimeIndex)
        
        # Should not have missing values
        self.assertFalse(processed_data.isnull().any().any())
        
        # Should have reasonable data ranges
        self.assertTrue(all(processed_data['volume'] >= 0))
        self.assertTrue(all(processed_data['high'] >= processed_data['low']))
        
        # Should have consistent OHLC relationships
        self.assertTrue(all(processed_data['high'] >= processed_data[['open', 'close']].max(axis=1)))
        self.assertTrue(all(processed_data['low'] <= processed_data[['open', 'close']].min(axis=1)))
    
    def test_preprocess_pipeline_with_time_gaps(self):
        """Test preprocessing pipeline with time gaps"""
        # Create data with time gaps
        gapped_data = self.sample_data.copy()
        
        # Remove some rows to create gaps
        gapped_data = gapped_data.drop(gapped_data.index[20:30])
        
        processed_data = self.preprocessor.preprocess_pipeline(gapped_data, '1min')
        
        # Should fill the gaps
        self.assertGreater(len(processed_data), len(gapped_data))
        
        # Should have continuous time index
        time_diffs = processed_data.index.to_series().diff().dropna()
        expected_diff = pd.Timedelta('1min')
        
        # Most time differences should be 1 minute
        most_common_diff = time_diffs.mode().iloc[0]
        self.assertEqual(most_common_diff, expected_diff)
    
    def test_error_handling(self):
        """Test error handling in preprocessing methods"""
        # Test with invalid data
        invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
        
        # Should raise DataError for missing datetime
        with self.assertRaises(DataError):
            self.preprocessor.setup_datetime_index(invalid_data)
        
        # Should handle errors gracefully in integrity validation
        # Invalid data without OHLCV columns should still return a report
        integrity_report = self.preprocessor.validate_data_integrity(invalid_data)
        self.assertIsInstance(integrity_report, dict)
        self.assertIn('is_valid', integrity_report)
        
        # Test with data that has OHLCV columns but invalid values
        invalid_ohlcv = pd.DataFrame({
            'open': [100, -50],  # Negative price
            'high': [110, 60],
            'low': [90, 40],
            'close': [105, 55],
            'volume': [1000, -500]  # Negative volume
        })
        
        integrity_report = self.preprocessor.validate_data_integrity(invalid_ohlcv)
        self.assertFalse(integrity_report['is_valid'])


if __name__ == '__main__':
    unittest.main()