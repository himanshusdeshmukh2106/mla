"""
Unit tests for ModelEvaluator class
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt

from src.models.evaluator import (
    ModelEvaluator, DetailedModelMetrics, ConfusionMatrixAnalysis,
    FeatureImportanceAnalysis, ModelEvaluationError
)
from xgboost import XGBClassifier


class TestModelEvaluator:
    """Test ModelEvaluator class"""
    
    @pytest.fixture
    def sample_test_data(self):
        """Create sample test data"""
        np.random.seed(42)
        n_samples = 200
        
        X_test = np.random.randn(n_samples, 4)
        y_test = np.random.randint(0, 2, n_samples)
        
        return X_test, y_test
        
    @pytest.fixture
    def mock_model(self):
        """Create mock XGBoost model"""
        model = Mock(spec=XGBClassifier)
        
        # Mock predictions
        np.random.seed(42)
        n_samples = 200
        y_pred = np.random.randint(0, 2, n_samples)
        y_pred_proba = np.random.random((n_samples, 2))
        y_pred_proba[:, 1] = np.random.random(n_samples)  # Positive class probabilities
        y_pred_proba[:, 0] = 1 - y_pred_proba[:, 1]  # Negative class probabilities
        
        model.predict.return_value = y_pred
        model.predict_proba.return_value = y_pred_proba
        
        # Mock feature importance
        feature_importances = np.array([0.4, 0.3, 0.2, 0.1])
        model.feature_importances_ = feature_importances
        
        return model
        
    @pytest.fixture
    def feature_names(self):
        """Sample feature names"""
        return ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        
    @pytest.fixture
    def evaluator(self, feature_names):
        """Create ModelEvaluator instance"""
        return ModelEvaluator(feature_names=feature_names)
        
    def test_initialization(self):
        """Test ModelEvaluator initialization"""
        evaluator = ModelEvaluator()
        
        assert evaluator.feature_names is None
        assert evaluator.logger is not None
        
    def test_initialization_with_feature_names(self, feature_names):
        """Test ModelEvaluator initialization with feature names"""
        evaluator = ModelEvaluator(feature_names=feature_names)
        
        assert evaluator.feature_names == feature_names
        
    def test_evaluate_model_success(self, evaluator, mock_model, sample_test_data, feature_names):
        """Test successful model evaluation"""
        X_test, y_test = sample_test_data
        
        metrics = evaluator.evaluate_model(mock_model, X_test, y_test, feature_names)
        
        assert isinstance(metrics, DetailedModelMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1_score <= 1
        assert 0 <= metrics.roc_auc <= 1
        assert 0 <= metrics.average_precision <= 1
        assert metrics.confusion_matrix.shape == (2, 2)
        assert isinstance(metrics.feature_importance, dict)
        assert len(metrics.feature_importance) == 4
        assert isinstance(metrics.classification_report, str)
        assert len(metrics.predictions) == len(y_test)
        assert len(metrics.prediction_probabilities) == len(y_test)
        
        # Verify mock calls
        mock_model.predict.assert_called_once_with(X_test)
        mock_model.predict_proba.assert_called_once_with(X_test)
        
    def test_evaluate_model_none_model(self, evaluator, sample_test_data):
        """Test model evaluation with None model"""
        X_test, y_test = sample_test_data
        
        with pytest.raises(ModelEvaluationError, match="Model is None"):
            evaluator.evaluate_model(None, X_test, y_test)
            
    def test_evaluate_model_none_data(self, evaluator, mock_model):
        """Test model evaluation with None data"""
        with pytest.raises(ModelEvaluationError, match="Test data is None"):
            evaluator.evaluate_model(mock_model, None, None)
            
    def test_evaluate_model_empty_data(self, evaluator, mock_model):
        """Test model evaluation with empty data"""
        X_test = np.array([])
        y_test = np.array([])
        
        with pytest.raises(ModelEvaluationError, match="Test data is empty"):
            evaluator.evaluate_model(mock_model, X_test, y_test)
            
    def test_evaluate_model_mismatched_data_lengths(self, evaluator, mock_model):
        """Test model evaluation with mismatched data lengths"""
        X_test = np.random.randn(100, 4)
        y_test = np.random.randint(0, 2, 50)  # Different length
        
        with pytest.raises(ModelEvaluationError, match="X_test and y_test have different lengths"):
            evaluator.evaluate_model(mock_model, X_test, y_test)
            
    def test_evaluate_model_prediction_failure(self, evaluator, sample_test_data):
        """Test model evaluation when prediction fails"""
        X_test, y_test = sample_test_data
        
        # Create mock model that raises exception
        mock_model = Mock(spec=XGBClassifier)
        mock_model.predict.side_effect = Exception("Prediction failed")
        
        with pytest.raises(ModelEvaluationError, match="Model evaluation failed"):
            evaluator.evaluate_model(mock_model, X_test, y_test)
            
    def test_analyze_confusion_matrix_success(self, evaluator):
        """Test successful confusion matrix analysis"""
        # Create a sample confusion matrix
        cm = np.array([[50, 10], [5, 35]])
        
        analysis = evaluator.analyze_confusion_matrix(cm)
        
        assert isinstance(analysis, ConfusionMatrixAnalysis)
        assert analysis.true_negatives == 50
        assert analysis.false_positives == 10
        assert analysis.false_negatives == 5
        assert analysis.true_positives == 35
        
        # Check calculated metrics
        assert analysis.sensitivity == 35 / (35 + 5)  # TP / (TP + FN)
        assert analysis.specificity == 50 / (50 + 10)  # TN / (TN + FP)
        assert analysis.positive_predictive_value == 35 / (35 + 10)  # TP / (TP + FP)
        assert analysis.negative_predictive_value == 50 / (50 + 5)  # TN / (TN + FN)
        assert analysis.false_positive_rate == 10 / (10 + 50)  # FP / (FP + TN)
        assert analysis.false_negative_rate == 5 / (5 + 35)  # FN / (FN + TP)
        
    def test_analyze_confusion_matrix_wrong_shape(self, evaluator):
        """Test confusion matrix analysis with wrong shape"""
        cm = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 3x3 matrix
        
        with pytest.raises(ModelEvaluationError, match="Confusion matrix must be 2x2"):
            evaluator.analyze_confusion_matrix(cm)
            
    def test_analyze_confusion_matrix_zero_division(self, evaluator):
        """Test confusion matrix analysis with zero division cases"""
        # All predictions are negative
        cm = np.array([[100, 0], [0, 0]])
        
        analysis = evaluator.analyze_confusion_matrix(cm)
        
        assert analysis.sensitivity == 0.0  # No true positives
        assert analysis.positive_predictive_value == 0.0  # No predicted positives
        assert analysis.false_negative_rate == 0.0  # No actual positives
        
    def test_analyze_feature_importance_success(self, evaluator, mock_model, feature_names):
        """Test successful feature importance analysis"""
        analysis = evaluator.analyze_feature_importance(mock_model, feature_names, top_n=3)
        
        assert isinstance(analysis, FeatureImportanceAnalysis)
        assert len(analysis.importance_scores) == 4
        assert len(analysis.ranked_features) == 4
        assert len(analysis.top_features) == 3
        assert len(analysis.cumulative_importance) == 4
        
        # Check that features are ranked in descending order
        importances = [importance for _, importance in analysis.ranked_features]
        assert importances == sorted(importances, reverse=True)
        
        # Check cumulative importance (allow for floating point precision)
        assert abs(analysis.cumulative_importance[analysis.ranked_features[-1][0]] - 1.0) < 1e-10
        
    def test_analyze_feature_importance_no_feature_importances(self, evaluator):
        """Test feature importance analysis with model without feature_importances_"""
        mock_model = Mock(spec=[])  # Empty spec means no attributes
        
        with pytest.raises(ModelEvaluationError, match="does not have feature_importances_"):
            evaluator.analyze_feature_importance(mock_model)
            
    def test_analyze_feature_importance_without_names(self, mock_model):
        """Test feature importance analysis without feature names"""
        # Create evaluator without feature names
        evaluator_no_names = ModelEvaluator(feature_names=None)
        analysis = evaluator_no_names.analyze_feature_importance(mock_model, feature_names=None)
        
        assert isinstance(analysis, FeatureImportanceAnalysis)
        # Should use generic feature names
        expected_names = ['feature_0', 'feature_1', 'feature_2', 'feature_3']
        assert list(analysis.importance_scores.keys()) == expected_names
        
    def test_generate_evaluation_report(self, evaluator):
        """Test evaluation report generation"""
        # Create sample metrics
        metrics = DetailedModelMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            roc_auc=0.82,
            average_precision=0.78,
            confusion_matrix=np.array([[50, 10], [5, 35]]),
            feature_importance={'feature_1': 0.4, 'feature_2': 0.3},
            classification_report="Sample report",
            predictions=np.array([0, 1, 0, 1]),
            prediction_probabilities=np.array([0.2, 0.8, 0.3, 0.9])
        )
        
        cm_analysis = ConfusionMatrixAnalysis(
            true_positives=35, true_negatives=50,
            false_positives=10, false_negatives=5,
            sensitivity=0.875, specificity=0.833,
            positive_predictive_value=0.778, negative_predictive_value=0.909,
            false_positive_rate=0.167, false_negative_rate=0.125
        )
        
        feature_analysis = FeatureImportanceAnalysis(
            importance_scores={'feature_1': 0.4, 'feature_2': 0.3},
            ranked_features=[('feature_1', 0.4), ('feature_2', 0.3)],
            top_features=['feature_1', 'feature_2'],
            cumulative_importance={'feature_1': 0.4, 'feature_2': 0.7}
        )
        
        report = evaluator.generate_evaluation_report(metrics, cm_analysis, feature_analysis)
        
        assert isinstance(report, str)
        assert "MODEL EVALUATION REPORT" in report
        assert "CLASSIFICATION METRICS:" in report
        assert "CONFUSION MATRIX ANALYSIS:" in report
        assert "TOP 10 MOST IMPORTANT FEATURES:" in report
        assert "0.85" in report  # Accuracy
        assert "feature_1" in report
        
    def test_compare_models(self, evaluator):
        """Test model comparison functionality"""
        # Create sample metrics for multiple models
        metrics1 = DetailedModelMetrics(
            accuracy=0.85, precision=0.80, recall=0.75, f1_score=0.77,
            roc_auc=0.82, average_precision=0.78,
            confusion_matrix=np.array([[50, 10], [5, 35]]),
            feature_importance={}, classification_report="",
            predictions=np.array([]), prediction_probabilities=np.array([])
        )
        
        metrics2 = DetailedModelMetrics(
            accuracy=0.88, precision=0.85, recall=0.80, f1_score=0.82,
            roc_auc=0.87, average_precision=0.83,
            confusion_matrix=np.array([[55, 5], [3, 37]]),
            feature_importance={}, classification_report="",
            predictions=np.array([]), prediction_probabilities=np.array([])
        )
        
        models_metrics = {
            'Model_A': metrics1,
            'Model_B': metrics2
        }
        
        comparison_df = evaluator.compare_models(models_metrics)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert 'Model_A' in comparison_df.index
        assert 'Model_B' in comparison_df.index
        assert 'Accuracy' in comparison_df.columns
        assert 'Precision' in comparison_df.columns
        assert 'Recall' in comparison_df.columns
        assert 'F1-Score' in comparison_df.columns
        assert 'ROC-AUC' in comparison_df.columns
        assert 'Avg Precision' in comparison_df.columns
        
        # Check values
        assert comparison_df.loc['Model_A', 'Accuracy'] == 0.85
        assert comparison_df.loc['Model_B', 'Accuracy'] == 0.88
        
    def test_compare_models_empty_dict(self, evaluator):
        """Test model comparison with empty dictionary"""
        with pytest.raises(ModelEvaluationError, match="No models provided for comparison"):
            evaluator.compare_models({})
            
    def test_extract_feature_importance_with_names(self, evaluator, mock_model, feature_names):
        """Test feature importance extraction with provided names"""
        importance_dict = evaluator._extract_feature_importance(mock_model, feature_names)
        
        assert isinstance(importance_dict, dict)
        assert len(importance_dict) == 4
        assert list(importance_dict.keys()) == feature_names
        assert all(isinstance(v, float) for v in importance_dict.values())
        
    def test_extract_feature_importance_without_names(self, mock_model):
        """Test feature importance extraction without names"""
        # Create evaluator without feature names
        evaluator_no_names = ModelEvaluator(feature_names=None)
        importance_dict = evaluator_no_names._extract_feature_importance(mock_model, None)
        
        assert isinstance(importance_dict, dict)
        assert len(importance_dict) == 4
        expected_names = ['feature_0', 'feature_1', 'feature_2', 'feature_3']
        assert list(importance_dict.keys()) == expected_names
        
    def test_extract_feature_importance_no_attribute(self):
        """Test feature importance extraction from model without attribute"""
        evaluator_no_names = ModelEvaluator(feature_names=None)
        mock_model = Mock()
        # Don't set feature_importances_ attribute
        
        importance_dict = evaluator_no_names._extract_feature_importance(mock_model, None)
        
        assert importance_dict == {}
        
    @patch('matplotlib.pyplot.subplots')
    @patch('seaborn.heatmap')
    def test_plot_confusion_matrix(self, mock_heatmap, mock_subplots, evaluator):
        """Test confusion matrix plotting"""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        cm = np.array([[50, 10], [5, 35]])
        
        fig = evaluator.plot_confusion_matrix(cm)
        
        assert fig == mock_fig
        mock_subplots.assert_called_once_with(figsize=(8, 6))
        mock_heatmap.assert_called_once()
        mock_ax.set_title.assert_called_once()
        mock_ax.set_xlabel.assert_called_once()
        mock_ax.set_ylabel.assert_called_once()
        
    @patch('matplotlib.pyplot.subplots')
    def test_plot_feature_importance(self, mock_subplots, evaluator):
        """Test feature importance plotting"""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Mock barh method
        mock_bars = [Mock() for _ in range(3)]
        for i, bar in enumerate(mock_bars):
            bar.get_width.return_value = 0.1 * (i + 1)
            bar.get_y.return_value = i
            bar.get_height.return_value = 0.8
        mock_ax.barh.return_value = mock_bars
        
        feature_analysis = FeatureImportanceAnalysis(
            importance_scores={'feature_1': 0.4, 'feature_2': 0.3, 'feature_3': 0.2},
            ranked_features=[('feature_1', 0.4), ('feature_2', 0.3), ('feature_3', 0.2)],
            top_features=['feature_1', 'feature_2', 'feature_3'],
            cumulative_importance={'feature_1': 0.4, 'feature_2': 0.7, 'feature_3': 0.9}
        )
        
        fig = evaluator.plot_feature_importance(feature_analysis, top_n=3)
        
        assert fig == mock_fig
        mock_subplots.assert_called_once_with(figsize=(10, 8))
        mock_ax.barh.assert_called_once()
        mock_ax.set_yticks.assert_called_once()
        mock_ax.set_yticklabels.assert_called_once()
        mock_ax.invert_yaxis.assert_called_once()
        mock_ax.set_xlabel.assert_called_once()
        mock_ax.set_title.assert_called_once()


class TestDetailedModelMetrics:
    """Test DetailedModelMetrics dataclass"""
    
    def test_creation(self):
        """Test DetailedModelMetrics creation"""
        metrics = DetailedModelMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            roc_auc=0.82,
            average_precision=0.78,
            confusion_matrix=np.array([[50, 10], [5, 35]]),
            feature_importance={'feature_1': 0.4},
            classification_report="Sample report",
            predictions=np.array([0, 1, 0, 1]),
            prediction_probabilities=np.array([0.2, 0.8, 0.3, 0.9])
        )
        
        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.80
        assert metrics.recall == 0.75
        assert metrics.f1_score == 0.77
        assert metrics.roc_auc == 0.82
        assert metrics.average_precision == 0.78
        assert metrics.confusion_matrix.shape == (2, 2)
        assert metrics.feature_importance == {'feature_1': 0.4}
        assert metrics.classification_report == "Sample report"
        assert len(metrics.predictions) == 4
        assert len(metrics.prediction_probabilities) == 4


class TestConfusionMatrixAnalysis:
    """Test ConfusionMatrixAnalysis dataclass"""
    
    def test_creation(self):
        """Test ConfusionMatrixAnalysis creation"""
        analysis = ConfusionMatrixAnalysis(
            true_positives=35,
            true_negatives=50,
            false_positives=10,
            false_negatives=5,
            sensitivity=0.875,
            specificity=0.833,
            positive_predictive_value=0.778,
            negative_predictive_value=0.909,
            false_positive_rate=0.167,
            false_negative_rate=0.125
        )
        
        assert analysis.true_positives == 35
        assert analysis.true_negatives == 50
        assert analysis.false_positives == 10
        assert analysis.false_negatives == 5
        assert analysis.sensitivity == 0.875
        assert analysis.specificity == 0.833
        assert analysis.positive_predictive_value == 0.778
        assert analysis.negative_predictive_value == 0.909
        assert analysis.false_positive_rate == 0.167
        assert analysis.false_negative_rate == 0.125


class TestFeatureImportanceAnalysis:
    """Test FeatureImportanceAnalysis dataclass"""
    
    def test_creation(self):
        """Test FeatureImportanceAnalysis creation"""
        analysis = FeatureImportanceAnalysis(
            importance_scores={'feature_1': 0.4, 'feature_2': 0.3},
            ranked_features=[('feature_1', 0.4), ('feature_2', 0.3)],
            top_features=['feature_1', 'feature_2'],
            cumulative_importance={'feature_1': 0.4, 'feature_2': 0.7}
        )
        
        assert analysis.importance_scores == {'feature_1': 0.4, 'feature_2': 0.3}
        assert analysis.ranked_features == [('feature_1', 0.4), ('feature_2', 0.3)]
        assert analysis.top_features == ['feature_1', 'feature_2']
        assert analysis.cumulative_importance == {'feature_1': 0.4, 'feature_2': 0.7}


class TestModelEvaluatorIntegration:
    """Integration tests for ModelEvaluator"""
    
    @pytest.fixture
    def real_model_and_data(self):
        """Create real XGBoost model and data for integration testing"""
        from xgboost import XGBClassifier
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 1000
        n_features = 5
        
        X = np.random.randn(n_samples, n_features)
        # Create target with some relationship to features
        y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
        
        # Split data
        split_idx = int(n_samples * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        model = XGBClassifier(
            n_estimators=10,  # Small for fast testing
            max_depth=3,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        return model, X_test, y_test, feature_names
        
    def test_full_evaluation_pipeline(self, real_model_and_data):
        """Test complete evaluation pipeline with real model"""
        model, X_test, y_test, feature_names = real_model_and_data
        
        evaluator = ModelEvaluator(feature_names=feature_names)
        
        # Evaluate model
        metrics = evaluator.evaluate_model(model, X_test, y_test, feature_names)
        
        # Analyze confusion matrix
        cm_analysis = evaluator.analyze_confusion_matrix(metrics.confusion_matrix)
        
        # Analyze feature importance
        feature_analysis = evaluator.analyze_feature_importance(model, feature_names)
        
        # Generate report
        report = evaluator.generate_evaluation_report(metrics, cm_analysis, feature_analysis)
        
        # Verify results
        assert isinstance(metrics, DetailedModelMetrics)
        assert isinstance(cm_analysis, ConfusionMatrixAnalysis)
        assert isinstance(feature_analysis, FeatureImportanceAnalysis)
        assert isinstance(report, str)
        
        # Check that metrics are reasonable
        assert 0.4 <= metrics.accuracy <= 1.0  # Should be decent accuracy
        assert 0 <= metrics.roc_auc <= 1.0
        assert len(feature_analysis.ranked_features) == 5
        
        # Check that report contains expected sections
        assert "MODEL EVALUATION REPORT" in report
        assert "CLASSIFICATION METRICS:" in report
        assert "CONFUSION MATRIX ANALYSIS:" in report
        assert "TOP 10 MOST IMPORTANT FEATURES:" in report
        
    def test_model_comparison_integration(self, real_model_and_data):
        """Test model comparison with real models"""
        model, X_test, y_test, feature_names = real_model_and_data
        
        # Create second model with different parameters
        model2 = XGBClassifier(
            n_estimators=5,  # Different parameters
            max_depth=2,
            random_state=42,
            verbosity=0
        )
        
        # Train on same training data (we'll use test data as proxy for training)
        model2.fit(X_test[:100], y_test[:100])  # Use part of test data for training
        
        evaluator = ModelEvaluator(feature_names=feature_names)
        
        # Evaluate both models
        metrics1 = evaluator.evaluate_model(model, X_test, y_test, feature_names)
        metrics2 = evaluator.evaluate_model(model2, X_test, y_test, feature_names)
        
        # Compare models
        comparison_df = evaluator.compare_models({
            'Model_1': metrics1,
            'Model_2': metrics2
        })
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert 'Model_1' in comparison_df.index
        assert 'Model_2' in comparison_df.index
        
        # Check that all metrics are present and reasonable
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Avg Precision']:
            assert metric in comparison_df.columns
            assert 0 <= comparison_df.loc['Model_1', metric] <= 1
            assert 0 <= comparison_df.loc['Model_2', metric] <= 1


if __name__ == '__main__':
    pytest.main([__file__])