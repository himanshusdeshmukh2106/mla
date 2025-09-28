"""
Model evaluation and metrics system for XGBoost trading models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from xgboost import XGBClassifier

from ..interfaces import ModelMetrics
from ..exceptions import TradingSystemError


@dataclass
class DetailedModelMetrics:
    """Extended model evaluation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    average_precision: float
    confusion_matrix: np.ndarray
    feature_importance: Dict[str, float]
    classification_report: str
    predictions: np.ndarray
    prediction_probabilities: np.ndarray
    
    
@dataclass
class ConfusionMatrixAnalysis:
    """Detailed confusion matrix analysis"""
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    sensitivity: float  # True Positive Rate / Recall
    specificity: float  # True Negative Rate
    positive_predictive_value: float  # Precision
    negative_predictive_value: float
    false_positive_rate: float
    false_negative_rate: float
    

@dataclass
class FeatureImportanceAnalysis:
    """Feature importance analysis and ranking"""
    importance_scores: Dict[str, float]
    ranked_features: List[Tuple[str, float]]
    top_features: List[str]
    cumulative_importance: Dict[str, float]


class ModelEvaluationError(TradingSystemError):
    """Raised when model evaluation fails"""
    pass


class ModelEvaluator:
    """
    Comprehensive model evaluation and metrics system
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize model evaluator
        
        Args:
            feature_names: List of feature names for importance analysis
        """
        self.feature_names = feature_names
        self.logger = logging.getLogger(__name__)
        
    def evaluate_model(self, 
                      model: XGBClassifier,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      feature_names: Optional[List[str]] = None) -> DetailedModelMetrics:
        """
        Comprehensive model evaluation with detailed metrics
        
        Args:
            model: Trained XGBoost model
            X_test: Test features
            y_test: Test targets
            feature_names: Feature names for importance analysis
            
        Returns:
            DetailedModelMetrics with comprehensive evaluation results
        """
        if model is None:
            raise ModelEvaluationError("Model is None")
            
        if X_test is None or y_test is None:
            raise ModelEvaluationError("Test data is None")
            
        if len(X_test) == 0 or len(y_test) == 0:
            raise ModelEvaluationError("Test data is empty")
            
        if len(X_test) != len(y_test):
            raise ModelEvaluationError("X_test and y_test have different lengths")
            
        try:
            self.logger.info("Starting comprehensive model evaluation...")
            
            # Generate predictions and probabilities
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
            
            # Calculate basic classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            
            # Calculate advanced metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Generate classification report
            class_report = classification_report(y_test, y_pred, zero_division=0)
            
            # Extract feature importance
            feature_importance = self._extract_feature_importance(model, feature_names)
            
            self.logger.info(f"Model evaluation completed - Accuracy: {accuracy:.4f}, "
                           f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
                           f"F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
            
            return DetailedModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                roc_auc=roc_auc,
                average_precision=avg_precision,
                confusion_matrix=cm,
                feature_importance=feature_importance,
                classification_report=class_report,
                predictions=y_pred,
                prediction_probabilities=y_pred_proba
            )
            
        except Exception as e:
            raise ModelEvaluationError(f"Model evaluation failed: {str(e)}")
            
    def analyze_confusion_matrix(self, confusion_matrix: np.ndarray) -> ConfusionMatrixAnalysis:
        """
        Detailed analysis of confusion matrix
        
        Args:
            confusion_matrix: 2x2 confusion matrix
            
        Returns:
            ConfusionMatrixAnalysis with detailed metrics
        """
        if confusion_matrix.shape != (2, 2):
            raise ModelEvaluationError("Confusion matrix must be 2x2 for binary classification")
            
        try:
            # Extract values from confusion matrix
            tn, fp, fn, tp = confusion_matrix.ravel()
            
            # Calculate derived metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall/TPR
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # NPV
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # FPR
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # FNR
            
            self.logger.info(f"Confusion matrix analysis - TP: {tp}, TN: {tn}, "
                           f"FP: {fp}, FN: {fn}")
            
            return ConfusionMatrixAnalysis(
                true_positives=int(tp),
                true_negatives=int(tn),
                false_positives=int(fp),
                false_negatives=int(fn),
                sensitivity=sensitivity,
                specificity=specificity,
                positive_predictive_value=ppv,
                negative_predictive_value=npv,
                false_positive_rate=fpr,
                false_negative_rate=fnr
            )
            
        except Exception as e:
            raise ModelEvaluationError(f"Confusion matrix analysis failed: {str(e)}")
            
    def analyze_feature_importance(self, 
                                 model: XGBClassifier,
                                 feature_names: Optional[List[str]] = None,
                                 top_n: int = 10) -> FeatureImportanceAnalysis:
        """
        Analyze and rank feature importance
        
        Args:
            model: Trained XGBoost model
            feature_names: List of feature names
            top_n: Number of top features to include
            
        Returns:
            FeatureImportanceAnalysis with rankings and analysis
        """
        if not hasattr(model, 'feature_importances_'):
            raise ModelEvaluationError("Model does not have feature_importances_ attribute")
            
        try:
            # Extract feature importance scores
            importance_scores = self._extract_feature_importance(model, feature_names)
            
            # Check if we got any importance scores
            if not importance_scores:
                raise ModelEvaluationError("No feature importance scores available")
            
            # Rank features by importance
            ranked_features = sorted(
                importance_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Get top N features
            top_features = [feature for feature, _ in ranked_features[:top_n]]
            
            # Calculate cumulative importance
            cumulative_importance = {}
            cumulative_sum = 0.0
            for feature, importance in ranked_features:
                cumulative_sum += importance
                cumulative_importance[feature] = cumulative_sum
                
            self.logger.info(f"Feature importance analysis completed. "
                           f"Top feature: {ranked_features[0][0]} ({ranked_features[0][1]:.4f})")
            
            return FeatureImportanceAnalysis(
                importance_scores=importance_scores,
                ranked_features=ranked_features,
                top_features=top_features,
                cumulative_importance=cumulative_importance
            )
            
        except Exception as e:
            raise ModelEvaluationError(f"Feature importance analysis failed: {str(e)}")
            
    def generate_evaluation_report(self, 
                                 metrics: DetailedModelMetrics,
                                 cm_analysis: ConfusionMatrixAnalysis,
                                 feature_analysis: FeatureImportanceAnalysis) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            metrics: Detailed model metrics
            cm_analysis: Confusion matrix analysis
            feature_analysis: Feature importance analysis
            
        Returns:
            Formatted evaluation report string
        """
        try:
            report_lines = [
                "=" * 60,
                "MODEL EVALUATION REPORT",
                "=" * 60,
                "",
                "CLASSIFICATION METRICS:",
                f"  Accuracy:           {metrics.accuracy:.4f}",
                f"  Precision:          {metrics.precision:.4f}",
                f"  Recall:             {metrics.recall:.4f}",
                f"  F1-Score:           {metrics.f1_score:.4f}",
                f"  ROC-AUC:            {metrics.roc_auc:.4f}",
                f"  Average Precision:  {metrics.average_precision:.4f}",
                "",
                "CONFUSION MATRIX ANALYSIS:",
                f"  True Positives:     {cm_analysis.true_positives}",
                f"  True Negatives:     {cm_analysis.true_negatives}",
                f"  False Positives:    {cm_analysis.false_positives}",
                f"  False Negatives:    {cm_analysis.false_negatives}",
                f"  Sensitivity (TPR):  {cm_analysis.sensitivity:.4f}",
                f"  Specificity (TNR):  {cm_analysis.specificity:.4f}",
                f"  False Positive Rate: {cm_analysis.false_positive_rate:.4f}",
                f"  False Negative Rate: {cm_analysis.false_negative_rate:.4f}",
                "",
                "TOP 10 MOST IMPORTANT FEATURES:",
            ]
            
            # Add top features
            for i, (feature, importance) in enumerate(feature_analysis.ranked_features[:10], 1):
                report_lines.append(f"  {i:2d}. {feature:<20} {importance:.4f}")
                
            report_lines.extend([
                "",
                "DETAILED CLASSIFICATION REPORT:",
                metrics.classification_report,
                "=" * 60
            ])
            
            return "\n".join(report_lines)
            
        except Exception as e:
            raise ModelEvaluationError(f"Report generation failed: {str(e)}")
            
    def compare_models(self, 
                      models_metrics: Dict[str, DetailedModelMetrics]) -> pd.DataFrame:
        """
        Compare multiple models' performance metrics
        
        Args:
            models_metrics: Dictionary of model names to their metrics
            
        Returns:
            DataFrame with comparison of model metrics
        """
        if not models_metrics:
            raise ModelEvaluationError("No models provided for comparison")
            
        try:
            comparison_data = []
            
            for model_name, metrics in models_metrics.items():
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': metrics.accuracy,
                    'Precision': metrics.precision,
                    'Recall': metrics.recall,
                    'F1-Score': metrics.f1_score,
                    'ROC-AUC': metrics.roc_auc,
                    'Avg Precision': metrics.average_precision
                })
                
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.set_index('Model')
            
            self.logger.info(f"Model comparison completed for {len(models_metrics)} models")
            
            return comparison_df
            
        except Exception as e:
            raise ModelEvaluationError(f"Model comparison failed: {str(e)}")
            
    def _extract_feature_importance(self, 
                                  model: XGBClassifier,
                                  feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Extract feature importance from model
        
        Args:
            model: Trained XGBoost model
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not hasattr(model, 'feature_importances_'):
            return {}
            
        try:
            importance_values = model.feature_importances_
            
            # Use provided feature names or fall back to generic names
            if feature_names and len(feature_names) == len(importance_values):
                names = feature_names
            elif self.feature_names and len(self.feature_names) == len(importance_values):
                names = self.feature_names
            else:
                names = [f'feature_{i}' for i in range(len(importance_values))]
                
            return dict(zip(names, importance_values.tolist()))
        except (TypeError, AttributeError):
            # Handle cases where importance_values is not iterable or doesn't have len()
            return {}
        
    def plot_confusion_matrix(self, 
                            confusion_matrix: np.ndarray,
                            class_names: List[str] = None,
                            title: str = "Confusion Matrix") -> plt.Figure:
        """
        Plot confusion matrix heatmap
        
        Args:
            confusion_matrix: 2x2 confusion matrix
            class_names: Names for the classes
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if class_names is None:
            class_names = ['Negative', 'Positive']
            
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            sns.heatmap(
                confusion_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax
            )
            
            ax.set_title(title)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            raise ModelEvaluationError(f"Confusion matrix plotting failed: {str(e)}")
            
    def plot_feature_importance(self, 
                              feature_analysis: FeatureImportanceAnalysis,
                              top_n: int = 15,
                              title: str = "Feature Importance") -> plt.Figure:
        """
        Plot feature importance bar chart
        
        Args:
            feature_analysis: Feature importance analysis results
            top_n: Number of top features to plot
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        try:
            top_features = feature_analysis.ranked_features[:top_n]
            features, importances = zip(*top_features)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, importances, color='skyblue')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()  # Top feature at the top
            ax.set_xlabel('Importance Score')
            ax.set_title(title)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', ha='left', va='center')
                       
            plt.tight_layout()
            return fig
            
        except Exception as e:
            raise ModelEvaluationError(f"Feature importance plotting failed: {str(e)}")