"""
Evaluation Metrics Module
Comprehensive metrics and evaluation functions for regression and classification tasks.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from typing import Dict, Tuple, Optional
import warnings


class RegressionMetrics:
    """
    Calculate and format regression evaluation metrics.
    Designed for continuous value predictions (snowfall forecasting).
    """
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              threshold: float = 0.1) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            threshold: Threshold for binary snow detection (default 0.1 inches)
            
        Returns:
            Dictionary containing all regression metrics
        """
        # Ensure arrays are flat
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Basic regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Relative metrics
        mape = RegressionMetrics.mean_absolute_percentage_error(y_true, y_pred)
        
        # Variance metrics
        explained_variance = 1 - (np.var(y_true - y_pred) / np.var(y_true))
        
        # Binary snow detection metrics
        y_true_binary = (y_true > threshold).astype(int)
        y_pred_binary = (y_pred > threshold).astype(int)
        
        snow_accuracy = accuracy_score(y_true_binary, y_pred_binary)
        snow_precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        snow_recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        snow_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # Error distribution
        errors = y_pred - y_true
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Prediction bias
        bias = np.mean(y_pred) - np.mean(y_true)
        
        return {
            # Primary regression metrics
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'explained_variance': float(explained_variance),
            
            # Binary classification metrics
            'snow_accuracy': float(snow_accuracy),
            'snow_precision': float(snow_precision),
            'snow_recall': float(snow_recall),
            'snow_f1': float(snow_f1),
            
            # Error analysis
            'mean_error': float(mean_error),
            'std_error': float(std_error),
            'bias': float(bias),
            
            # Summary statistics
            'mean_true': float(np.mean(y_true)),
            'mean_pred': float(np.mean(y_pred)),
            'std_true': float(np.std(y_true)),
            'std_pred': float(np.std(y_pred)),
        }
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, 
                                       y_pred: np.ndarray,
                                       epsilon: float = 1e-10) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            epsilon: Small value to avoid division by zero
            
        Returns:
            MAPE value as percentage
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Avoid division by zero
        mask = np.abs(y_true) > epsilon
        
        if not np.any(mask):
            return 100.0  # All zeros, return 100% error
        
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return float(mape)
    
    @staticmethod
    def calculate_residuals(y_true: np.ndarray, 
                           y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate residual statistics for error analysis.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing residual statistics
        """
        residuals = y_pred - y_true
        
        return {
            'residuals': residuals,
            'abs_residuals': np.abs(residuals),
            'squared_residuals': residuals ** 2,
            'standardized_residuals': residuals / np.std(residuals) if np.std(residuals) > 0 else residuals,
        }
    
    @staticmethod
    def format_metrics_report(metrics: Dict[str, float]) -> str:
        """
        Format metrics dictionary into readable report.
        
        Args:
            metrics: Dictionary from calculate_all_metrics()
            
        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 70)
        report.append("REGRESSION EVALUATION METRICS")
        report.append("=" * 70)
        
        report.append("\nPrimary Metrics:")
        report.append(f"  Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
        report.append(f"  Mean Absolute Error (MAE):      {metrics['mae']:.4f}")
        report.append(f"  R² Score:                        {metrics['r2']:.4f}")
        report.append(f"  Mean Absolute % Error (MAPE):    {metrics['mape']:.2f}%")
        report.append(f"  Explained Variance:              {metrics['explained_variance']:.4f}")
        
        report.append("\nSnow Detection Metrics (Binary Classification):")
        report.append(f"  Accuracy:  {metrics['snow_accuracy']:.4f} ({metrics['snow_accuracy']*100:.2f}%)")
        report.append(f"  Precision: {metrics['snow_precision']:.4f}")
        report.append(f"  Recall:    {metrics['snow_recall']:.4f}")
        report.append(f"  F1-Score:  {metrics['snow_f1']:.4f}")
        
        report.append("\nError Analysis:")
        report.append(f"  Mean Error (Bias):     {metrics['mean_error']:.4f}")
        report.append(f"  Std Error:             {metrics['std_error']:.4f}")
        report.append(f"  Overall Bias:          {metrics['bias']:.4f}")
        
        report.append("\nPrediction Statistics:")
        report.append(f"  Mean True:       {metrics['mean_true']:.4f}")
        report.append(f"  Mean Predicted:  {metrics['mean_pred']:.4f}")
        report.append(f"  Std True:        {metrics['std_true']:.4f}")
        report.append(f"  Std Predicted:   {metrics['std_pred']:.4f}")
        
        report.append("=" * 70)
        
        return "\n".join(report)


class ClassificationMetrics:
    """
    Calculate and format classification evaluation metrics.
    Designed for severity classification (Mild/Snowy/Severe).
    """
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              class_names: Optional[list] = None) -> Dict:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_names: Optional list of class names for display
            
        Returns:
            Dictionary containing all classification metrics
        """
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(np.unique(y_true)))]
        
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Multi-class metrics (macro average)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Multi-class metrics (weighted average)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
        
        return {
            # Overall metrics
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            
            # Per-class metrics
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            
            # Confusion matrix
            'confusion_matrix': cm.tolist(),
            
            # Classification report
            'classification_report': report,
            
            # Class names
            'class_names': class_names
        }
    
    @staticmethod
    def calculate_confusion_matrix_metrics(cm: np.ndarray) -> Dict[str, Dict]:
        """
        Calculate detailed metrics from confusion matrix.
        
        Args:
            cm: Confusion matrix (n_classes x n_classes)
            
        Returns:
            Dictionary with per-class TP, TN, FP, FN
        """
        n_classes = cm.shape[0]
        metrics = {}
        
        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            
            metrics[f"class_{i}"] = {
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn),
                'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            }
        
        return metrics
    
    @staticmethod
    def format_metrics_report(metrics: Dict) -> str:
        """
        Format classification metrics into readable report.
        
        Args:
            metrics: Dictionary from calculate_all_metrics()
            
        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 70)
        report.append("CLASSIFICATION EVALUATION METRICS")
        report.append("=" * 70)
        
        report.append(f"\nOverall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        
        report.append("\nMacro-Averaged Metrics:")
        report.append(f"  Precision: {metrics['precision_macro']:.4f}")
        report.append(f"  Recall:    {metrics['recall_macro']:.4f}")
        report.append(f"  F1-Score:  {metrics['f1_macro']:.4f}")
        
        report.append("\nWeighted-Averaged Metrics:")
        report.append(f"  Precision: {metrics['precision_weighted']:.4f}")
        report.append(f"  Recall:    {metrics['recall_weighted']:.4f}")
        report.append(f"  F1-Score:  {metrics['f1_weighted']:.4f}")
        
        report.append("\nPer-Class Metrics:")
        for i, name in enumerate(metrics['class_names']):
            report.append(f"\n  {name}:")
            report.append(f"    Precision: {metrics['precision_per_class'][i]:.4f}")
            report.append(f"    Recall:    {metrics['recall_per_class'][i]:.4f}")
            report.append(f"    F1-Score:  {metrics['f1_per_class'][i]:.4f}")
        
        report.append("\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        
        # Header
        header = "       " + "  ".join([f"{name:>8}" for name in metrics['class_names']])
        report.append(header)
        
        # Rows
        for i, name in enumerate(metrics['class_names']):
            row = f"{name:>6} " + "  ".join([f"{cm[i,j]:>8}" for j in range(len(metrics['class_names']))])
            report.append(row)
        
        report.append("\nDetailed Classification Report:")
        report.append(metrics['classification_report'])
        
        report.append("=" * 70)
        
        return "\n".join(report)


class TimeSeriesMetrics:
    """
    Specialized metrics for time-series forecasting evaluation.
    """
    
    @staticmethod
    def calculate_forecast_accuracy(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    dates: Optional[pd.DatetimeIndex] = None) -> Dict:
        """
        Calculate time-series specific accuracy metrics.
        
        Args:
            y_true: Ground truth time series
            y_pred: Predicted time series
            dates: Optional datetime index
            
        Returns:
            Dictionary with time-series metrics
        """
        # Basic metrics
        basic_metrics = RegressionMetrics.calculate_all_metrics(y_true, y_pred)
        
        # Direction accuracy (did we predict increase/decrease correctly?)
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            direction_accuracy = np.mean(true_direction == pred_direction)
        else:
            direction_accuracy = 0.0
        
        # Forecast horizon analysis (if dates provided)
        horizon_metrics = {}
        if dates is not None:
            # Group by day of week, month, etc.
            df = pd.DataFrame({
                'true': y_true,
                'pred': y_pred,
                'date': dates
            })
            
            # By month
            df['month'] = df['date'].dt.month
            monthly_rmse = df.groupby('month').apply(
                lambda x: np.sqrt(mean_squared_error(x['true'], x['pred']))
            ).to_dict()
            
            horizon_metrics['monthly_rmse'] = monthly_rmse
        
        return {
            **basic_metrics,
            'direction_accuracy': float(direction_accuracy),
            **horizon_metrics
        }
    
    @staticmethod
    def calculate_persistence_baseline(y_true: np.ndarray) -> float:
        """
        Calculate persistence forecast baseline (yesterday's value).
        
        Args:
            y_true: Ground truth time series
            
        Returns:
            RMSE of persistence forecast
        """
        if len(y_true) < 2:
            return 0.0
        
        # Persistence forecast: t+1 = t
        persistence_pred = y_true[:-1]
        persistence_true = y_true[1:]
        
        rmse = np.sqrt(mean_squared_error(persistence_true, persistence_pred))
        return float(rmse)


def compare_models(models_predictions: Dict[str, np.ndarray],
                  y_true: np.ndarray) -> pd.DataFrame:
    """
    Compare multiple models side-by-side.
    
    Args:
        models_predictions: Dictionary of {model_name: predictions}
        y_true: Ground truth values
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison = []
    
    for model_name, y_pred in models_predictions.items():
        metrics = RegressionMetrics.calculate_all_metrics(y_true, y_pred)
        comparison.append({
            'Model': model_name,
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'R²': metrics['r2'],
            'Snow Accuracy': metrics['snow_accuracy'],
            'Bias': metrics['bias']
        })
    
    df = pd.DataFrame(comparison)
    df = df.sort_values('RMSE')  # Sort by RMSE (best first)
    
    return df


def print_model_comparison(comparison_df: pd.DataFrame) -> None:
    """
    Print formatted model comparison table.
    
    Args:
        comparison_df: DataFrame from compare_models()
    """
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print()
    print(comparison_df.to_string(index=False))
    print()
    print("=" * 70)
    print("Lower RMSE and MAE are better.")
    print("Higher R² and Snow Accuracy are better.")
    print("Bias close to 0 is better (no systematic over/under prediction).")
    print("=" * 70)


if __name__ == "__main__":
    """
    Example usage and testing of evaluation metrics.
    """
    print("Testing Evaluation Metrics Module")
    print("=" * 70)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    # Regression example
    print("\n1. REGRESSION METRICS TEST")
    print("-" * 70)
    
    y_true_reg = np.random.gamma(2, 0.5, n_samples)  # Simulated snowfall
    y_pred_reg = y_true_reg + np.random.normal(0, 0.3, n_samples)  # Add noise
    
    reg_metrics = RegressionMetrics.calculate_all_metrics(y_true_reg, y_pred_reg)
    print(RegressionMetrics.format_metrics_report(reg_metrics))
    
    # Classification example
    print("\n\n2. CLASSIFICATION METRICS TEST")
    print("-" * 70)
    
    y_true_class = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])
    y_pred_class = y_true_class.copy()
    # Add some errors
    error_idx = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    y_pred_class[error_idx] = np.random.choice([0, 1, 2], len(error_idx))
    
    class_metrics = ClassificationMetrics.calculate_all_metrics(
        y_true_class, 
        y_pred_class,
        class_names=['Mild', 'Snowy', 'Severe']
    )
    print(ClassificationMetrics.format_metrics_report(class_metrics))
    
    # Model comparison example
    print("\n\n3. MODEL COMPARISON TEST")
    print("-" * 70)
    
    models = {
        'LSTM': y_pred_reg,
        'Baseline': np.full(n_samples, np.mean(y_true_reg)),
        'Persistence': np.roll(y_true_reg, 1)
    }
    
    comparison = compare_models(models, y_true_reg)
    print_model_comparison(comparison)
    
    print("\n\nAll tests completed successfully!")