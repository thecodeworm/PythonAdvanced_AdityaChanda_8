"""
Evaluation Plots Module
Comprehensive visualization functions for model evaluation and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
from pathlib import Path
import warnings

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class RegressionPlots:
    """
    Visualization functions for regression model evaluation.
    """
    
    @staticmethod
    def plot_predictions_vs_actual(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   dates: Optional[pd.DatetimeIndex] = None,
                                   title: str = "Predictions vs Actual",
                                   save_path: Optional[str] = None) -> None:
        """
        Plot predicted vs actual values over time.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            dates: Optional datetime index
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Time series plot
        if dates is not None:
            x = dates
            ax1.set_xlabel('Date', fontsize=12)
        else:
            x = np.arange(len(y_true))
            ax1.set_xlabel('Sample Index', fontsize=12)
        
        ax1.plot(x, y_true, 'o-', label='Actual', linewidth=2, markersize=4, alpha=0.7)
        ax1.plot(x, y_pred, 'x-', label='Predicted', linewidth=2, markersize=4, alpha=0.7)
        ax1.set_ylabel('Snowfall (inches)', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        if dates is not None:
            ax1.tick_params(axis='x', rotation=45)
        
        # Scatter plot
        max_val = max(y_true.max(), y_pred.max())
        ax2.scatter(y_true, y_pred, alpha=0.6, s=50)
        ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Snowfall (inches)', fontsize=12)
        ax2.set_ylabel('Predicted Snowfall (inches)', fontsize=12)
        ax2.set_title('Prediction Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Add R² to scatter plot
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        ax2.text(0.05, 0.95, f'R² = {r2:.3f}', 
                transform=ax2.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_residuals(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      title: str = "Residual Analysis",
                      save_path: Optional[str] = None) -> None:
        """
        Plot residual analysis (4 subplots).
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            title: Plot title
            save_path: Optional path to save figure
        """
        residuals = y_pred - y_true
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted Values', fontsize=11)
        axes[0, 0].set_ylabel('Residuals', fontsize=11)
        axes[0, 0].set_title('Residuals vs Predicted', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals vs Actual
        axes[0, 1].scatter(y_true, residuals, alpha=0.6, color='orange')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Actual Values', fontsize=11)
        axes[0, 1].set_ylabel('Residuals', fontsize=11)
        axes[0, 1].set_title('Residuals vs Actual', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residual distribution
        axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Residual Value', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].set_title('Residual Distribution', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add stats
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        axes[1, 0].text(0.05, 0.95, f'Mean: {mean_res:.3f}\nStd: {std_res:.3f}',
                       transform=axes[1, 0].transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_error_distribution(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               bins: int = 30,
                               save_path: Optional[str] = None) -> None:
        """
        Plot error magnitude distribution by snowfall amount.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            bins: Number of bins for histogram
            save_path: Optional path to save figure
        """
        errors = np.abs(y_pred - y_true)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Error histogram
        axes[0].hist(errors, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Absolute Error (inches)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Error Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        axes[0].axvline(mean_error, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.3f}')
        axes[0].axvline(median_error, color='g', linestyle='--', linewidth=2, label=f'Median: {median_error:.3f}')
        axes[0].legend()
        
        # Error vs magnitude
        axes[1].scatter(y_true, errors, alpha=0.6, s=50)
        axes[1].set_xlabel('Actual Snowfall (inches)', fontsize=12)
        axes[1].set_ylabel('Absolute Error (inches)', fontsize=12)
        axes[1].set_title('Error by Snowfall Amount', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(y_true.flatten(), errors.flatten(), 1)
        p = np.poly1d(z)
        x_trend = np.linspace(y_true.min(), y_true.max(), 100)
        axes[1].plot(x_trend, p(x_trend), 'r--', linewidth=2, label='Trend')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


class ClassificationPlots:
    """
    Visualization functions for classification model evaluation.
    """
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray,
                             class_names: List[str],
                             title: str = "Confusion Matrix",
                             save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix as heatmap.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add accuracy
        accuracy = np.trace(cm) / np.sum(cm)
        ax.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.2%}',
               ha='center', transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_classification_report(metrics_dict: dict,
                                   save_path: Optional[str] = None) -> None:
        """
        Visualize classification metrics as bar plot.
        
        Args:
            metrics_dict: Dictionary from ClassificationMetrics.calculate_all_metrics()
            save_path: Optional path to save figure
        """
        class_names = metrics_dict['class_names']
        precision = metrics_dict['precision_per_class']
        recall = metrics_dict['recall_per_class']
        f1 = metrics_dict['f1_per_class']
        
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='steelblue')
        bars2 = ax.bar(x, recall, width, label='Recall', color='orange')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='green')
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Classification Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


class TrainingPlots:
    """
    Visualization functions for training progress.
    """
    
    @staticmethod
    def plot_training_history(history: dict,
                              title: str = "Training History",
                              save_path: Optional[str] = None) -> None:
        """
        Plot training and validation loss curves.
        
        Args:
            history: Dictionary with 'train_loss', 'val_loss', 'learning_rates'
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Find best epoch
        best_epoch = np.argmin(history['val_loss']) + 1
        best_loss = min(history['val_loss'])
        ax1.axvline(best_epoch, color='g', linestyle='--', linewidth=2, alpha=0.7)
        ax1.text(best_epoch, best_loss, f'  Best: epoch {best_epoch}',
                fontsize=10, va='center')
        
        # Learning rate schedule
        ax2.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_model_comparison(comparison_df: pd.DataFrame,
                             metric: str = 'RMSE',
                             save_path: Optional[str] = None) -> None:
        """
        Plot model comparison bar chart.
        
        Args:
            comparison_df: DataFrame from compare_models()
            metric: Metric to plot
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = comparison_df['Model']
        values = comparison_df[metric]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
        bars = ax.barh(models, values, color=colors)
        
        ax.set_xlabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value, i, f' {value:.4f}', 
                   va='center', fontsize=10, fontweight='bold')
        
        # Highlight best model
        best_idx = 0  # Already sorted
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    """
    Example usage and testing of plotting functions.
    """
    print("Testing Evaluation Plots Module")
    print("=" * 70)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    # Regression plots
    print("\n1. Testing Regression Plots...")
    y_true_reg = np.random.gamma(2, 0.5, n_samples)
    y_pred_reg = y_true_reg + np.random.normal(0, 0.3, n_samples)
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')
    
    RegressionPlots.plot_predictions_vs_actual(y_true_reg, y_pred_reg, dates)
    RegressionPlots.plot_residuals(y_true_reg, y_pred_reg)
    RegressionPlots.plot_error_distribution(y_true_reg, y_pred_reg)
    
    # Classification plots
    print("\n2. Testing Classification Plots...")
    cm = np.array([[45, 3, 2], [5, 28, 2], [1, 1, 13]])
    class_names = ['Mild', 'Snowy', 'Severe']
    
    ClassificationPlots.plot_confusion_matrix(cm, class_names)
    
    # Training plots
    print("\n3. Testing Training Plots...")
    history = {
        'train_loss': np.logspace(0, -1, 50) + np.random.normal(0, 0.01, 50),
        'val_loss': np.logspace(0, -0.5, 50) + np.random.normal(0, 0.02, 50),
        'learning_rates': [0.001 * (0.5 ** (i // 10)) for i in range(50)]
    }
    
    TrainingPlots.plot_training_history(history)
    
    print("\nAll tests completed successfully!")