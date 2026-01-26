"""
Comprehensive Model Evaluation Script
Run this to generate all evaluation metrics and plots.
"""

import numpy as np
import torch
from pathlib import Path

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data, prepare_data_for_lstm, create_feature_columns
from src.models.lstm_model import load_model as load_lstm_model
from src.evaluation.metrics import RegressionMetrics, compare_models
from src.evaluation.plots import RegressionPlots

def main():
    print("="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    df = load_data("data/raw/pittsburgh_winters_10years.csv", 
                   validate=False, filter_station=True, winter_only=True)
    df_processed = preprocess_data(df, add_features=True)
    
    # Split into train/test (80/20 temporal split)
    split_idx = int(len(df_processed) * 0.8)
    train_df = df_processed.iloc[:split_idx].copy()
    test_df = df_processed.iloc[split_idx:].copy()
    
    print(f"\nTrain: {len(train_df)} days ({train_df['DATE'].min()} to {train_df['DATE'].max()})")
    print(f"Test: {len(test_df)} days ({test_df['DATE'].min()} to {test_df['DATE'].max()})")
    
    # Prepare sequences for LSTM
    print("\nPreparing sequences...")
    data = prepare_data_for_lstm(train_df, test_df, sequence_length=7)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"Train sequences: {X_train.shape}")
    print(f"Test sequences: {X_test.shape}")
    print(f"Test samples: {len(y_test)}")
    
    # Load trained model
    print("\nLoading trained LSTM model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = load_lstm_model("models/lstm.pth", device=device)
        model.eval()
        print(f"✓ Model loaded successfully on {device}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nPlease train the model first:")
        print("  python test_lstm_training.py")
        return
    
    # Make predictions
    print("\nGenerating predictions on test set...")
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        predictions = model(X_test_tensor).cpu().numpy()
    
    # Flatten arrays
    y_test_flat = y_test.flatten()
    y_pred_flat = predictions.flatten()
    
    print(f"✓ Generated {len(y_pred_flat)} predictions")
    
    # Ensure reports/figures directory exists
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("REGRESSION EVALUATION (LSTM)")
    print("="*70)
    
    # Calculate and display metrics
    metrics = RegressionMetrics.calculate_all_metrics(y_test_flat, y_pred_flat)
    print(RegressionMetrics.format_metrics_report(metrics))
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION PLOTS")
    print("="*70)
    print("This may take a few moments...\n")
    
    # Plot 1: Predictions vs Actual
    print("1. Creating predictions vs actual plot...")
    try:
        RegressionPlots.plot_predictions_vs_actual(
            y_test_flat, y_pred_flat,
            title="LSTM Snowfall Predictions - Test Set",
            save_path="reports/figures/lstm_predictions_comprehensive.png"
        )
        print("   ✓ Saved to reports/figures/lstm_predictions_comprehensive.png")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Plot 2: Residual Analysis
    print("\n2. Creating residual analysis plot...")
    try:
        RegressionPlots.plot_residuals(
            y_test_flat, y_pred_flat,
            title="LSTM Residual Analysis",
            save_path="reports/figures/lstm_residuals_comprehensive.png"
        )
        print("   ✓ Saved to reports/figures/lstm_residuals_comprehensive.png")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Plot 3: Error Distribution
    print("\n3. Creating error distribution plot...")
    try:
        RegressionPlots.plot_error_distribution(
            y_test_flat, y_pred_flat,
            save_path="reports/figures/lstm_errors_comprehensive.png"
        )
        print("   ✓ Saved to reports/figures/lstm_errors_comprehensive.png")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Model comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    # Create baseline models for comparison
    print("\nCalculating baseline models...")
    
    # Always predict mean
    baseline_mean = np.full_like(y_test_flat, y_test_flat.mean())
    
    # Always predict no snow
    baseline_zero = np.zeros_like(y_test_flat)
    
    # Persistence baseline (yesterday's value)
    baseline_persistence = np.roll(y_test_flat, 1)
    baseline_persistence[0] = y_test_flat[0]
    
    models = {
        'LSTM': y_pred_flat,
        'Mean Baseline': baseline_mean,
        'Zero Baseline': baseline_zero,
        'Persistence (Yesterday)': baseline_persistence
    }
    
    comparison = compare_models(models, y_test_flat)
    print("\n" + comparison.to_string(index=False))
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    print("\nGenerated Files:")
    print("  ✓ reports/figures/lstm_predictions_comprehensive.png")
    print("  ✓ reports/figures/lstm_residuals_comprehensive.png")
    print("  ✓ reports/figures/lstm_errors_comprehensive.png")
    
    print("\nLSTM Performance:")
    print(f"  RMSE: {metrics['rmse']:.4f} inches")
    print(f"  MAE: {metrics['mae']:.4f} inches")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Snow Detection Accuracy: {metrics['snow_accuracy']*100:.1f}%")
    print(f"  Precision: {metrics['snow_precision']:.4f}")
    print(f"  Recall: {metrics['snow_recall']:.4f}")
    
    # Compare to best baseline
    if len(comparison) > 1:
        best_baseline = comparison.iloc[1]  # Second row (first is LSTM if sorted)
        improvement = ((best_baseline['RMSE'] - metrics['rmse']) / best_baseline['RMSE'] * 100)
        
        print(f"\nComparison to Best Baseline ({best_baseline['Model']}):")
        print(f"  Baseline RMSE: {best_baseline['RMSE']:.4f} inches")
        if improvement > 0:
            print(f"  LSTM Improvement: {improvement:.1f}% better")
        else:
            print(f"  LSTM Performance: {abs(improvement):.1f}% worse")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    
    print("\nNext Steps:")
    print("  1. Review the generated plots in reports/figures/")
    print("  2. Analyze the metrics above")
    print("  3. Consider model improvements if needed")
    print("  4. Include plots in your project documentation")

if __name__ == "__main__":
    main()