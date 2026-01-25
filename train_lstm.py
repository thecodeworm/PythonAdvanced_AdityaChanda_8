"""
Test script for LSTM model training with Pittsburgh data
Runs the complete pipeline: data loading → preprocessing → training → evaluation
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.load_data import load_data
from data.preprocess import (
    preprocess_data,
    train_test_split_temporal,
    prepare_data_for_lstm
)
from models.lstm_model import create_model, get_model_summary
from models.train import (
    create_dataloaders,
    train_model,
    validate,
    calculate_metrics
)
import torch
import torch.nn as nn


def plot_training_history(history, save_path="reports/figures/training_history.png"):
    """Plot training and validation loss over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Learning rate plot
    ax2.plot(history['learning_rates'], color='green', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_predictions(predictions, targets, dates, save_path="reports/figures/predictions.png"):
    """Plot predictions vs actual snowfall."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Time series plot
    ax1.plot(dates, targets, label='Actual Snowfall', marker='o', linewidth=2, markersize=4)
    ax1.plot(dates, predictions, label='Predicted Snowfall', marker='x', linewidth=2, markersize=4, alpha=0.7)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Snowfall (inches)', fontsize=12)
    ax1.set_title('Snowfall Predictions vs Actual - Pittsburgh', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Scatter plot
    ax2.scatter(targets, predictions, alpha=0.6, s=50)
    
    # Perfect prediction line
    max_val = max(targets.max(), predictions.max())
    ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Snowfall (inches)', fontsize=12)
    ax2.set_ylabel('Predicted Snowfall (inches)', fontsize=12)
    ax2.set_title('Prediction Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Predictions plot saved to {save_path}")
    plt.close()


def main():
    print("="*70)
    print("LSTM MODEL TRAINING TEST - PITTSBURGH DATA")
    print("="*70)
    
    try:
        # ===== STEP 1: Load and Preprocess Data =====
        print("\nSTEP 1: Loading and preprocessing Pittsburgh data...")
        df = load_data(
            data_file="data/raw/pittsburgh_winters_10years.csv",
            validate=False,
            filter_station=True,
            winter_only=True
        )
        
        print(f"\nLoaded {len(df)} winter days")
        print(f"Date range: {df['DATE'].min()} to {df['DATE'].max()}")
        
        df_processed = preprocess_data(df, add_features=True)
        
        # Split data
        train_df, test_df = train_test_split_temporal(df_processed, test_size=0.2)
        
        # Prepare for LSTM
        lstm_data = prepare_data_for_lstm(
            train_df,
            test_df,
            sequence_length=7,
            scaler_path="models/scaler.pkl"
        )
        
        print(f"\nData prepared:")
        print(f"  Training samples: {lstm_data['X_train'].shape[0]}")
        print(f"  Test samples: {lstm_data['X_test'].shape[0]}")
        print(f"  Features: {lstm_data['X_train'].shape[2]}")
        print(f"  Snow days in training: {(lstm_data['y_train'] > 0).sum()} ({(lstm_data['y_train'] > 0).sum()/len(lstm_data['y_train'])*100:.1f}%)")
        
        # ===== STEP 2: Create Model =====
        print("\nSTEP 2: Creating LSTM model...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = create_model(
            input_size=lstm_data['X_train'].shape[2],  # Number of features
            hidden_size=128,
            num_layers=3,
            dropout=0.3,
            device=device
        )
        
        get_model_summary(model, input_size=(7, lstm_data['X_train'].shape[2]))
        
        # ===== STEP 3: Create DataLoaders =====
        print("\nSTEP 3: Creating DataLoaders...")
        
        # Split training data into train/validation (80/20)
        val_split = int(0.8 * len(lstm_data['X_train']))
        
        X_train_split = lstm_data['X_train'][:val_split]
        y_train_split = lstm_data['y_train'][:val_split]
        X_val_split = lstm_data['X_train'][val_split:]
        y_val_split = lstm_data['y_train'][val_split:]
        
        train_loader, val_loader = create_dataloaders(
            X_train_split,
            y_train_split,
            X_val_split,
            y_val_split,
            batch_size=32,
            shuffle=True
        )
        
        # ===== STEP 4: Train Model =====
        print("\nSTEP 4: Training model (may take several minutes)...\n")
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=150,
            learning_rate=0.0005,
            device=device,
            early_stopping_patience=20,
            model_save_path="models/lstm.pth"
        )
        
        # Plot training history
        plot_training_history(history)
        
        # ===== STEP 5: Evaluate on Test Set =====
        print("\nSTEP 5: Evaluating on test set...")
        
        # Create test dataloader
        X_test_tensor = torch.FloatTensor(lstm_data['X_test']).to(device)
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test_tensor).cpu().numpy()
        
        # Calculate metrics
        test_targets = lstm_data['y_test'].reshape(-1, 1)
        metrics = calculate_metrics(test_predictions, test_targets)
        
        print("\n" + "="*70)
        print("TEST SET RESULTS - PITTSBURGH DATA")
        print("="*70)
        print(f"  RMSE: {metrics['rmse']:.3f} inches")
        print(f"  MAE: {metrics['mae']:.3f} inches")
        print(f"  R²: {metrics['r2']:.3f}")
        print(f"  Snow Detection Accuracy: {metrics['snow_accuracy']:.1f}%")
        print("="*70 + "\n")
        
        # Interpretation
        print("Performance Interpretation:")
        if metrics['r2'] > 0.6:
            print("  Excellent: Model explains over 60% of variance")
        elif metrics['r2'] > 0.4:
            print("  Good: Model captures major patterns")
        elif metrics['r2'] > 0.2:
            print("  Fair: Captures some patterns but needs improvement")
        else:
            print("  Poor: Model struggling to learn patterns")
        
        if metrics['rmse'] < 0.5:
            print("  RMSE < 0.5 inches: very accurate predictions")
        elif metrics['rmse'] < 0.7:
            print("  RMSE acceptable for snowfall prediction")
        else:
            print("  RMSE high: predictions have significant error")
        
        if metrics['snow_accuracy'] > 80:
            print("  Snow detection > 80%: excellent")
        elif metrics['snow_accuracy'] > 70:
            print("  Snow detection good")
        else:
            print("  Snow detection needs improvement")
        
        # Plot predictions
        plot_predictions(
            predictions=test_predictions.flatten(),
            targets=test_targets.flatten(),
            dates=lstm_data['test_dates']
        )
        
        # Show some example predictions
        print("\nSample Predictions:")
        print(f"{'Date':<12} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
        print("-" * 42)
        
        for i in range(min(15, len(test_predictions))):
            date_str = str(lstm_data['test_dates'][i])[:10]
            actual = test_targets[i, 0]
            pred = test_predictions[i, 0]
            error = abs(actual - pred)
            print(f"{date_str:<12} {actual:>8.2f} {pred:>10.2f} {error:>8.2f}")
        
        print("\n" + "="*70)
        print("Training and Evaluation Complete")
        print("="*70)
        print("\nFiles saved:")
        print("  - models/lstm.pth (trained model)")
        print("  - models/scaler.pkl (feature scaler)")
        print("  - reports/figures/training_history.png")
        print("  - reports/figures/predictions.png")
        
        print("\nNext steps:")
        print("  1. Build severity classifier")
        print("  2. Create clothing recommendation engine")
        print("  3. Build Tkinter GUI")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Make sure you have:")
        print("  data/raw/pittsburgh_winters_10years.csv")
        
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
