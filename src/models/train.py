"""
Training Module for LSTM Snowfall Prediction
Includes training loop, validation, early stopping, and learning rate scheduling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import time
from typing import Dict, Tuple, Optional


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = True):
        """
        Args:
            patience: How many epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Current model state
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            if self.verbose:
                print(f"  Initial best loss: {val_loss:.4f}")
        
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  Early stopping triggered! Best loss: {self.best_loss:.4f}")
        
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0
            if self.verbose:
                print(f"  New best loss: {val_loss:.4f}")


def create_dataloaders(X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       batch_size: int = 32,
                       shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        batch_size: Batch size for training
        shuffle: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Created DataLoaders:")
    print(f"  Training batches: {len(train_loader)} (batch_size={batch_size})")
    print(f"  Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


def train_epoch(model: nn.Module,
               train_loader: DataLoader,
               criterion: nn.Module,
               optimizer: optim.Optimizer,
               device: str) -> float:
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move to device
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model: nn.Module,
            val_loader: DataLoader,
            criterion: nn.Module,
            device: str) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (avg_loss, predictions, targets)
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            # Move to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Store predictions and targets
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    return total_loss / len(val_loader), predictions, targets


def train_model(model: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               num_epochs: int = 150,           # INCREASED from 100
               learning_rate: float = 0.0005,   # DECREASED for stability
               device: str = None,
               early_stopping_patience: int = 20,  # INCREASED from 15
               model_save_path: str = "models/lstm.pth") -> Dict:
    """
    Complete training loop with early stopping and learning rate scheduling.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of epochs
        learning_rate: Initial learning rate
        device: Device to train on
        early_stopping_patience: Patience for early stopping
        model_save_path: Path to save best model
        
    Returns:
        Dictionary with training history
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*70)
    print("STARTING LSTM TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max epochs: {num_epochs}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print("="*70 + "\n")
    
    # Loss function (MSE for regression)
    criterion = nn.MSELoss()
    
    # Optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler (reduce on plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, _, _ = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(current_lr)
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{num_epochs}] - {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            # Load best model
            model.load_state_dict(early_stopping.best_model)
            break
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Best validation loss: {early_stopping.best_loss:.4f}")
    print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    print("="*70 + "\n")
    
    # Save best model
    save_dir = Path(model_save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': model.input_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'best_val_loss': early_stopping.best_loss,
    }, model_save_path)
    
    print(f"✓ Best model saved to {model_save_path}\n")
    
    return history


def calculate_metrics(predictions: np.ndarray, 
                     targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        
    Returns:
        Dictionary of metrics
    """
    # Flatten arrays
    pred = predictions.flatten()
    true = targets.flatten()
    
    # RMSE
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    
    # MAE
    mae = np.mean(np.abs(pred - true))
    
    # R-squared
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Accuracy for binary "will it snow?" (threshold = 0.1 inches)
    threshold = 0.1
    pred_binary = (pred > threshold).astype(int)
    true_binary = (true > threshold).astype(int)
    snow_accuracy = np.mean(pred_binary == true_binary) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'snow_accuracy': snow_accuracy
    }


if __name__ == "__main__":
    print("Training module loaded successfully!")
    print("\nTo train a model:")
    print("1. Load and preprocess data")
    print("2. Create model: model = create_model()")
    print("3. Create dataloaders: train_loader, val_loader = create_dataloaders(...)")
    print("4. Train: history = train_model(model, train_loader, val_loader)")