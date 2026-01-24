"""
LSTM Model Architecture for Snowfall Prediction
Uses PyTorch to build a multi-layer LSTM for time-series forecasting.
"""

import torch
import torch.nn as nn
import numpy as np


class SnowfallLSTM(nn.Module):
    def __init__(self, input_size: int = 24, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.3, output_size: int = 1):
        super(SnowfallLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions.cpu().numpy()


def get_model_summary(model, input_size=(7, 24)):
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    print(f"\n{model}\n")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: Total={total_params:,}, Trainable={trainable_params:,}")

    dummy_input = torch.randn(32, *input_size)
    try:
        output = model(dummy_input)
        print("Forward pass successful!")
        print(f"Input shape: {dummy_input.shape}, Output shape: {output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
    print("="*60 + "\n")


def create_model(input_size: int = 24, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.3, device: str = None) -> SnowfallLSTM:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SnowfallLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    return model.to(device)


def save_model(model, filepath: str):
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': model.input_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
    }, filepath)


def load_model(filepath: str, device: str = None) -> SnowfallLSTM:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(filepath, map_location=device)
    model = SnowfallLSTM(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device)


if __name__ == "__main__":
    print("Testing LSTM model creation...\n")
    model = create_model(input_size=24, hidden_size=64, num_layers=2, dropout=0.2)
    get_model_summary(model, input_size=(7, 24))
    dummy_input = torch.randn(5, 7, 24)
    predictions = model.predict(dummy_input)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:3, 0]}")
