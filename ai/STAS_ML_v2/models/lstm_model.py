"""
LSTM model implementation for STAS_ML v2
"""

import numpy as np
import joblib
from typing import Dict, Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from ..core.config import Config
from ..core.base import BaseModel, Logger, MetricsCalculator


class LSTMNetwork(nn.Module):
    """LSTM neural network."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2):
        super(LSTMNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take last output
        lstm_out = self.dropout(lstm_out)
        predictions = self.linear(lstm_out)
        return predictions


class LSTMModel(BaseModel):
    """LSTM model implementation."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.logger = Logger("LSTMModel")
        
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train LSTM model."""
        
        # Reshape data for LSTM (samples, sequence_length, features)
        sequence_length = self.config.features.lookback_window
        features_per_timestep = X.shape[1] // sequence_length
        
        X_reshaped = X.reshape(X.shape[0], sequence_length, features_per_timestep)
        
        # Create model
        self.model = LSTMNetwork(
            input_size=features_per_timestep,
            hidden_size=self.config.model.lstm_params['hidden_size'],
            num_layers=self.config.model.lstm_params['num_layers'],
            output_size=1,
            dropout=self.config.model.lstm_params['dropout']
        ).to(self.device)
        
        # Loss and optimizer
        if self.is_classification():
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(self.model.parameters(), 
                             lr=self.config.model.lstm_params['learning_rate'])
        
        # Create data loader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_reshaped), 
            torch.FloatTensor(y.reshape(-1, 1))
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.model.lstm_params['batch_size'], 
            shuffle=True
        )
        
        # Training loop
        self.model.train()
        train_losses = []
        
        for epoch in range(self.config.model.lstm_params['epochs']):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                self.logger.info(f"Epoch {epoch+1}/{self.config.model.lstm_params['epochs']}, Loss: {avg_loss:.6f}")
        
        self.is_trained = True
        
        # Calculate metrics
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_reshaped).to(self.device)
            train_pred = self.model(X_tensor).cpu().numpy().flatten()
            
            if self.is_classification():
                train_pred = (torch.sigmoid(torch.FloatTensor(train_pred)) > 0.5).numpy().astype(int)
        
        metrics = {
            'train_samples': len(X),
            'train_losses': train_losses
        }
        
        if X_val is not None and y_val is not None:
            X_val_reshaped = X_val.reshape(X_val.shape[0], sequence_length, features_per_timestep)
            X_val_tensor = torch.FloatTensor(X_val_reshaped).to(self.device)
            
            with torch.no_grad():
                val_pred = self.model(X_val_tensor).cpu().numpy().flatten()
                
                if self.is_classification():
                    val_pred = (torch.sigmoid(torch.FloatTensor(val_pred)) > 0.5).numpy().astype(int)
            
            if self.is_classification():
                metrics.update({
                    'train_accuracy': MetricsCalculator.classification_metrics(y, train_pred)['accuracy'],
                    'val_accuracy': MetricsCalculator.classification_metrics(y_val, val_pred)['accuracy']
                })
            else:
                train_metrics = MetricsCalculator.regression_metrics(y, train_pred)
                val_metrics = MetricsCalculator.regression_metrics(y_val, val_pred)
                metrics.update({
                    'train_rmse': train_metrics['rmse'],
                    'val_rmse': val_metrics['rmse'],
                    'train_r2': train_metrics['r2'],
                    'val_r2': val_metrics['r2']
                })
        
        self.training_metrics = metrics
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Reshape for LSTM
        sequence_length = self.config.features.lookback_window
        features_per_timestep = X.shape[1] // sequence_length
        X_reshaped = X.reshape(X.shape[0], sequence_length, features_per_timestep)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_reshaped).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().flatten()
            
            if self.is_classification():
                predictions = (torch.sigmoid(torch.FloatTensor(predictions)) > 0.5).numpy().astype(int)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for classification."""
        if not self.is_classification():
            raise ValueError("Probabilities only available for classification models")
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Reshape for LSTM
        sequence_length = self.config.features.lookback_window
        features_per_timestep = X.shape[1] // sequence_length
        X_reshaped = X.reshape(X.shape[0], sequence_length, features_per_timestep)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_reshaped).to(self.device)
            logits = self.model(X_tensor).cpu().numpy().flatten()
            probabilities = torch.sigmoid(torch.FloatTensor(logits)).numpy()
        
        # Return probabilities for both classes
        return np.column_stack([1 - probabilities, probabilities])
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """LSTM doesn't provide feature importance."""
        return None
    
    def save(self, filepath: str):
        """Save model."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics
        }, filepath)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Recreate model architecture
        sequence_length = self.config.features.lookback_window
        features_per_timestep = 10  # This should be stored in checkpoint
        
        self.model = LSTMNetwork(
            input_size=features_per_timestep,
            hidden_size=self.config.model.lstm_params['hidden_size'],
            num_layers=self.config.model.lstm_params['num_layers'],
            output_size=1,
            dropout=self.config.model.lstm_params['dropout']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint.get('is_trained', True)
        self.training_metrics = checkpoint.get('training_metrics', {})
        
        self.logger.info(f"Model loaded from {filepath}")