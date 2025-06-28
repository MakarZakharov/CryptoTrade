"""
Attention Mechanisms

Implementation of various attention mechanisms for trading models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    
    Implements scaled dot-product attention with multiple heads
    for capturing different aspects of market relationships.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        """
        Initialize multi-head attention
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            temperature: Temperature for attention softmax
        """
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.temperature = temperature
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output tensor and attention weights
        """
        batch_size, seq_len, _ = query.size()
        residual = query
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.w_o(output)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot-product attention"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (np.sqrt(self.d_k) * self.temperature)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class TemporalAttention(nn.Module):
    """
    Temporal attention for time series data
    
    Focuses on important time steps in the trading sequence.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize temporal attention
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # LSTM for temporal encoding
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Attended output and attention weights
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim * 2]
        
        # Attention weights
        attention_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        
        # Weighted sum
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)  # [batch_size, hidden_dim * 2]
        attended_output = self.dropout(attended_output)
        
        return attended_output, attention_weights.squeeze(-1)


class CrossAssetAttention(nn.Module):
    """
    Cross-asset attention mechanism
    
    Captures relationships between different trading assets.
    """
    
    def __init__(
        self,
        asset_dim: int,
        num_assets: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize cross-asset attention
        
        Args:
            asset_dim: Feature dimension per asset
            num_assets: Number of assets
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.asset_dim = asset_dim
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        
        # Asset embedding
        self.asset_projection = nn.Linear(asset_dim, hidden_dim)
        
        # Cross-asset attention
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, asset_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            asset_features: [batch_size, num_assets, asset_dim]
            
        Returns:
            Attended features and attention weights
        """
        batch_size, num_assets, _ = asset_features.size()
        
        # Project asset features
        projected = self.asset_projection(asset_features)  # [batch_size, num_assets, hidden_dim]
        residual = projected
        
        # Attention projections
        Q = self.query_projection(projected)
        K = self.key_projection(projected)
        V = self.value_projection(projected)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.hidden_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Attended values
        attended = torch.matmul(attention_weights, V)
        
        # Output projection
        output = self.output_projection(attended)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output, attention_weights


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for feature relationships
    
    Captures relationships within feature vectors.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize self-attention
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        self.output = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor and attention weights
        """
        residual = x
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.hidden_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Attended values
        attended = torch.matmul(attention_weights, V)
        output = self.output(attended)
        output = self.dropout(output)
        
        # Residual connection
        output = self.layer_norm(output + residual)
        
        return output, attention_weights


class AdaptiveAttention(nn.Module):
    """
    Adaptive attention mechanism
    
    Dynamically adjusts attention based on market conditions.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_conditions: int = 4,  # bull, bear, sideways, volatile
        dropout: float = 0.1
    ):
        """
        Initialize adaptive attention
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_conditions: Number of market conditions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_conditions = num_conditions
        
        # Market condition classifier
        self.condition_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_conditions),
            nn.Softmax(dim=-1)
        )
        
        # Condition-specific attention weights
        self.attention_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ) for _ in range(num_conditions)
        ])
        
        self.output_projection = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor, attention weights, and condition probabilities
        """
        batch_size, seq_len, _ = x.size()
        
        # Market condition classification (use last time step)
        condition_probs = self.condition_classifier(x[:, -1, :])  # [batch_size, num_conditions]
        
        # Compute condition-specific attention weights
        attention_outputs = []
        for i, attention_layer in enumerate(self.attention_weights):
            weights = attention_layer(x)  # [batch_size, seq_len, 1]
            attention_outputs.append(weights)
        
        # Stack and weight by condition probabilities
        all_attention_weights = torch.stack(attention_outputs, dim=-1)  # [batch_size, seq_len, 1, num_conditions]
        all_attention_weights = all_attention_weights.squeeze(2)  # [batch_size, seq_len, num_conditions]
        
        # Weighted combination of attention weights
        condition_probs_expanded = condition_probs.unsqueeze(1).expand(-1, seq_len, -1)
        final_attention_weights = torch.sum(all_attention_weights * condition_probs_expanded, dim=-1, keepdim=True)
        
        # Apply attention
        attended_output = x * final_attention_weights
        output = self.output_projection(attended_output)
        output = self.dropout(output)
        
        return output, final_attention_weights.squeeze(-1), condition_probs