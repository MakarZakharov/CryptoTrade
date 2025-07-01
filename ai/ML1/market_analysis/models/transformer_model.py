"""
Transformer model implementation for market analysis.
"""

import numpy as np
import os
import sys
from typing import Any, Dict, Optional, Tuple, Union, List
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, LayerNormalization, MultiHeadAttention, Dropout,
    Input, GlobalAveragePooling1D, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

# Handle both package import and direct script execution
try:
    from .base_model import BaseModel
    from ..config import (
        DEFAULT_DROPOUT_RATE, DEFAULT_LEARNING_RATE,
        DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_EARLY_STOPPING_PATIENCE,
        DEFAULT_REDUCE_LR_PATIENCE, DEFAULT_REDUCE_LR_FACTOR, DEFAULT_MIN_LR
    )
except ImportError:
    # Add current directory to path for direct script execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(current_dir)
    sys.path.append(parent_dir)
    
    from base_model import BaseModel
    from config import (
        DEFAULT_DROPOUT_RATE, DEFAULT_LEARNING_RATE,
        DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_EARLY_STOPPING_PATIENCE,
        DEFAULT_REDUCE_LR_PATIENCE, DEFAULT_REDUCE_LR_FACTOR, DEFAULT_MIN_LR
    )


class TransformerBlock(tf.keras.layers.Layer):
    """
    Transformer block with multi-head self-attention and feed-forward network.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1):
        """
        Initialize the transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward network dimension
            rate: Dropout rate
        """
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        
    def call(self, inputs, training=False):
        """
        Forward pass through the transformer block.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerModel(BaseModel):
    """
    Transformer model for time series prediction.
    
    Transformers use self-attention mechanisms instead of recurrence, which allows them
    to capture long-range dependencies more effectively. This implementation uses a stack
    of transformer blocks followed by a global average pooling and dense layers.
    """
    
    def __init__(self, input_shape: Tuple[int, int], name: str = None, 
                 embed_dim: int = 32, num_heads: int = 2, ff_dim: int = 32, 
                 num_transformer_blocks: int = 2, mlp_units: List[int] = [64], 
                 dropout: float = DEFAULT_DROPOUT_RATE, 
                 learning_rate: float = DEFAULT_LEARNING_RATE):
        """
        Initialize the Transformer model.
        
        Args:
            input_shape: Shape of the input data (sequence_length, features)
            name: Optional name for the model
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward network dimension
            num_transformer_blocks: Number of transformer blocks
            mlp_units: Units in the final MLP layers
            dropout: Dropout rate
            learning_rate: Learning rate for the optimizer
        """
        super().__init__(name=name or "Transformer")
        self.input_shape = input_shape
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.build()
    
    def build(self, **kwargs) -> None:
        """
        Build the Transformer model architecture.
        
        Args:
            **kwargs: Additional model parameters
        """
        # Override parameters if provided
        embed_dim = kwargs.get('embed_dim', self.embed_dim)
        num_heads = kwargs.get('num_heads', self.num_heads)
        ff_dim = kwargs.get('ff_dim', self.ff_dim)
        num_transformer_blocks = kwargs.get('num_transformer_blocks', self.num_transformer_blocks)
        mlp_units = kwargs.get('mlp_units', self.mlp_units)
        dropout = kwargs.get('dropout', self.dropout)
        learning_rate = kwargs.get('learning_rate', self.learning_rate)
        
        # Build the model
        inputs = Input(shape=self.input_shape)
        
        # Initial projection to embed_dim if needed
        if self.input_shape[1] != embed_dim:
            x = Dense(embed_dim)(inputs)
        else:
            x = inputs
        
        # Transformer blocks
        for _ in range(num_transformer_blocks):
            x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)(x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # MLP head
        for dim in mlp_units:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(dropout)(x)
        
        # Output layer
        outputs = Dense(1)(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse'
        )
        
        self.model = model
        self.metadata.update({
            'input_shape': self.input_shape,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'ff_dim': ff_dim,
            'num_transformer_blocks': num_transformer_blocks,
            'mlp_units': mlp_units,
            'dropout': dropout,
            'learning_rate': learning_rate
        })
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """
        Train the Transformer model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Training parameters
                - batch_size: Batch size for training
                - epochs: Number of training epochs
                - verbose: Verbosity level
                - callbacks: Additional callbacks
                
        Returns:
            Dictionary containing training history
        """
        if self.model is None:
            self.build()
        
        # Get training parameters
        batch_size = kwargs.get('batch_size', DEFAULT_BATCH_SIZE)
        epochs = kwargs.get('epochs', DEFAULT_EPOCHS)
        verbose = kwargs.get('verbose', 1)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Prepare callbacks
        callbacks = kwargs.get('callbacks', [])
        
        # Add default callbacks if not provided
        if not callbacks:
            # Early stopping callback
            early_stopping = EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=DEFAULT_EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=verbose
            )
            
            # Learning rate reduction callback
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=DEFAULT_REDUCE_LR_FACTOR,
                patience=DEFAULT_REDUCE_LR_PATIENCE,
                min_lr=DEFAULT_MIN_LR,
                verbose=verbose
            )
            
            # Combine callbacks
            callbacks = [early_stopping, reduce_lr]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        self.history = history.history
        
        # Update metadata
        self.metadata.update({
            'training_parameters': {
                'batch_size': batch_size,
                'epochs': epochs,
                'final_epoch': len(history.history['loss']),
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1] if validation_data else None
            }
        })
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained Transformer model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """
        Save the Transformer model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been built yet")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        self.model.save(path)
        
        # Save metadata
        self.save_metadata(path)
    
    def load(self, path: str) -> None:
        """
        Load the Transformer model from disk.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load the model with custom objects
        custom_objects = {
            'TransformerBlock': TransformerBlock
        }
        self.model = load_model(path, custom_objects=custom_objects)
        self.is_trained = True
        
        # Load metadata
        self.load_metadata(path)
        
        # Update instance variables from metadata
        if 'input_shape' in self.metadata:
            self.input_shape = tuple(self.metadata['input_shape'])
        if 'embed_dim' in self.metadata:
            self.embed_dim = self.metadata['embed_dim']
        if 'num_heads' in self.metadata:
            self.num_heads = self.metadata['num_heads']
        if 'ff_dim' in self.metadata:
            self.ff_dim = self.metadata['ff_dim']
        if 'num_transformer_blocks' in self.metadata:
            self.num_transformer_blocks = self.metadata['num_transformer_blocks']
        if 'mlp_units' in self.metadata:
            self.mlp_units = self.metadata['mlp_units']
        if 'dropout' in self.metadata:
            self.dropout = self.metadata['dropout']
        if 'learning_rate' in self.metadata:
            self.learning_rate = self.metadata['learning_rate']
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the Transformer model parameters.
        
        Returns:
            Dictionary containing model parameters
        """
        return {
            'input_shape': self.input_shape,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'num_transformer_blocks': self.num_transformer_blocks,
            'mlp_units': self.mlp_units,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate
        }
    
    def set_params(self, **params) -> 'TransformerModel':
        """
        Set the Transformer model parameters.
        
        Args:
            **params: Model parameters
            
        Returns:
            Self for method chaining
        """
        if 'input_shape' in params:
            self.input_shape = params['input_shape']
        if 'embed_dim' in params:
            self.embed_dim = params['embed_dim']
        if 'num_heads' in params:
            self.num_heads = params['num_heads']
        if 'ff_dim' in params:
            self.ff_dim = params['ff_dim']
        if 'num_transformer_blocks' in params:
            self.num_transformer_blocks = params['num_transformer_blocks']
        if 'mlp_units' in params:
            self.mlp_units = params['mlp_units']
        if 'dropout' in params:
            self.dropout = params['dropout']
        if 'learning_rate' in params:
            self.learning_rate = params['learning_rate']
        
        # Rebuild the model with new parameters
        self.build()
        
        return self