"""
Model factory for creating and configuring different types of models.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

# Add all possible paths to sys.path
sys.path.append(current_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

# Define dummy classes to use as fallbacks if imports fail
class DummyModel:
    """Dummy model class used as fallback when imports fail."""
    def __init__(self, *args, **kwargs):
        self.name = "DummyModel"
        print(f"WARNING: Using dummy {self.name} because the real implementation couldn't be imported")
        print(f"Make sure all dependencies are installed (e.g., 'pip install xgboost')")
        print(f"And make sure the market_analysis package is in your Python path")

# Try different import strategies
try:
    # Try relative imports first (when used as a package)
    from .base_model import BaseModel
    from .lstm_model import LSTMModel
    from .gru_model import GRUModel
    from .transformer_model import TransformerModel
    from .xgboost_model import XGBoostModel
    from .ensemble import StackingEnsembleModel, VotingEnsembleModel
    print("Using relative imports")
except ImportError:
    try:
        # Try direct imports (when in the same directory)
        from base_model import BaseModel
        from lstm_model import LSTMModel
        from gru_model import GRUModel
        from transformer_model import TransformerModel
        from xgboost_model import XGBoostModel
        from ensemble import StackingEnsembleModel, VotingEnsembleModel
        print("Using direct imports")
    except ImportError:
        try:
            # Try absolute imports with market_analysis prefix
            from market_analysis.models.base_model import BaseModel
            from market_analysis.models.lstm_model import LSTMModel
            from market_analysis.models.gru_model import GRUModel
            from market_analysis.models.transformer_model import TransformerModel
            from market_analysis.models.xgboost_model import XGBoostModel
            from market_analysis.models.ensemble import StackingEnsembleModel, VotingEnsembleModel
            print("Using absolute imports with market_analysis prefix")
        except ImportError:
            # Use dummy classes as fallbacks
            print("WARNING: Failed to import model classes. Using dummy implementations.")
            print("Make sure all dependencies are installed and the market_analysis package is in your Python path.")
            
            # Define dummy classes for each model type
            BaseModel = DummyModel
            LSTMModel = DummyModel
            GRUModel = DummyModel
            TransformerModel = DummyModel
            
            # For XGBoost, create a specific dummy that warns about the missing xgboost package
            class XGBoostModel(DummyModel):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.name = "XGBoostModel"
                    print("To use XGBoostModel, install xgboost: pip install xgboost")
            
            # Ensemble models
            StackingEnsembleModel = DummyModel
            VotingEnsembleModel = DummyModel


class ModelFactory:
    """
    Factory class for creating and configuring different types of models.
    """
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """
        Create a model of the specified type with the given parameters.
        
        Args:
            model_type: Type of model to create ('lstm', 'gru', 'transformer', 'xgboost', 
                        'stacking_ensemble', 'voting_ensemble')
            **kwargs: Model-specific parameters
            
        Returns:
            Configured model instance
        
        Raises:
            ValueError: If the model type is not supported
        """
        model_type = model_type.lower()
        
        if model_type == 'lstm':
            return ModelFactory.create_lstm_model(**kwargs)
        elif model_type == 'gru':
            return ModelFactory.create_gru_model(**kwargs)
        elif model_type == 'transformer':
            return ModelFactory.create_transformer_model(**kwargs)
        elif model_type == 'xgboost':
            return ModelFactory.create_xgboost_model(**kwargs)
        elif model_type == 'stacking_ensemble':
            return ModelFactory.create_stacking_ensemble_model(**kwargs)
        elif model_type == 'voting_ensemble':
            return ModelFactory.create_voting_ensemble_model(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def create_lstm_model(input_shape: Tuple[int, int], name: Optional[str] = None, 
                          units: int = None, dropout: float = None, 
                          learning_rate: float = None) -> LSTMModel:
        """
        Create an LSTM model with the given parameters.
        
        Args:
            input_shape: Shape of the input data (sequence_length, features)
            name: Optional name for the model
            units: Number of LSTM units in each layer
            dropout: Dropout rate
            learning_rate: Learning rate for the optimizer
            
        Returns:
            Configured LSTM model
        """
        kwargs = {}
        if name is not None:
            kwargs['name'] = name
        if units is not None:
            kwargs['units'] = units
        if dropout is not None:
            kwargs['dropout'] = dropout
        if learning_rate is not None:
            kwargs['learning_rate'] = learning_rate
        
        return LSTMModel(input_shape=input_shape, **kwargs)
    
    @staticmethod
    def create_gru_model(input_shape: Tuple[int, int], name: Optional[str] = None, 
                         units: int = None, dropout: float = None, 
                         learning_rate: float = None) -> GRUModel:
        """
        Create a GRU model with the given parameters.
        
        Args:
            input_shape: Shape of the input data (sequence_length, features)
            name: Optional name for the model
            units: Number of GRU units in each layer
            dropout: Dropout rate
            learning_rate: Learning rate for the optimizer
            
        Returns:
            Configured GRU model
        """
        kwargs = {}
        if name is not None:
            kwargs['name'] = name
        if units is not None:
            kwargs['units'] = units
        if dropout is not None:
            kwargs['dropout'] = dropout
        if learning_rate is not None:
            kwargs['learning_rate'] = learning_rate
        
        return GRUModel(input_shape=input_shape, **kwargs)
    
    @staticmethod
    def create_transformer_model(input_shape: Tuple[int, int], name: Optional[str] = None, 
                                embed_dim: int = None, num_heads: int = None, 
                                ff_dim: int = None, num_transformer_blocks: int = None, 
                                mlp_units: List[int] = None, dropout: float = None, 
                                learning_rate: float = None) -> TransformerModel:
        """
        Create a Transformer model with the given parameters.
        
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
            
        Returns:
            Configured Transformer model
        """
        kwargs = {}
        if name is not None:
            kwargs['name'] = name
        if embed_dim is not None:
            kwargs['embed_dim'] = embed_dim
        if num_heads is not None:
            kwargs['num_heads'] = num_heads
        if ff_dim is not None:
            kwargs['ff_dim'] = ff_dim
        if num_transformer_blocks is not None:
            kwargs['num_transformer_blocks'] = num_transformer_blocks
        if mlp_units is not None:
            kwargs['mlp_units'] = mlp_units
        if dropout is not None:
            kwargs['dropout'] = dropout
        if learning_rate is not None:
            kwargs['learning_rate'] = learning_rate
        
        return TransformerModel(input_shape=input_shape, **kwargs)
    
    @staticmethod
    def create_xgboost_model(name: Optional[str] = None, learning_rate: float = None, 
                            max_depth: int = None, n_estimators: int = None, 
                            subsample: float = None, colsample_bytree: float = None, 
                            objective: str = None, early_stopping_rounds: int = None) -> XGBoostModel:
        """
        Create an XGBoost model with the given parameters.
        
        Args:
            name: Optional name for the model
            learning_rate: Learning rate (eta)
            max_depth: Maximum depth of a tree
            n_estimators: Number of boosting rounds
            subsample: Subsample ratio of the training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            objective: Learning objective
            early_stopping_rounds: Early stopping rounds
            
        Returns:
            Configured XGBoost model
        """
        kwargs = {}
        if name is not None:
            kwargs['name'] = name
        if learning_rate is not None:
            kwargs['learning_rate'] = learning_rate
        if max_depth is not None:
            kwargs['max_depth'] = max_depth
        if n_estimators is not None:
            kwargs['n_estimators'] = n_estimators
        if subsample is not None:
            kwargs['subsample'] = subsample
        if colsample_bytree is not None:
            kwargs['colsample_bytree'] = colsample_bytree
        if objective is not None:
            kwargs['objective'] = objective
        if early_stopping_rounds is not None:
            kwargs['early_stopping_rounds'] = early_stopping_rounds
        
        return XGBoostModel(**kwargs)
    
    @staticmethod
    def create_stacking_ensemble_model(base_models: List[BaseModel], meta_model: Optional[Any] = None, 
                                      name: Optional[str] = None, 
                                      use_features_in_meta: bool = False) -> StackingEnsembleModel:
        """
        Create a stacking ensemble model with the given parameters.
        
        Args:
            base_models: List of base models
            meta_model: Meta-model to combine base model predictions
            name: Optional name for the model
            use_features_in_meta: Whether to include original features in meta-model input
            
        Returns:
            Configured stacking ensemble model
        """
        kwargs = {}
        if meta_model is not None:
            kwargs['meta_model'] = meta_model
        if name is not None:
            kwargs['name'] = name
        
        kwargs['use_features_in_meta'] = use_features_in_meta
        
        return StackingEnsembleModel(base_models=base_models, **kwargs)
    
    @staticmethod
    def create_voting_ensemble_model(base_models: List[BaseModel], weights: Optional[List[float]] = None, 
                                    name: Optional[str] = None) -> VotingEnsembleModel:
        """
        Create a voting ensemble model with the given parameters.
        
        Args:
            base_models: List of base models
            weights: List of weights for each base model
            name: Optional name for the model
            
        Returns:
            Configured voting ensemble model
        """
        kwargs = {}
        if weights is not None:
            kwargs['weights'] = weights
        if name is not None:
            kwargs['name'] = name
        
        return VotingEnsembleModel(base_models=base_models, **kwargs)
    
    @staticmethod
    def create_ensemble_from_config(config: Dict[str, Any], input_shape: Tuple[int, int] = None) -> BaseModel:
        """
        Create an ensemble model from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with the following structure:
                {
                    'ensemble_type': 'stacking' or 'voting',
                    'name': Optional name for the ensemble,
                    'base_models': [
                        {
                            'type': Model type (e.g., 'lstm', 'gru'),
                            'name': Optional name for the model,
                            ... model-specific parameters ...
                        },
                        ...
                    ],
                    'meta_model': {  # Only for stacking ensemble
                        'type': Meta-model type (e.g., 'linear', 'random_forest'),
                        ... meta-model-specific parameters ...
                    },
                    'weights': [w1, w2, ...],  # Only for voting ensemble
                    'use_features_in_meta': True/False  # Only for stacking ensemble
                }
            input_shape: Shape of the input data (sequence_length, features)
                         Required for neural network models
            
        Returns:
            Configured ensemble model
        
        Raises:
            ValueError: If the configuration is invalid
        """
        if 'ensemble_type' not in config:
            raise ValueError("Ensemble type must be specified in the configuration")
        
        if 'base_models' not in config or not config['base_models']:
            raise ValueError("Base models must be specified in the configuration")
        
        # Create base models
        base_models = []
        for model_config in config['base_models']:
            if 'type' not in model_config:
                raise ValueError("Model type must be specified for each base model")
            
            model_type = model_config['type'].lower()
            model_kwargs = {k: v for k, v in model_config.items() if k != 'type'}
            
            # Add input_shape for neural network models if needed
            if model_type in ['lstm', 'gru', 'transformer'] and input_shape is not None:
                model_kwargs['input_shape'] = input_shape
            
            # Create the model
            model = ModelFactory.create_model(model_type, **model_kwargs)
            base_models.append(model)
        
        # Create ensemble model
        ensemble_type = config['ensemble_type'].lower()
        ensemble_kwargs = {}
        
        if 'name' in config:
            ensemble_kwargs['name'] = config['name']
        
        if ensemble_type == 'stacking':
            # Create meta-model if specified
            if 'meta_model' in config:
                meta_config = config['meta_model']
                meta_type = meta_config['type'].lower()
                
                if meta_type == 'linear':
                    from sklearn.linear_model import LinearRegression
                    meta_model = LinearRegression()
                elif meta_type == 'random_forest':
                    from sklearn.ensemble import RandomForestRegressor
                    meta_kwargs = {k: v for k, v in meta_config.items() if k != 'type'}
                    meta_model = RandomForestRegressor(**meta_kwargs)
                else:
                    raise ValueError(f"Unsupported meta-model type: {meta_type}")
                
                ensemble_kwargs['meta_model'] = meta_model
            
            if 'use_features_in_meta' in config:
                ensemble_kwargs['use_features_in_meta'] = config['use_features_in_meta']
            
            return ModelFactory.create_stacking_ensemble_model(base_models, **ensemble_kwargs)
        
        elif ensemble_type == 'voting':
            if 'weights' in config:
                ensemble_kwargs['weights'] = config['weights']
            
            return ModelFactory.create_voting_ensemble_model(base_models, **ensemble_kwargs)
        
        else:
            raise ValueError(f"Unsupported ensemble type: {ensemble_type}")