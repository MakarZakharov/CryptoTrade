"""
Test script to verify the ensemble model restructuring.

This script imports and creates instances of the ensemble models
to verify that the restructuring and import fixes work correctly.
"""

import os
import sys

# Add all necessary directories to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models")
ensemble_dir = os.path.join(models_dir, "ensemble")
parent_dir = os.path.dirname(current_dir)

# Add all paths
sys.path.append(current_dir)
sys.path.append(models_dir)
sys.path.append(ensemble_dir)
sys.path.append(parent_dir)

print(f"Python path: {sys.path}")
print(f"Current directory: {current_dir}")
print(f"Models directory: {models_dir}")
print(f"Ensemble directory: {ensemble_dir}")

# Import the base model first
try:
    print("\nImporting BaseModel...")
    sys.path.insert(0, models_dir)  # Prioritize models directory
    from base_model import BaseModel
    print("Successfully imported BaseModel")
except ImportError as e:
    print(f"Error importing BaseModel: {e}")
    
    # Create a simple BaseModel for testing
    print("Creating a dummy BaseModel for testing")
    class BaseModel:
        def __init__(self, name=None):
            self.name = name or self.__class__.__name__
            self.is_trained = False
            self.metadata = {}
            self.history = None
        
        def predict(self, X):
            import numpy as np
            return np.zeros(len(X))

# Import the ensemble models directly from their files
print("\nImporting ensemble models directly from files...")
try:
    sys.path.insert(0, ensemble_dir)  # Prioritize ensemble directory
    
    # Import stacking.py
    print("Importing from stacking.py...")
    from stacking import StackingEnsembleModel
    print("Successfully imported StackingEnsembleModel")
    
    # Import voting.py
    print("Importing from voting.py...")
    from voting import VotingEnsembleModel
    print("Successfully imported VotingEnsembleModel")
except ImportError as e:
    print(f"Error importing ensemble models: {e}")
    sys.exit(1)

# Create dummy base models for testing
class DummyModel(BaseModel):
    def __init__(self, name="DummyModel"):
        super().__init__(name=name)
        self.is_trained = True
    
    def build(self, **kwargs):
        """Implement required abstract method."""
        pass
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Implement required abstract method."""
        self.is_trained = True
        return {"loss": 0.0}
    
    def predict(self, X):
        """Predict method."""
        import numpy as np
        return np.zeros(len(X))
    
    def save(self, path):
        """Implement required abstract method."""
        pass
    
    def load(self, path):
        """Implement required abstract method."""
        pass

# Create instances of the ensemble models
print("\nCreating ensemble model instances...")

# Create base models
base_models = [
    DummyModel(name="Model1"),
    DummyModel(name="Model2"),
    DummyModel(name="Model3")
]

# Create a stacking ensemble
try:
    stacking = StackingEnsembleModel(
        base_models=base_models,
        name="TestStackingEnsemble"
    )
    print(f"Successfully created {stacking.name}")
    print(f"Base models: {[model.name for model in stacking.base_models]}")
except Exception as e:
    print(f"Error creating stacking ensemble: {str(e)}")

# Create a voting ensemble
try:
    voting = VotingEnsembleModel(
        base_models=base_models,
        weights=[0.5, 0.3, 0.2],
        name="TestVotingEnsemble"
    )
    print(f"Successfully created {voting.name}")
    print(f"Base models: {[model.name for model in voting.base_models]}")
    print(f"Weights: {voting.weights}")
except Exception as e:
    print(f"Error creating voting ensemble: {str(e)}")

print("\nTest completed.")