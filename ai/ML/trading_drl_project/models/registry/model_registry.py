"""
Model Registry

Central registry for managing model lifecycle and metadata.
"""

import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import logging


@dataclass
class ModelMetadata:
    """Model metadata structure"""
    model_id: str
    name: str
    version: str
    model_type: str
    framework: str  # 'pytorch', 'tensorflow', 'stable_baselines3'
    created_at: datetime
    created_by: str
    description: str
    tags: List[str]
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    file_path: str
    file_size: int
    file_hash: str
    parent_model: Optional[str] = None
    status: str = "active"  # active, archived, deprecated


class ModelRegistry:
    """
    Model registry for tracking and managing ML models
    
    Provides centralized model management with versioning,
    metadata tracking, and lifecycle management.
    """
    
    def __init__(self, registry_path: Union[str, Path]):
        """
        Initialize model registry
        
        Args:
            registry_path: Path to registry storage directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.registry_path / "metadata.json"
        self.models_dir = self.registry_path / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata from storage"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    # Convert datetime strings back to datetime objects
                    for model_id, meta in data.items():
                        meta['created_at'] = datetime.fromisoformat(meta['created_at'])
                    return data
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self) -> None:
        """Save metadata to storage"""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_data = {}
            for model_id, meta in self.metadata.items():
                meta_copy = meta.copy()
                meta_copy['created_at'] = meta_copy['created_at'].isoformat()
                serializable_data[model_id] = meta_copy
                
            with open(self.metadata_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def register_model(
        self,
        model: Any,
        name: str,
        version: str,
        model_type: str,
        framework: str,
        created_by: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        parent_model: Optional[str] = None
    ) -> str:
        """
        Register a new model
        
        Args:
            model: Model object to register
            name: Model name
            version: Model version
            model_type: Type of model (e.g., 'ppo', 'sac', 'dqn')
            framework: Framework used
            created_by: Creator identifier
            description: Model description
            tags: List of tags
            parameters: Model parameters/config
            metrics: Performance metrics
            parent_model: Parent model ID if this is derived
            
        Returns:
            Model ID
        """
        # Generate unique model ID
        model_id = f"{name}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save model file
        model_filename = f"{model_id}.pkl"
        model_path = self.models_dir / model_filename
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            self.logger.error(f"Error saving model {model_id}: {e}")
            raise
        
        # Calculate file hash and size
        file_hash = self._calculate_file_hash(model_path)
        file_size = model_path.stat().st_size
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            model_type=model_type,
            framework=framework,
            created_at=datetime.now(),
            created_by=created_by,
            description=description,
            tags=tags or [],
            parameters=parameters or {},
            metrics=metrics or {},
            file_path=str(model_path),
            file_size=file_size,
            file_hash=file_hash,
            parent_model=parent_model
        )
        
        # Store metadata
        self.metadata[model_id] = asdict(metadata)
        self._save_metadata()
        
        self.logger.info(f"Registered model {model_id}")
        return model_id
    
    def load_model(self, model_id: str) -> Any:
        """
        Load model by ID
        
        Args:
            model_id: Model identifier
            
        Returns:
            Loaded model object
        """
        if model_id not in self.metadata:
            raise ValueError(f"Model {model_id} not found in registry")
        
        metadata = self.metadata[model_id]
        model_path = Path(metadata['file_path'])
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
            self.logger.info(f"Loaded model {model_id}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model metadata"""
        return self.metadata.get(model_id)
    
    def list_models(
        self,
        name: Optional[str] = None,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: str = "active"
    ) -> List[Dict[str, Any]]:
        """
        List models with optional filtering
        
        Args:
            name: Filter by model name
            model_type: Filter by model type
            tags: Filter by tags (any match)
            status: Filter by status
            
        Returns:
            List of model metadata
        """
        models = []
        
        for model_id, metadata in self.metadata.items():
            # Apply filters
            if status and metadata.get('status', 'active') != status:
                continue
                
            if name and metadata['name'] != name:
                continue
                
            if model_type and metadata['model_type'] != model_type:
                continue
                
            if tags and not any(tag in metadata.get('tags', []) for tag in tags):
                continue
                
            models.append(metadata)
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x['created_at'], reverse=True)
        return models
    
    def get_latest_model(
        self,
        name: str,
        model_type: Optional[str] = None
    ) -> Optional[str]:
        """
        Get latest model ID for given name and type
        
        Args:
            name: Model name
            model_type: Model type filter
            
        Returns:
            Latest model ID or None
        """
        models = self.list_models(name=name, model_type=model_type)
        
        if models:
            return models[0]['model_id']
        
        return None
    
    def update_model_status(self, model_id: str, status: str) -> bool:
        """
        Update model status
        
        Args:
            model_id: Model identifier
            status: New status ('active', 'archived', 'deprecated')
            
        Returns:
            True if successful
        """
        if model_id not in self.metadata:
            return False
        
        self.metadata[model_id]['status'] = status
        self._save_metadata()
        
        self.logger.info(f"Updated model {model_id} status to {status}")
        return True
    
    def update_model_metrics(self, model_id: str, metrics: Dict[str, float]) -> bool:
        """
        Update model performance metrics
        
        Args:
            model_id: Model identifier
            metrics: Performance metrics to update
            
        Returns:
            True if successful
        """
        if model_id not in self.metadata:
            return False
        
        self.metadata[model_id]['metrics'].update(metrics)
        self._save_metadata()
        
        self.logger.info(f"Updated metrics for model {model_id}")
        return True
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete model and its files
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if successful
        """
        if model_id not in self.metadata:
            return False
        
        metadata = self.metadata[model_id]
        model_path = Path(metadata['file_path'])
        
        # Delete model file
        if model_path.exists():
            model_path.unlink()
        
        # Remove from metadata
        del self.metadata[model_id]
        self._save_metadata()
        
        self.logger.info(f"Deleted model {model_id}")
        return True
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_models = len(self.metadata)
        active_models = len([m for m in self.metadata.values() if m.get('status', 'active') == 'active'])
        
        model_types = {}
        frameworks = {}
        total_size = 0
        
        for metadata in self.metadata.values():
            # Count by type
            model_type = metadata['model_type']
            model_types[model_type] = model_types.get(model_type, 0) + 1
            
            # Count by framework
            framework = metadata['framework']
            frameworks[framework] = frameworks.get(framework, 0) + 1
            
            # Total size
            total_size += metadata['file_size']
        
        return {
            'total_models': total_models,
            'active_models': active_models,
            'archived_models': total_models - active_models,
            'model_types': model_types,
            'frameworks': frameworks,
            'total_size_mb': total_size / (1024 * 1024),
            'registry_path': str(self.registry_path)
        }