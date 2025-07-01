"""
Feature Store

Centralized storage and management of computed features for ML models.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
import hashlib
from dataclasses import dataclass, asdict
import sqlite3


@dataclass
class FeatureMetadata:
    """Metadata for a feature set"""
    feature_set_id: str
    name: str
    version: str
    description: str
    symbols: List[str]
    timeframe: str
    features: List[str]
    created_at: datetime
    created_by: str
    data_sources: List[str]
    computation_time: float
    file_path: str
    file_size: int
    checksum: str


class FeatureStore:
    """
    Centralized feature store for trading data
    
    Manages storage, versioning, and retrieval of computed features
    for machine learning models.
    """
    
    def __init__(
        self,
        store_path: Union[str, Path],
        backend: str = "parquet"  # "parquet", "hdf5", "pickle"
    ):
        """
        Initialize feature store
        
        Args:
            store_path: Path to feature store directory
            backend: Storage backend format
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.backend = backend
        self.logger = logging.getLogger(__name__)
        
        # Initialize metadata database
        self.metadata_db = self.store_path / "metadata.db"
        self._initialize_metadata_db()
        
        # Cache for loaded features
        self._feature_cache = {}
        
    def _initialize_metadata_db(self) -> None:
        """Initialize SQLite metadata database"""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_sets (
                    feature_set_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    description TEXT,
                    symbols TEXT,  -- JSON array
                    timeframe TEXT,
                    features TEXT,  -- JSON array
                    created_at TEXT,
                    created_by TEXT,
                    data_sources TEXT,  -- JSON array
                    computation_time REAL,
                    file_path TEXT,
                    file_size INTEGER,
                    checksum TEXT,
                    UNIQUE(name, version)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_name_version 
                ON feature_sets(name, version)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbols 
                ON feature_sets(symbols)
            """)
    
    def store_features(
        self,
        features: pd.DataFrame,
        name: str,
        version: str,
        description: str = "",
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        data_sources: Optional[List[str]] = None,
        created_by: str = "system",
        overwrite: bool = False
    ) -> str:
        """
        Store feature set
        
        Args:
            features: Feature DataFrame
            name: Feature set name
            version: Version string
            description: Description of features
            symbols: List of symbols these features apply to
            timeframe: Data timeframe
            data_sources: List of data sources used
            created_by: Creator identifier
            overwrite: Whether to overwrite existing feature set
            
        Returns:
            Feature set ID
        """
        start_time = datetime.now()
        
        # Generate feature set ID
        feature_set_id = self._generate_feature_set_id(name, version)
        
        # Check if already exists
        if not overwrite and self._feature_set_exists(name, version):
            raise ValueError(f"Feature set {name} v{version} already exists")
        
        # Validate features DataFrame
        self._validate_features(features)
        
        # Generate file path
        file_path = self._generate_file_path(feature_set_id)
        
        # Store features to file
        self._save_features_to_file(features, file_path)
        
        # Calculate file metadata
        file_size = file_path.stat().st_size
        checksum = self._calculate_checksum(file_path)
        
        computation_time = (datetime.now() - start_time).total_seconds()
        
        # Create metadata
        metadata = FeatureMetadata(
            feature_set_id=feature_set_id,
            name=name,
            version=version,
            description=description,
            symbols=symbols or [],
            timeframe=timeframe,
            features=list(features.columns),
            created_at=start_time,
            created_by=created_by,
            data_sources=data_sources or [],
            computation_time=computation_time,
            file_path=str(file_path),
            file_size=file_size,
            checksum=checksum
        )
        
        # Store metadata
        self._store_metadata(metadata)
        
        self.logger.info(f"Stored feature set {name} v{version} with {len(features)} rows, {len(features.columns)} features")
        return feature_set_id
    
    def load_features(
        self,
        name: str,
        version: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[pd.DataFrame, FeatureMetadata]:
        """
        Load feature set
        
        Args:
            name: Feature set name
            version: Specific version (latest if None)
            use_cache: Whether to use cached features
            
        Returns:
            Tuple of (features DataFrame, metadata)
        """
        # Get latest version if not specified
        if version is None:
            version = self.get_latest_version(name)
            if version is None:
                raise ValueError(f"No feature sets found for name: {name}")
        
        feature_set_id = self._generate_feature_set_id(name, version)
        
        # Check cache first
        if use_cache and feature_set_id in self._feature_cache:
            self.logger.debug(f"Loading features from cache: {name} v{version}")
            return self._feature_cache[feature_set_id]
        
        # Load metadata
        metadata = self._load_metadata(name, version)
        if metadata is None:
            raise ValueError(f"Feature set not found: {name} v{version}")
        
        # Load features from file
        file_path = Path(metadata.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Feature file not found: {file_path}")
        
        # Verify checksum
        if self._calculate_checksum(file_path) != metadata.checksum:
            raise ValueError(f"Checksum mismatch for feature set {name} v{version}")
        
        features = self._load_features_from_file(file_path)
        
        # Cache if enabled
        if use_cache:
            self._feature_cache[feature_set_id] = (features, metadata)
        
        self.logger.info(f"Loaded feature set {name} v{version} with {len(features)} rows")
        return features, metadata
    
    def list_feature_sets(
        self,
        name_pattern: Optional[str] = None,
        symbols: Optional[List[str]] = None
    ) -> List[FeatureMetadata]:
        """
        List available feature sets
        
        Args:
            name_pattern: Filter by name pattern (SQL LIKE)
            symbols: Filter by symbols
            
        Returns:
            List of feature metadata
        """
        query = "SELECT * FROM feature_sets WHERE 1=1"
        params = []
        
        if name_pattern:
            query += " AND name LIKE ?"
            params.append(name_pattern)
        
        if symbols:
            # Filter by symbols intersection
            symbol_conditions = []
            for symbol in symbols:
                symbol_conditions.append("symbols LIKE ?")
                params.append(f'%"{symbol}"%')
            
            if symbol_conditions:
                query += " AND (" + " OR ".join(symbol_conditions) + ")"
        
        query += " ORDER BY name, version DESC"
        
        with sqlite3.connect(self.metadata_db) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            feature_sets = []
            for row in cursor.fetchall():
                metadata = self._row_to_metadata(row)
                feature_sets.append(metadata)
        
        return feature_sets
    
    def get_latest_version(self, name: str) -> Optional[str]:
        """Get latest version of feature set"""
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute(
                "SELECT version FROM feature_sets WHERE name = ? ORDER BY created_at DESC LIMIT 1",
                (name,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
    
    def delete_feature_set(self, name: str, version: str) -> bool:
        """
        Delete feature set
        
        Args:
            name: Feature set name
            version: Version to delete
            
        Returns:
            True if successful
        """
        # Load metadata to get file path
        metadata = self._load_metadata(name, version)
        if metadata is None:
            return False
        
        # Delete file
        file_path = Path(metadata.file_path)
        if file_path.exists():
            file_path.unlink()
        
        # Delete metadata
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute(
                "DELETE FROM feature_sets WHERE name = ? AND version = ?",
                (name, version)
            )
            deleted = cursor.rowcount > 0
        
        # Remove from cache
        feature_set_id = self._generate_feature_set_id(name, version)
        self._feature_cache.pop(feature_set_id, None)
        
        if deleted:
            self.logger.info(f"Deleted feature set {name} v{version}")
        
        return deleted
    
    def get_feature_info(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information about feature set"""
        if version is None:
            version = self.get_latest_version(name)
        
        metadata = self._load_metadata(name, version)
        if metadata is None:
            return {}
        
        # Additional statistics
        try:
            features, _ = self.load_features(name, version, use_cache=False)
            
            stats = {
                'shape': features.shape,
                'memory_usage_mb': features.memory_usage(deep=True).sum() / 1024 / 1024,
                'date_range': {
                    'start': features.index.min() if isinstance(features.index, pd.DatetimeIndex) else None,
                    'end': features.index.max() if isinstance(features.index, pd.DatetimeIndex) else None
                },
                'feature_types': features.dtypes.to_dict(),
                'missing_values': features.isnull().sum().to_dict()
            }
        except Exception as e:
            self.logger.warning(f"Could not load statistics for {name} v{version}: {e}")
            stats = {}
        
        return {
            'metadata': asdict(metadata),
            'statistics': stats
        }
    
    def update_features(
        self,
        name: str,
        version: str,
        new_features: pd.DataFrame,
        append: bool = False
    ) -> str:
        """
        Update existing feature set
        
        Args:
            name: Feature set name
            version: Version to update
            new_features: New features to add/replace
            append: Whether to append or replace
            
        Returns:
            New feature set ID
        """
        if append:
            # Load existing features and append
            existing_features, metadata = self.load_features(name, version)
            
            # Combine features
            if isinstance(existing_features.index, pd.DatetimeIndex) and isinstance(new_features.index, pd.DatetimeIndex):
                # Time series append
                combined_features = pd.concat([existing_features, new_features]).sort_index()
                combined_features = combined_features[~combined_features.index.duplicated(keep='last')]
            else:
                # Regular append
                combined_features = pd.concat([existing_features, new_features], ignore_index=True)
            
            # Create new version
            new_version = self._increment_version(version)
            return self.store_features(
                combined_features,
                name,
                new_version,
                description=f"Updated from v{version} (appended {len(new_features)} rows)",
                symbols=metadata.symbols,
                timeframe=metadata.timeframe,
                overwrite=False
            )
        else:
            # Replace existing version
            return self.store_features(
                new_features,
                name,
                version,
                description="Updated feature set",
                overwrite=True
            )
    
    def _generate_feature_set_id(self, name: str, version: str) -> str:
        """Generate unique feature set ID"""
        return f"{name}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _generate_file_path(self, feature_set_id: str) -> Path:
        """Generate file path for feature set"""
        if self.backend == "parquet":
            return self.store_path / f"{feature_set_id}.parquet"
        elif self.backend == "hdf5":
            return self.store_path / f"{feature_set_id}.h5"
        else:  # pickle
            return self.store_path / f"{feature_set_id}.pkl"
    
    def _save_features_to_file(self, features: pd.DataFrame, file_path: Path) -> None:
        """Save features to file"""
        if self.backend == "parquet":
            features.to_parquet(file_path, compression='snappy')
        elif self.backend == "hdf5":
            features.to_hdf(file_path, key='features', mode='w', complevel=9)
        else:  # pickle
            with open(file_path, 'wb') as f:
                pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_features_from_file(self, file_path: Path) -> pd.DataFrame:
        """Load features from file"""
        if self.backend == "parquet":
            return pd.read_parquet(file_path)
        elif self.backend == "hdf5":
            return pd.read_hdf(file_path, key='features')
        else:  # pickle
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _validate_features(self, features: pd.DataFrame) -> None:
        """Validate features DataFrame"""
        if features.empty:
            raise ValueError("Features DataFrame is empty")
        
        if features.isnull().all().any():
            null_cols = features.columns[features.isnull().all()].tolist()
            raise ValueError(f"Features contain entirely null columns: {null_cols}")
    
    def _feature_set_exists(self, name: str, version: str) -> bool:
        """Check if feature set exists"""
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM feature_sets WHERE name = ? AND version = ?",
                (name, version)
            )
            return cursor.fetchone() is not None
    
    def _store_metadata(self, metadata: FeatureMetadata) -> None:
        """Store feature set metadata"""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO feature_sets VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                metadata.feature_set_id,
                metadata.name,
                metadata.version,
                metadata.description,
                json.dumps(metadata.symbols),
                metadata.timeframe,
                json.dumps(metadata.features),
                metadata.created_at.isoformat(),
                metadata.created_by,
                json.dumps(metadata.data_sources),
                metadata.computation_time,
                metadata.file_path,
                metadata.file_size,
                metadata.checksum
            ))
    
    def _load_metadata(self, name: str, version: str) -> Optional[FeatureMetadata]:
        """Load feature set metadata"""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM feature_sets WHERE name = ? AND version = ?",
                (name, version)
            )
            row = cursor.fetchone()
            
            return self._row_to_metadata(row) if row else None
    
    def _row_to_metadata(self, row: sqlite3.Row) -> FeatureMetadata:
        """Convert database row to metadata object"""
        return FeatureMetadata(
            feature_set_id=row['feature_set_id'],
            name=row['name'],
            version=row['version'],
            description=row['description'],
            symbols=json.loads(row['symbols']),
            timeframe=row['timeframe'],
            features=json.loads(row['features']),
            created_at=datetime.fromisoformat(row['created_at']),
            created_by=row['created_by'],
            data_sources=json.loads(row['data_sources']),
            computation_time=row['computation_time'],
            file_path=row['file_path'],
            file_size=row['file_size'],
            checksum=row['checksum']
        )
    
    def _increment_version(self, version: str) -> str:
        """Increment version string"""
        try:
            parts = version.split('.')
            if len(parts) >= 2:
                parts[-1] = str(int(parts[-1]) + 1)
                return '.'.join(parts)
            else:
                return f"{version}.1"
        except ValueError:
            return f"{version}_1"
    
    def clear_cache(self) -> None:
        """Clear feature cache"""
        self._feature_cache.clear()
        self.logger.info("Feature cache cleared")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_size = 0
        total_files = 0
        
        for file_path in self.store_path.glob("*"):
            if file_path.is_file() and file_path.name != "metadata.db":
                total_size += file_path.stat().st_size
                total_files += 1
        
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM feature_sets")
            total_feature_sets = cursor.fetchone()[0]
        
        return {
            'total_size_mb': total_size / 1024 / 1024,
            'total_files': total_files,
            'total_feature_sets': total_feature_sets,
            'cache_size': len(self._feature_cache),
            'backend': self.backend,
            'store_path': str(self.store_path)
        }