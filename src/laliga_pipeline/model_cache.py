"""Model caching and persistence system for Dixon-Coles models."""

import pickle
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd
from datetime import datetime

class ModelCache:
    """Handles saving, loading, and validation of trained Dixon-Coles models."""
    
    def __init__(self, cache_dir: str = "./model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_data_hash(self, matches: pd.DataFrame) -> str:
        """Generate hash of training data for cache validation."""
        # Use shape, date range, and sample of data for hash
        data_info = {
            'shape': matches.shape,
            'date_min': str(matches['Date'].min()),
            'date_max': str(matches['Date'].max()),
            'sample_hash': hashlib.md5(str(matches.head(10).values).encode()).hexdigest()
        }
        return hashlib.md5(str(data_info).encode()).hexdigest()
    
    def _get_model_key(self, data_hash: str, model_params: Dict) -> str:
        """Generate unique key for model based on data and parameters."""
        key_data = {
            'data_hash': data_hash,
            'params': model_params
        }
        return hashlib.md5(str(key_data).encode()).hexdigest()
    
    def save_model(self, model_params: Dict, training_data: pd.DataFrame, 
                   model_config: Dict, performance_metrics: Optional[Dict] = None) -> str:
        """Save trained model with metadata."""
        data_hash = self._get_data_hash(training_data)
        model_key = self._get_model_key(data_hash, model_config)
        
        model_data = {
            'params': model_params,
            'config': model_config,
            'data_hash': data_hash,
            'training_size': len(training_data),
            'date_range': {
                'min': str(training_data['Date'].min()),
                'max': str(training_data['Date'].max())
            },
            'created_at': datetime.now().isoformat(),
            'performance': performance_metrics or {}
        }
        
        # Save model parameters
        model_file = self.cache_dir / f"model_{model_key}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
            
        # Save metadata
        metadata_file = self.cache_dir / f"metadata_{model_key}.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'model_key': model_key,
                'data_hash': data_hash,
                'config': model_config,
                'training_size': len(training_data),
                'date_range': model_data['date_range'],
                'created_at': model_data['created_at'],
                'performance': performance_metrics or {}
            }, f, indent=2)
            
        return model_key
    
    def load_model(self, model_key: str) -> Optional[Dict]:
        """Load cached model by key."""
        model_file = self.cache_dir / f"model_{model_key}.pkl"
        if not model_file.exists():
            return None
            
        with open(model_file, 'rb') as f:
            return pickle.load(f)
    
    def find_compatible_model(self, training_data: pd.DataFrame, 
                            model_config: Dict) -> Optional[str]:
        """Find existing model compatible with current data and config."""
        data_hash = self._get_data_hash(training_data)
        model_key = self._get_model_key(data_hash, model_config)
        
        # Check for exact match first
        if (self.cache_dir / f"model_{model_key}.pkl").exists():
            return model_key
            
        # Check for compatible models (same config, subset of data)
        for metadata_file in self.cache_dir.glob("metadata_*.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            # Check if config matches
            if metadata['config'] == model_config:
                # Check if cached model was trained on subset of current data
                cached_max_date = pd.to_datetime(metadata['date_range']['max'])
                current_max_date = training_data['Date'].max()
                
                if cached_max_date <= current_max_date:
                    # Model is compatible but may need updating
                    return metadata['model_key']
                    
        return None
    
    def needs_update(self, model_key: str, current_data: pd.DataFrame, 
                    days_threshold: int = 30) -> bool:
        """Check if cached model needs updating based on new data."""
        metadata_file = self.cache_dir / f"metadata_{model_key}.json"
        if not metadata_file.exists():
            return True
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        cached_max_date = pd.to_datetime(metadata['date_range']['max'])
        current_max_date = current_data['Date'].max()
        
        # Check if there's significant new data
        days_diff = (current_max_date - cached_max_date).days
        return days_diff > days_threshold
    
    def list_models(self) -> list:
        """List all cached models with metadata."""
        models = []
        for metadata_file in self.cache_dir.glob("metadata_*.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                models.append(metadata)
        return sorted(models, key=lambda x: x['created_at'], reverse=True)
    
    def cleanup_old_models(self, keep_latest: int = 5):
        """Remove old cached models, keeping only the latest ones."""
        models = self.list_models()
        if len(models) <= keep_latest:
            return
            
        for model in models[keep_latest:]:
            model_key = model['model_key']
            model_file = self.cache_dir / f"model_{model_key}.pkl"
            metadata_file = self.cache_dir / f"metadata_{model_key}.json"
            
            if model_file.exists():
                model_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()