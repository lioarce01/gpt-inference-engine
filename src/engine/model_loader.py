"""Model loader for loading PyTorch checkpoints and configurations.

This module provides a unified interface for loading models, using CheckpointManager
as the primary method when available, with fallback to direct PyTorch loading.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
from safetensors.torch import load_file as safetensors_load

# Import CheckpointManager from src.models
try:
    from src.models import CheckpointManager
    CHECKPOINT_MANAGER_AVAILABLE = True
except ImportError:
    CHECKPOINT_MANAGER_AVAILABLE = False
    CheckpointManager = None

# Make available at module level for external access
__all__ = ['ModelLoader', 'ModelLoadError', 'CHECKPOINT_MANAGER_AVAILABLE']


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass


class ModelLoader:
    """Loads model checkpoints and configurations."""
    
    @staticmethod
    def load_configs(model_dir: str) -> Dict[str, Any]:
        """
        Load configuration files from model directory.
        
        Args:
            model_dir: Path to directory containing config files
            
        Returns:
            Dictionary with 'config', 'model_args', and 'export_meta' keys
        """
        model_dir = Path(model_dir)
        configs = {}
        
        # Load config.json
        config_path = model_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                configs['config'] = json.load(f)
        else:
            # Try without .json extension
            config_path = model_dir / "config"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    configs['config'] = json.load(f)
        
        # Load model_args.json
        model_args_path = model_dir / "model_args.json"
        if model_args_path.exists():
            with open(model_args_path, 'r') as f:
                configs['model_args'] = json.load(f)
        else:
            # Try without .json extension
            model_args_path = model_dir / "model_args"
            if model_args_path.exists():
                with open(model_args_path, 'r') as f:
                    configs['model_args'] = json.load(f)
        
        # Load export_meta.json
        export_meta_path = model_dir / "export_meta.json"
        if export_meta_path.exists():
            with open(export_meta_path, 'r') as f:
                configs['export_meta'] = json.load(f)
        else:
            # Try without .json extension
            export_meta_path = model_dir / "export_meta"
            if export_meta_path.exists():
                with open(export_meta_path, 'r') as f:
                    configs['export_meta'] = json.load(f)
        
        return configs
    
    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        device: str = 'cpu',
        use_safetensors: Optional[bool] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Load model checkpoint from .pt or .safetensors file.
        
        Args:
            checkpoint_path: Path to checkpoint file or directory containing checkpoint
            device: Device to load model on ('cpu' or 'cuda')
            use_safetensors: If True, prefer safetensors. If None, auto-detect.
            
        Returns:
            Tuple of (model, metadata)
        """
        checkpoint_path = Path(checkpoint_path)
        
        # If it's a directory, look for checkpoint files inside
        if checkpoint_path.is_dir():
            ckpt_pt = checkpoint_path / "ckpt.pt"
            ckpt_safetensors = checkpoint_path / "model.safetensors"
            
            # Auto-detect format if not specified
            if use_safetensors is None:
                use_safetensors = ckpt_safetensors.exists() and ckpt_pt.exists()
                if not use_safetensors and ckpt_safetensors.exists():
                    use_safetensors = True
                elif ckpt_pt.exists():
                    use_safetensors = False
            
            if use_safetensors and ckpt_safetensors.exists():
                checkpoint_path = ckpt_safetensors
            elif ckpt_pt.exists():
                checkpoint_path = ckpt_pt
            else:
                raise ModelLoadError(
                    f"No checkpoint found in directory: {checkpoint_path}. "
                    f"Expected ckpt.pt or model.safetensors"
                )
        
        # Load checkpoint
        if checkpoint_path.suffix == '.safetensors' or str(checkpoint_path).endswith('.safetensors'):
            return ModelLoader._load_safetensors(checkpoint_path, device)
        else:
            return ModelLoader._load_pytorch(checkpoint_path, device)
    
    @staticmethod
    def _load_pytorch(checkpoint_path: Path, device: str) -> Tuple[Any, Dict[str, Any]]:
        """Load PyTorch .pt checkpoint."""
        try:
            from collections import OrderedDict
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Checkpoint format can vary:
            # 1. Direct model object (nn.Module): checkpoint is the model
            # 2. Dict with 'model' key: checkpoint['model'] is the model
            # 3. OrderedDict (state_dict): need to reconstruct model
            # 4. Dict with 'state_dict' key: need to reconstruct model
            
            # Check if it's a state_dict (OrderedDict or dict with only tensor keys)
            if isinstance(checkpoint, OrderedDict):
                # This is a state_dict, cannot reconstruct without model class
                raise ModelLoadError(
                    "Checkpoint is a state_dict (OrderedDict). "
                    "Cannot reconstruct model without model class. "
                    "Please use CheckpointManager.load_model() or ensure checkpoint contains full model object."
                )
            
            if isinstance(checkpoint, dict):
                # Check if it's a state_dict (dict with tensor values, no 'model' key)
                has_model_key = 'model' in checkpoint
                has_state_dict_key = 'state_dict' in checkpoint
                
                # If it looks like a state_dict (all values are tensors and no 'model' key)
                if not has_model_key and not has_state_dict_key:
                    # Check if all values are tensors (likely a state_dict)
                    all_tensors = all(
                        isinstance(v, torch.Tensor) for v in checkpoint.values()
                    ) if checkpoint else False
                    
                    if all_tensors:
                        raise ModelLoadError(
                            "Checkpoint appears to be a state_dict (dict of tensors). "
                            "Cannot reconstruct model without model class. "
                            "Please use CheckpointManager.load_model() or ensure checkpoint contains full model object."
                        )
                
                # Try to extract model and metadata
                model = checkpoint.get('model') or checkpoint.get('state_dict')
                if model is None:
                    raise ModelLoadError(
                        "Checkpoint dict does not contain 'model' or 'state_dict' key. "
                        "Cannot determine how to load model."
                    )
                
                # If we got state_dict, we still can't reconstruct
                if isinstance(model, (dict, OrderedDict)) and not isinstance(model, torch.nn.Module):
                    raise ModelLoadError(
                        "Checkpoint contains state_dict but model class not available. "
                        "Please use CheckpointManager.load_model() or ensure checkpoint contains full model object."
                    )
                
                metadata = {k: v for k, v in checkpoint.items() if k not in ('model', 'state_dict')}
                return model, metadata
            else:
                # Checkpoint is the model itself (or something else)
                if isinstance(checkpoint, torch.nn.Module):
                    return checkpoint, {}
                else:
                    # Unknown format
                    raise ModelLoadError(
                        f"Unknown checkpoint format. Got type: {type(checkpoint)}. "
                        "Expected nn.Module, dict with 'model' key, or state_dict."
                    )
                
        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(f"Failed to load PyTorch checkpoint: {e}") from e
    
    @staticmethod
    def _load_safetensors(checkpoint_path: Path, device: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load safetensors checkpoint."""
        try:
            state_dict = safetensors_load(str(checkpoint_path), device=device)
            # Safetensors only contains weights, need model class to load into
            # Return state_dict and empty metadata for now
            return state_dict, {}
        except Exception as e:
            raise ModelLoadError(f"Failed to load safetensors checkpoint: {e}") from e
    
    @staticmethod
    def load_model_from_dir(
        model_dir: str,
        device: str = 'auto',
        checkpoint_file: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Load model and configs from a model directory.
        
        This method prioritizes CheckpointManager when available, as it handles
        model reconstruction from state_dicts. Falls back to direct loading if needed.
        
        Args:
            model_dir: Directory containing model files
            device: Device to load on ('auto', 'cpu', or 'cuda')
            checkpoint_file: Optional specific checkpoint file to load
            
        Returns:
            Tuple of (model, full_metadata_dict)
        """
        model_dir = Path(model_dir)
        
        # Determine device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load configs first (needed for metadata)
        configs = ModelLoader.load_configs(model_dir)
        
        # Determine checkpoint path
        if checkpoint_file:
            checkpoint_path = model_dir / checkpoint_file
        else:
            # Look for common checkpoint filenames
            for ckpt_name in ['ckpt.pt', 'checkpoint.pt', 'model.pt']:
                potential_path = model_dir / ckpt_name
                if potential_path.exists():
                    checkpoint_path = potential_path
                    break
            else:
                # If no checkpoint file found, use directory as path
                checkpoint_path = model_dir
        
        # Try CheckpointManager first (handles state_dict reconstruction)
        if CHECKPOINT_MANAGER_AVAILABLE and checkpoint_path.exists():
            try:
                checkpoint_file_str = str(checkpoint_path)
                model, checkpoint_metadata = CheckpointManager.load_model(
                    checkpoint_file_str,
                    device=device
                )
                # Merge metadata
                metadata = {
                    **configs,
                    **checkpoint_metadata
                }
                return model, metadata
            except Exception as e:
                # If CheckpointManager fails, fall back to direct loading
                print(f"Warning: CheckpointManager failed, using fallback: {e}")
        
        # Fallback: Load checkpoint directly
        model, checkpoint_metadata = ModelLoader.load_checkpoint(
            checkpoint_path,
            device=device
        )
        
        # Merge metadata
        metadata = {
            **configs,
            **checkpoint_metadata
        }
        
        return model, metadata

