"""
Model definitions and checkpoint management.

This module provides access to:
- OptimizedGPT model implementation (GQA, RMSNorm, RoPE, SwiGLU)
- CheckpointManager for loading/saving models
"""

from .checkpoint import CheckpointManager
from .model_optimized import OptimizedGPT, OptimizedGPTConfig

__all__ = [
    'CheckpointManager',
    'OptimizedGPT',
    'OptimizedGPTConfig',
]

