"""Dependencies for FastAPI dependency injection."""
import os
from functools import lru_cache
from typing import Optional
from src.engine.inference_engine import InferenceEngine


# Global engine instance (alternative to lru_cache for more control)
_engine_instance: Optional[InferenceEngine] = None


@lru_cache(maxsize=1)
def get_engine() -> InferenceEngine:
    """
    Get or create inference engine instance (singleton pattern).
    
    Uses lru_cache to ensure only one instance is created.
    Thread-safe for read operations.
    
    Returns:
        InferenceEngine instance
    """
    global _engine_instance
    
    if _engine_instance is None:
        # Get paths from environment variables or use defaults
        model_path = os.getenv('MODEL_PATH', 'gpt-38M-scientific-pretrain-optimized')
        config_path = os.getenv('CONFIG_PATH', None)
        device = os.getenv('DEVICE', 'auto')
        tokenizer_encoding = os.getenv('TOKENIZER_ENCODING', 'gpt2')
        
        _engine_instance = InferenceEngine(
            model_path=model_path,
            config_path=config_path,
            device=device,
            tokenizer_encoding=tokenizer_encoding
        )
    
    return _engine_instance


def reset_engine():
    """Reset the engine instance (useful for testing)."""
    global _engine_instance
    _engine_instance = None
    get_engine.cache_clear()

