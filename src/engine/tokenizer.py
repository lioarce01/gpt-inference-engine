"""Tokenizer wrapper for model tokenization."""
from typing import List, Optional
import tiktoken


class Tokenizer:
    """Wrapper for tokenization using tiktoken."""
    
    def __init__(self, encoding_name: str = "gpt2"):
        """
        Initialize tokenizer.
        
        Args:
            encoding_name: Name of tiktoken encoding (default: "gpt2")
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token IDs
        """
        return self.encoding.encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        return self.encoding.decode(token_ids)
    
    
    @property
    def eot_token(self) -> int:
        """End of text token ID."""
        return self.encoding.eot_token
    
    @property
    def bos_token(self) -> Optional[int]:
        """Beginning of text token ID (if available)."""
        # GPT-2 doesn't have explicit BOS token, but we can use eot_token
        return self.encoding.eot_token
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.encoding.n_vocab
    
    def truncate_to_length(self, token_ids: List[int], max_length: int) -> List[int]:
        """
        Truncate token IDs to a maximum length, keeping the last N tokens.
        
        Args:
            token_ids: List of token IDs to truncate
            max_length: Maximum length to keep
            
        Returns:
            Truncated list of token IDs (last max_length tokens)
        """
        if len(token_ids) <= max_length:
            return token_ids
        return token_ids[-max_length:]

