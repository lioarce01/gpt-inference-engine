"""Inference engine for autoregressive language model generation."""
import os
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn

from .model_loader import ModelLoader, ModelLoadError
from .tokenizer import Tokenizer


class InferenceError(Exception):
    """Exception raised during inference."""
    pass


class InferenceEngine:
    """Engine for running inference on autoregressive language models."""
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = 'auto',
        tokenizer_encoding: str = "gpt2"
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model checkpoint file or directory containing model files
            config_path: Optional path to config directory (if different from model_path)
            device: Device to run on ('auto', 'cpu', or 'cuda')
            tokenizer_encoding: Tiktoken encoding name (default: "gpt2")
        """
        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model using ModelLoader, which handles CheckpointManager internally
        # ModelLoader will use CheckpointManager if available, otherwise fall back to direct loading
        model_dir = config_path if config_path else model_path
        
        if os.path.isdir(model_dir):
            # Load from directory (ModelLoader handles CheckpointManager internally)
            self.model, self.metadata = ModelLoader.load_model_from_dir(
                model_dir,
                device=self.device
            )
        else:
            # Load checkpoint file directly
            # First try to find directory containing the checkpoint
            checkpoint_dir = os.path.dirname(model_path) if os.path.isfile(model_path) else os.path.dirname(model_path)
            
            # Try loading as directory first (allows ModelLoader to use CheckpointManager)
            if os.path.isdir(checkpoint_dir):
                self.model, self.metadata = ModelLoader.load_model_from_dir(
                    checkpoint_dir,
                    device=self.device,
                    checkpoint_file=os.path.basename(model_path) if os.path.isfile(model_path) else None
                )
            else:
                # Fallback: load checkpoint file directly
                self.model, checkpoint_meta = ModelLoader.load_checkpoint(
                    model_path,
                    device=self.device
                )
                # Try to load configs from same directory
                configs = ModelLoader.load_configs(checkpoint_dir) if os.path.exists(checkpoint_dir) else {}
                self.metadata = {**configs, **checkpoint_meta}
        
        # Check if model is actually a state_dict (OrderedDict)
        # This should not happen if ModelLoader used CheckpointManager correctly
        from collections import OrderedDict
        if isinstance(self.model, (dict, OrderedDict)) and not isinstance(self.model, nn.Module):
            # Model is a state_dict, need to reconstruct
            raise InferenceError(
                "Model loaded as state_dict (OrderedDict) instead of model object. "
                "This usually means CheckpointManager was not available or failed to load. "
                "Please ensure:\n"
                "1. checkpoint.py is in the project root\n"
                "2. model.py and model_optimized.py are also in the project root\n"
                "3. The checkpoint format is correct\n"
                f"Current model_path: {model_path}\n"
                f"Model type: {type(self.model)}\n"
                "ModelLoader should have used CheckpointManager to reconstruct the model."
            )
        
        # Move model to device and set to eval mode
        if isinstance(self.model, nn.Module):
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Torch Compile (only on GPU - skip on CPU/Windows to avoid C++ compiler requirement)
            # torch.compile requires C++ compiler on CPU which causes issues on Windows
            if (hasattr(torch, 'compile') and 
                self.device == 'cuda' and 
                torch.cuda.is_available() and 
                'cuda' in str(next(self.model.parameters()).device)):
                try:
                    # Use 'reduce-overhead' mode which is more stable
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                except Exception:
                    # Silently continue without compilation if it fails
                    # Common reasons: CUDA version mismatch, etc.
                    pass
            
            # Half Precision FP16/BF16
            if self.device == 'cuda' and torch.cuda.is_available():
                # Prefer BF16 if available (better numerical stability)
                if torch.cuda.is_bf16_supported():
                    self.model = self.model.bfloat16()
                else:
                    self.model = self.model.half()
            
            # Store model forward reference
            self._model_forward = self.model
        elif not callable(self.model):
            raise InferenceError(
                f"Model is not callable. Got type: {type(self.model)}. "
                "Expected nn.Module or callable model object."
            )
        else:
            # For non-nn.Module callable models, store reference directly
            self._model_forward = self.model
        
        # Get model configuration
        self.model_args = self.metadata.get('model_args', {})
        self.config = self.metadata.get('config', {})
        
        # Get block_size (context window)
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'block_size'):
            self.block_size = self.model.config.block_size
        else:
            self.block_size = self.model_args.get('block_size', self.config.get('block_size', 256))
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(encoding_name=tokenizer_encoding)
        
        # Verify vocab size matches
        model_vocab_size = self.model_args.get('vocab_size', 50257)
        tokenizer_vocab_size = self.tokenizer.get_vocab_size()
        if model_vocab_size != tokenizer_vocab_size:
            # Warning but continue - might be intentional
            print(f"Warning: Model vocab size ({model_vocab_size}) != tokenizer vocab size ({tokenizer_vocab_size})")
    
    def _top_k_top_p_filter(
        self,
        logits: torch.Tensor,
        top_k: int,
        top_p: float
    ) -> torch.Tensor:
        """
        Apply top-k and top-p (nucleus) filtering to logits.
        
        Args:
            logits: Logits tensor of shape (batch, vocab_size)
            top_k: Keep only top k tokens (0 = disabled)
            top_p: Keep tokens with cumulative probability <= top_p (1.0 = disabled)
            
        Returns:
            Filtered logits
        """
        # Optimize top-k filtering
        if top_k > 0:
            # Use topk directly to get values and indices
            top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
            # Create filtered tensor directly
            filtered_logits = torch.full_like(logits, float("-inf"))
            filtered_logits.scatter_(1, top_k_indices, top_k_values)
            logits = filtered_logits
        
        if top_p < 1.0:
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            # Calculate cumulative probabilities
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            # Create mask for tokens exceeding top_p
            mask = cumulative_probs > top_p
            # Always keep the first token
            mask[..., 0] = False
            # Apply mask
            sorted_logits[mask] = float("-inf")
            # Reconstruct logits in original order
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(1, sorted_indices, sorted_logits)
        
        return logits
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (0 = disabled)
            top_p: Top-p (nucleus) sampling (1.0 = disabled)
            stop_sequences: Optional list of strings that stop generation when encountered
            
        Returns:
            Generated text (including prompt)
        """
        if not prompt:
            raise InferenceError("Prompt cannot be empty")
        
        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt)
        
        # Truncate to block_size if necessary
        prompt_ids = self.tokenizer.truncate_to_length(prompt_ids, self.block_size)
        
        # Convert to tensor
        x = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Pre-allocate tensor
        max_seq_len = len(prompt_ids) + max_tokens
        generated = torch.zeros((1, max_seq_len), dtype=torch.long, device=self.device)
        generated[0, :len(prompt_ids)] = x[0]
        current_pos = len(prompt_ids)
        
        # torch.inference_mode()
        with torch.inference_mode():
            for _ in range(max_tokens):
                # Eliminate unnecessary slice
                # Use indices directly instead of creating new tensor
                start_idx = max(0, current_pos - self.block_size)
                idx_cond = generated[:, start_idx:current_pos]
                
                # Use stored model reference
                output = self._model_forward(idx_cond)
                if isinstance(output, tuple):
                    logits, _ = output
                else:
                    logits = output
                
                # Get logits for last token
                logits = logits[:, -1, :] / max(temperature, 1e-6)
                
                # Apply top-k and top-p filtering
                logits = self._top_k_top_p_filter(logits, top_k, top_p)
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Pre-allocated tensor - assign directly
                generated[0, current_pos] = next_token[0, 0]
                current_pos += 1
                
                # Optimize stop sequences check
                if stop_sequences:
                    # Only check last N tokens (max stop sequence length + margin)
                    max_stop_len = max(len(seq) for seq in stop_sequences) if stop_sequences else 0
                    check_window = max_stop_len + 10  # Safety margin
                    
                    if current_pos >= check_window:
                        # Decode only last tokens for checking
                        check_tokens = generated[0, max(0, current_pos - check_window):current_pos].tolist()
                        check_text = self.tokenizer.decode(check_tokens)
                        
                        for stop_seq in stop_sequences:
                            if stop_seq in check_text:
                                # Decode full text only when match found
                                full_text = self.tokenizer.decode(generated[0, :current_pos].tolist())
                                stop_idx = full_text.rfind(stop_seq)
                                if stop_idx != -1:
                                    return full_text[:stop_idx]
                
                # Early stop if we hit EOS token (optional)
                if next_token.item() == self.tokenizer.eot_token:
                    break
        
        # Decode generated tokens (only up to current_pos)
        generated_text = self.tokenizer.decode(generated[0, :current_pos].tolist())
        
        return generated_text
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate response from chat messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            stop_sequences: Optional stop sequences
            
        Returns:
            Generated response text
        """
        # Format messages into prompt
        prompt = self._format_chat_prompt(messages)
        
        # Generate response
        full_response = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_sequences=stop_sequences
        )
        
        # Extract only the assistant's response (remove prompt)
        # This is a simple implementation - may need refinement based on model format
        if prompt in full_response:
            response = full_response[len(prompt):].strip()
        else:
            response = full_response
        
        return response
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages into a prompt string.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Formatted prompt string
        """
        formatted_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                formatted_parts.append(f"System: {content}")
            elif role == 'user':
                formatted_parts.append(f"User: {content}")
            elif role == 'assistant':
                formatted_parts.append(f"Assistant: {content}")
            else:
                formatted_parts.append(f"{role.capitalize()}: {content}")
        
        # Add prompt for assistant response
        formatted_parts.append("Assistant:")
        
        return "\n".join(formatted_parts)
    
    def is_ready(self) -> bool:
        """Check if engine is ready for inference."""
        return self.model is not None and (
            not isinstance(self.model, nn.Module) or 
            next(self.model.parameters()).device.type == self.device.split(':')[0]
        )

