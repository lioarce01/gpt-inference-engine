"""Pydantic schemas for API request/response validation."""
import time
import uuid
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, validator


# OpenAI-compatible request schemas
class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request schema."""
    model: Optional[str] = Field(default="optimized-gpt", description="Model name (for compatibility)")
    prompt: str = Field(..., min_length=1, max_length=10000, description="Input prompt text")
    max_tokens: int = Field(default=100, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    stream: bool = Field(default=False, description="Stream response (not supported yet)")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        """Validate prompt is not just whitespace."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty or only whitespace")
        return v.strip()


class CompletionChoice(BaseModel):
    """Choice in completion response."""
    text: str = Field(..., description="Generated text")
    index: int = Field(default=0, description="Choice index")
    finish_reason: str = Field(default="stop", description="Finish reason")


class CompletionUsage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = Field(..., description="Number of tokens in prompt")
    completion_tokens: int = Field(..., description="Number of tokens in completion")
    total_tokens: int = Field(..., description="Total tokens")


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response schema."""
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:24]}", description="Response ID")
    object: str = Field(default="text_completion", description="Object type")
    created: int = Field(default_factory=lambda: int(time.time()), description="Creation timestamp")
    model: str = Field(default="optimized-gpt", description="Model name")
    choices: List[CompletionChoice] = Field(..., description="Completion choices")
    usage: CompletionUsage = Field(..., description="Token usage")


# OpenAI-compatible chat schemas
class ChatMessage(BaseModel):
    """Schema for a single chat message."""
    role: Literal["system", "user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., min_length=1, description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request schema."""
    model: Optional[str] = Field(default="optimized-gpt", description="Model name (for compatibility)")
    messages: List[ChatMessage] = Field(..., min_items=1, description="List of chat messages")
    max_tokens: int = Field(default=100, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    stream: bool = Field(default=False, description="Stream response (not supported yet)")
    
    @validator('messages')
    def validate_messages(cls, v):
        """Validate at least one message exists."""
        if not v:
            raise ValueError("At least one message is required")
        return v


class ChatCompletionMessage(BaseModel):
    """Message in chat completion choice."""
    role: str = Field(..., description="Message role")
    content: str = Field(..., description="Message content")


class ChatCompletionChoice(BaseModel):
    """Choice in chat completion response."""
    index: int = Field(default=0, description="Choice index")
    message: ChatCompletionMessage = Field(..., description="Message")
    finish_reason: str = Field(default="stop", description="Finish reason")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response schema."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}", description="Response ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(default_factory=lambda: int(time.time()), description="Creation timestamp")
    model: str = Field(default="optimized-gpt", description="Model name")
    choices: List[ChatCompletionChoice] = Field(..., description="Completion choices")
    usage: CompletionUsage = Field(..., description="Token usage")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="Service status")
    device: str = Field(..., description="Device where model is loaded")
    model_ready: bool = Field(..., description="Whether model is ready for inference")


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Additional error details")

