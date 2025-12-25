"""FastAPI application for LLM inference API."""
import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    CompletionUsage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionMessage,
    HealthResponse,
    ErrorResponse
)
from .dependencies import get_engine
from src.engine.inference_engine import InferenceEngine, InferenceError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LLM Inference API",
    description="API for autoregressive language model inference",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool executor for running inference in background threads
executor = ThreadPoolExecutor(max_workers=4)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "LLM Inference API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(engine: InferenceEngine = Depends(get_engine)):
    """
    Health check endpoint.
    
    Returns status of the service and model.
    """
    try:
        is_ready = engine.is_ready()
        return HealthResponse(
            status="ready" if is_ready else "not_ready",
            device=engine.device,
            model_ready=is_ready
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            device="unknown",
            model_ready=False
        )


# OpenAI-compatible endpoints
@app.post("/v1/completions", response_model=CompletionResponse, tags=["OpenAI Compatible"])
async def completions(
    request: CompletionRequest,
    engine: InferenceEngine = Depends(get_engine)
):
    """
    OpenAI-compatible text completion endpoint.
    
    This endpoint follows OpenAI's API structure for text completions.
    """
    start_time = time.time()
    
    try:
        # Check if engine is ready
        if not engine.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not ready for inference"
            )
        
        # Calculate prompt tokens
        prompt_tokens = len(engine.tokenizer.encode(request.prompt))
        
        # Run inference in thread pool
        loop = asyncio.get_event_loop()
        generated_text = await loop.run_in_executor(
            executor,
            engine.generate,
            request.prompt,
            request.max_tokens,
            request.temperature,
            0,  # top_k not supported in OpenAI API
            request.top_p if request.top_p < 1.0 else 1.0,
            request.stop
        )
        
        # Extract only the generated part (without prompt)
        completion_text = generated_text[len(request.prompt):].strip()
        completion_tokens = len(engine.tokenizer.encode(completion_text))
        total_tokens = prompt_tokens + completion_tokens
        
        generation_time = time.time() - start_time
        
        logger.info(
            f"Completion: {completion_tokens} tokens in {generation_time:.2f}s "
            f"(prompt: {request.prompt[:50]}...)"
        )
        
        return CompletionResponse(
            model=request.model or "optimized-gpt",
            choices=[
                CompletionChoice(
                    text=completion_text,
                    index=0,
                    finish_reason="stop"
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )
        
    except InferenceError as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during completion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during text completion"
        )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, tags=["OpenAI Compatible"])
async def chat_completions(
    request: ChatCompletionRequest,
    engine: InferenceEngine = Depends(get_engine)
):
    """
    OpenAI-compatible chat completion endpoint.
    
    This endpoint follows OpenAI's API structure for chat completions.
    """
    start_time = time.time()
    
    try:
        # Check if engine is ready
        if not engine.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not ready for inference"
            )
        
        # Convert messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Calculate prompt tokens (approximate)
        prompt_text = " ".join([msg["content"] for msg in messages])
        prompt_tokens = len(engine.tokenizer.encode(prompt_text))
        
        # Run inference in thread pool
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(
            executor,
            engine.chat,
            messages,
            request.max_tokens,
            request.temperature,
            0,  # top_k not supported in OpenAI API
            request.top_p if request.top_p < 1.0 else 1.0,
            request.stop
        )
        
        completion_tokens = len(engine.tokenizer.encode(response_text))
        total_tokens = prompt_tokens + completion_tokens
        
        generation_time = time.time() - start_time
        
        logger.info(
            f"Chat completion: {completion_tokens} tokens in {generation_time:.2f}s "
            f"({len(messages)} messages)"
        )
        
        return ChatCompletionResponse(
            model=request.model or "optimized-gpt",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=response_text
                    ),
                    finish_reason="stop"
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )
        
    except InferenceError as e:
        logger.error(f"Inference error in chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during chat completion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during chat completion"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=None
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if logger.level == logging.DEBUG else None
        ).dict()
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down...")
    executor.shutdown(wait=True)

