# LLM Inference Engine

OpenAI-compatible REST API for serving autoregressive language models (GPT-style). A lightweight inference engine optimized for fast text generation and chat completions.

## What it is

A production-ready inference service that serves GPT-style language models via HTTP API with OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`).

## Tech Stack

- **PyTorch** - Model inference and tensor operations
- **FastAPI** - REST API framework
- **Uvicorn** - ASGI server
- **tiktoken** - Tokenization
- **OptimizedGPT** - Model architecture (RMSNorm, RoPE, GQA, SwiGLU)

## Optimizations

- Torch Compile (JIT compilation)
- Half-precision inference (FP16/BF16)
- Pre-allocated tensors
- Optimized sampling (top-k/top-p)
- Efficient stop sequence checking
