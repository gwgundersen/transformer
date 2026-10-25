#!/bin/bash

# Run on CPU
JAX_PLATFORMS=cpu uv run python scripts/benchmark.py
echo "========================================================"

# Run on GPU
JAX_PLATFORMS=mps uv run python scripts/benchmark.py
echo "========================================================"

# Run on GPU with bucketed batches
JAX_PLATFORMS=mps uv run python scripts/benchmark.py --bucketing
echo "========================================================"

# Run on GPU with bucketed batches and aync dispatch
JAX_PLATFORMS=mps JAX_MPS_ASYNC_DISPATCH=1 uv run python scripts/benchmark.py --bucketing
