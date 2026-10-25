"""
Benchmark training the original base transformer. Run with benchmark.sh to run:
1. CPU vs GPU
2. Bucketed batches vs not
3. jax-mps experimental async flag

Results on Apple M4 Pro with 24GB
---------------------------------

% JAX_PLATFORMS=cpu uv run python scripts/benchmark.py 
    Average seconds per step: 2.928
    Average tokens per second: 897.369

% JAX_PLATFORMS=mps uv run python scripts/benchmark.py 
    Average seconds per step: 0.654
    Average tokens per second: 4014.057

% JAX_PLATFORMS=mps uv run python scripts/benchmark.py --bucketing
    Average seconds per step: 0.459
    Average tokens per second: 6973.377

% JAX_PLATFORMS=mps JAX_MPS_ASYNC_DISPATCH=1 uv run python scripts/benchmark.py
    Average seconds per step: 0.628
    Average tokens per second: 4182.534
"""

import argparse
import time

import jax
import numpy as np
from flax import nnx
from transformer.experiment import build_model, build_optimizer, load_data

from transformer.batches import get_batches, get_bucketed_batches
from transformer.config import IWSLT_EXPERIMENT
from transformer.tokenizer import PAD_ID
from transformer.train import train_step

# --------------------------------------------------------------------------------------------------

WARMUP_STEPS = 5
BENCHMARK_STEPS = 100


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--bucketing", action="store_true")
    args = parser.parse_args()

    data = load_data(
        cfg=IWSLT_EXPERIMENT.data
    )
    model = build_model(
        cfg=IWSLT_EXPERIMENT,
        vocabs={"src": data.src_vocab, "tgt": data.tgt_vocab}
    )
    optimizer = build_optimizer(
        cfg=IWSLT_EXPERIMENT.train,
        model=model
    )

    rng = np.random.default_rng(seed=0)
    if args.bucketing:
        print("Getting bucketed batches")
        batches = get_bucketed_batches(
            data.src,
            data.tgt,
            buckets=data.buckets,
            token_budget=IWSLT_EXPERIMENT.data.token_budget,
            rng=rng,
        )
    else:
        print("Getting batches (no bucketing)")
        batches = get_batches(
            data.src,
            data.tgt,
            batch_size=IWSLT_EXPERIMENT.train.batch_size,
            rng=rng,
        )

    # Pre-materialize the batches
    batches = list(batches)

    # Phase 1: Performance test JAX compile time
    t0 = time.perf_counter()
    loss = train_step(model, optimizer, batches[0])
    jax.block_until_ready(loss)
    compile_sec = time.perf_counter() - t0
    print(f"Compile time: {compile_sec:.3f} sec")

    # Phase 2: Run the warmup for WARMUP_STEPS-1 steps. Allow memory allocaiton and caches, etc.
    # to settle a bit (hand-wavy, but I think this is what is happening).
    if args.bucketing:
        for l in IWSLT_EXPERIMENT.data.bucket_boundaries:
            # Obviously slow.
            batch = next(b for b in batches if b["src"].shape[1] == l)
            train_step(model, optimizer, batch)
    else:
        for i in range(1, WARMUP_STEPS):
            train_step(model, optimizer, batches[i])
    print("Warm up complete")

    # Phase 3: Measure the train_step; don't move anything off device, e.g. float(loss)
    t0 = time.perf_counter()
    n_tokens = 0
    for i in range(WARMUP_STEPS, WARMUP_STEPS + BENCHMARK_STEPS):
        batch = batches[i]
        train_step(model, optimizer, batch)
        n_tokens += (batch["src"] != PAD_ID).sum()
        
    jax.block_until_ready((loss, nnx.state(model)))
    elapsed_sec = time.perf_counter() - t0
    seconds_per_step = elapsed_sec / BENCHMARK_STEPS
    tokens_per_sec = n_tokens / elapsed_sec
    
    print(f"Seconds per step: {seconds_per_step:.3f}")
    print(f"Tokens per second: {tokens_per_sec:.3f}")
