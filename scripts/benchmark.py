"""
Results on Apple M4 Pro with 24GB:

% JAX_PLATFORMS=cpu uv run python scripts/benchmark.py 
  WITHOUT LENGTH BUCKETING
  JAX devices: [CpuDevice(id=0)]
  Compile time: 4.106 sec
  Average seconds per step: 2.928
  Average tokens per second: 897.369

  WITH LENGTH BUCKETING
  JAX devices: [CpuDevice(id=0)]
  Compile time: 3.312 sec
  Average seconds per step: 1.864
  Average tokens per second: 1717.700

% JAX_PLATFORMS=mps uv run python scripts/benchmark.py 
    WITHOUT BUCKETING
        JAX devices: [MpsDevice(id=0)]
        Compile time: 1.507 sec
        Average seconds per step: 0.654
        Average tokens per second: 4014.057

    WITH BUCKETING
        JAX devices: [MpsDevice(id=0)]
        Compile time: 1.375 sec
        Average seconds per step: 0.459
        Average tokens per second: 6973.377

% JAX_PLATFORMS=mps JAX_MPS_ASYNC_DISPATCH=1 uv run python scripts/benchmark.py
  JAX devices: [MpsDevice(id=0)]
  Compile time: 1.452 sec
  Average seconds per step: 0.628
  Average tokens per second: 4182.534
"""

import time

import jax
import numpy as np
import optax
from flax import nnx

from transformer.config import IWSLT_EXPERIMENT
from transformer.batches import get_bucketed_batches
from transformer.model import Transformer
from transformer.tokenizer import PAD_ID, prepare
from transformer.train import train_step
from transformer.experiment import load_data, build_model, build_optimizer

# --------------------------------------------------------------------------------------------------

WARMUP_STEPS = 5
BENCHMARK_STEPS = 100
DO_LENGTH_BUCKETING = True


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")

    data = load_data(IWSLT_EXPERIMENT.data)
    model = build_model(
        IWSLT_EXPERIMENT.model,
        vocabs={"src": data.src_vocab, "tgt": data.tgt_vocab}
    )
    optimizer = build_optimizer(IWSLT_EXPERIMENT.train, model)

    rng = np.random.default_rng(seed=0)
    if DO_LENGTH_BUCKETING:
        batches = get_bucketed_batches(
            data.src,
            data.tgt,
            buckets=data.buckets,
            token_budget=IWSLT_EXPERIMENT.data.token_budget,
            rng=rng,
        )
    else:
        batches = get_batches(
            data.src,
            data.tgt,
            batch_size=data.batch_size,
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
    for l in np.unique(data.buckets):
        import ipdb; ipdb.set_trace()
        train_step(model, optimizer, next(batches))

    # Phase 3: Measure the train_step; don't move anything off device, e.g. float(loss)
    t0 = time.perf_counter()
    n_tokens = 0
    for i in range(WARMUP_STEPS, WARMUP_STEPS + BENCHMARK_STEPS):
        batch = next(batches)
        train_step(model, optimizer, batch)
        n_tokens += (batch["src"] != PAD_ID).sum()
        
    jax.block_until_ready((loss, nnx.state(model)))
    elapsed_sec = time.perf_counter() - t0
    seconds_per_step = elapsed_sec / BENCHMARK_STEPS
    tokens_per_sec = n_tokens / elapsed_sec
    
    print(f"Average seconds per step: {seconds_per_step:.3f}")
    print(f"Average tokens per second: {tokens_per_sec:.3f}")
