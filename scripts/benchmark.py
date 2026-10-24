"""
Results on Apple M4 Pro with 24GB:

% JAX_PLATFORMS=cpu uv run python scripts/benchmark.py 
  JAX devices: [CpuDevice(id=0)]
  Compile time: 4.106 sec
  Average seconds per step: 2.928
  Average tokens per second: 897.369

% JAX_PLATFORMS=mps uv run python scripts/benchmark.py 
  JAX devices: [MpsDevice(id=0)]
  Compile time: 1.507 sec
  Average seconds per step: 0.654
  Average tokens per second: 4014.057

% JAX_PLATFORMS=mps JAX_MPS_ASYNC_DISPATCH=1 uv run python scripts/benchmark.py
  JAX devices: [MpsDevice(id=0)]
  Compile time: 1.452 sec
  Average seconds per step: 0.628
  Average tokens per second: 4182.534
"""

import json
import numpy as np

from flax import nnx
import jax
import jax.numpy as jnp
import optax
import time

from transformer.checkpoints import save_checkpoint
from transformer.config import IWSLTConfig
from transformer.dataset import load_iwslt_en_de
from transformer.decode import greedy_decode
from transformer.model import Transformer
from transformer.tokenizer import (
    build_vocab, prepare, make_src_mask, make_tgt_mask, BOS_ID, EOS_ID, PAD_ID
)
from transformer.train import batches, train_step
from transformer.vocab import load_vocab


# --------------------------------------------------------------------------------------------------

WARMUP_STEPS = 5
BENCHMARK_STEPS = 100
BATCH_SIZE = 128


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")

    data = load_iwslt_en_de()
    train_data = data["train"]

    vocab_de = load_vocab("de")
    vocab_en = load_vocab("en")
    en_id_to_word = {i: w for w, i in vocab_en.items()}

    config = IWSLTConfig()

    model = Transformer(
        src_vocab=vocab_de,
        tgt_vocab=vocab_en,
        dropout_rate=0.0
    )

    optimizer = nnx.Optimizer(
        model,
        optax.adam(1e-4),
        wrt=nnx.Param,
    )

    train_data = data["train"].map(
        prepare,
        fn_kwargs={"vocab": {"de": vocab_de, "en": vocab_en}, "max_length": config.max_length},
        remove_columns=data["train"].column_names,
        load_from_cache_file=True,
    )

    # HuggingFace's API will return a list. Convert to a block of numpy arrays.
    train_data = train_data.with_format("numpy")
    src_data = np.asarray(train_data["src"])
    tgt_data = np.asarray(train_data["tgt"])

    # Pre-materialize each batch.
    rng = np.random.default_rng(seed=0)
    benchmark_batches = []
    for _ in range(WARMUP_STEPS + BENCHMARK_STEPS):
        batch = next(batches(src_data, tgt_data, batch_size=BATCH_SIZE, rng=rng))
        benchmark_batches.append(batch)

    # Phase 1: Performance test JAX compile time
    t0 = time.perf_counter()
    loss = train_step(model, optimizer, benchmark_batches[0])
    jax.block_until_ready(loss)
    compile_sec = time.perf_counter() - t0
    print(f"Compile time: {compile_sec:.3f} sec")

    # Phase 2: Run the warmup for WARMUP_STEPS-1 steps. Allow memory allocaiton and caches, etc.
    # to settle a bit (hand-wavy, but I think this is what is happening).
    for i in range(1, WARMUP_STEPS):
        train_step(model, optimizer, benchmark_batches[i])

    # Phase 3: Measure the train_step; don't move anything off device, e.g. float(loss)
    t0 = time.perf_counter()
    n_tokens = 0
    for i in range(WARMUP_STEPS, WARMUP_STEPS + BENCHMARK_STEPS):
        batch = benchmark_batches[i]
        train_step(model, optimizer, batch)
        n_tokens += (batch["src"] != PAD_ID).sum()
        
    jax.block_until_ready((loss, nnx.state(model)))
    elapsed_sec = time.perf_counter() - t0
    seconds_per_step = elapsed_sec / BENCHMARK_STEPS
    tokens_per_sec = n_tokens / elapsed_sec
    
    print(f"Average seconds per step: {seconds_per_step:.3f}")
    print(f"Average tokens per second: {tokens_per_sec:.3f}")
