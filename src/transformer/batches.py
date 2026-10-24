import numpy as np
import optax
from flax import nnx
from jax import numpy as jnp

from transformer.tokenizer import PAD_ID, make_src_mask, make_tgt_mask

# --------------------------------------------------------------------------------------------------

# FIXME: Configure?
BUCKET_BOUNDARIES = [16, 24, 32, 48, 64, 128]
# FIXME: How do people choose this?
TOKEN_BUDGET = 4096
BATCH_SIZES = {l: TOKEN_BUDGET // l for l in BUCKET_BOUNDARIES}


def get_batches(
    src: np.ndarray,
    tgt: np.ndarray,
    batch_size: int,
    rng: np.random.Generator
) -> dict:
    """
    """
    batches = []
    perm = rng.permutation(len(src))
    for start in range(0, len(src), batch_size):
        inds = perm[start:start + batch_size]
        # Only return batches with the same size:
        # 1. Avoids an extra JAX compile
        # 2. Loss computation is over same number of samples
        if len(inds) < batch_size:
            return
        yield {
            "src": src[inds],
            "tgt": tgt[inds]
        }


def get_bucketed_batches(
    src: np.ndarray,
    tgt: np.ndarray,
    buckets: np.ndarray,
    token_budget: int,
    rng: np.random.Generator
) -> list:
    """
    """
    # For each bucket, shuffle the available indices and build the batches
    batches = []
    for l in np.unique(buckets):
        # Convert boolean mask of bucket membership into actual indices into src/tgt
        inds = np.flatnonzero(buckets == l)
        rng.shuffle(inds)
        batch_size = token_budget // l
        for start in range(0, len(inds), batch_size):
            sel = inds[start:start + batch_size]
            if len(sel) == batch_size:
                batches.append((l, sel))

    # Shuffle the final batches
    perm = rng.permutation(len(batches))
    for i in perm:
        l, inds = batches[i]
        yield {
            "src": src[inds, :l],
            "tgt": tgt[inds, :l]
        }


def assign_buckets(src_data: np.ndarray, tgt_data: np.ndarray, bucket_boundaries: tuple) -> np.ndarray:
    """
    """
    n = len(src_data)
    assert n == len(tgt_data)
    out = []
    for i in range(n):
        src = src_data[i][src_data[i] != PAD_ID]
        tgt = tgt_data[i][tgt_data[i] != PAD_ID]
        b = int(np.searchsorted(bucket_boundaries, max(len(src), len(tgt)), side="left"))
        b = min(b, len(bucket_boundaries)-1)
        l = bucket_boundaries[b]
        out.append(l)
    return np.array(out)
