import numpy as np
import optax
from flax import nnx
from jax import numpy as jnp

from transformer.tokenizer import PAD_ID, make_src_mask, make_tgt_mask

# --------------------------------------------------------------------------------------------------

def batches(src: np.ndarray, tgt: np.ndarray, batch_size: int, rng: np.random.Generator) -> dict:
    """
    """
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


@nnx.jit
def train_step(model, optimizer, batch) -> jnp.array:
    """
    """
    model.train()

    src = batch["src"]
    tgt = batch["tgt"]

    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    # Build masks on the fly. This is cheap and jit-able.
    src_mask = make_src_mask(src)
    tgt_mask = make_tgt_mask(tgt_input)

    def loss_fn(model):
        logits = model(src, tgt_input, src_mask, tgt_mask)
        # FIXME: Explain teacher forcing
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, tgt_output)
        # FIXME: Explain predicting padding
        valid = tgt_output != PAD_ID
        return (loss * valid).sum() / valid.sum()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss
