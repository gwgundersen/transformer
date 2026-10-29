import optax
from flax import nnx
from jax import Array, numpy as jnp
from jax.typing import ArrayLike

from transformer.model import Transformer, make_src_mask, make_tgt_mask
from transformer.tokenizer import PAD_ID

# --------------------------------------------------------------------------------------------------

@nnx.jit
def train_step(model: Transformer, optimizer: nnx.Optimizer, batch: dict[str, ArrayLike]) -> Array:
    """
    Jitted training step.
    """
    model.train()

    def loss_fn(model):
        loss_sum, n_tokens = compute_loss(model, batch["src"], batch["tgt"])
        return loss_sum / n_tokens

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def compute_loss(model: Transformer, src: ArrayLike, tgt: ArrayLike) -> tuple[Array, Array]:
    """
    """
    # FIXME: Explain teacher forcing.
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    # Build masks on the fly. This is cheap and jit-able.
    src_mask = make_src_mask(src)
    tgt_mask = make_tgt_mask(tgt_input)

    # Compute raw logits.
    logits = model(src, tgt_input, src_mask, tgt_mask)

    # Compute cross-entropy loss.
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, tgt_output)

    # FIXME: Explain predicting padding
    valid = tgt_output != PAD_ID
    return (loss * valid).sum(), valid.sum()
