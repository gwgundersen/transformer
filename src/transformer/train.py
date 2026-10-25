import optax
from flax import nnx
from jax import numpy as jnp

from transformer.model import make_src_mask, make_tgt_mask
from transformer.tokenizer import PAD_ID

# --------------------------------------------------------------------------------------------------

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
