import optax
from flax import nnx
from jax import Array
from jax.typing import ArrayLike

from transformer.model import (
    DecoderOnlyTransformer,
    EncoderDecoderTransformer,
    Model,
    make_src_mask,
    make_tgt_mask,
)
from transformer.tokenizer import PAD_ID

# --------------------------------------------------------------------------------------------------

@nnx.jit
def train_step(model: Model, optimizer: nnx.Optimizer, batch: dict[str, ArrayLike]) -> Array:
    """
    Jitted training step.
    """
    model.train()

    def loss_fn(model):
        loss_sum, n_tokens = compute_loss(model, batch)
        return loss_sum / n_tokens

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def compute_loss(model: Model, batch: dict[str, ArrayLike]) -> tuple[Array, Array]:
    """
    Wrapper loss function for both encoder-decoders and decoder-only models.
    """
    if isinstance(model, DecoderOnlyTransformer):
        return _decoder_only_loss(model, batch["seq"], batch["is_tgt"])
    elif isinstance(model, EncoderDecoderTransformer):
        return _encoder_decoder_loss(model, batch["src"], batch["tgt"])


def _encoder_decoder_loss(
    model: EncoderDecoderTransformer,
    src: ArrayLike,
    tgt: ArrayLike
) -> tuple[Array, Array]:
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

    # Compute cross-entropy loss:
    #
    #   logits shape     == (batch, seq, vocab)
    #   tgt_output shape == (batch, seq)
    #   loss shape       == (batch, seq)
    #
    # This loss function applies log softmax to the vocab scores, turning them into log probs
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, tgt_output)

    # Don't count mistakes where there's just padding
    valid = tgt_output != PAD_ID
    return (loss * valid).sum(), valid.sum()


def _decoder_only_loss(
    model: DecoderOnlyTransformer,
    seq: ArrayLike,
    is_tgt: ArrayLike
) -> tuple[Array, Array]:
    """
    """
    seq_input = seq[:, :-1]
    seq_output = seq[:, 1:]
    seq_mask = make_tgt_mask(seq_input)

    # Compute raw logits.
    logits = model(seq_input, seq_mask)

    # Compute cross-entropy loss.
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, seq_output)

    # Remove BOS because unneeded and aligns with seq_output shape
    valid = is_tgt[:, 1:]
    return (loss * valid).sum(), valid.sum()
