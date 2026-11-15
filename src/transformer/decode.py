import jax.numpy as jnp
import numpy as np

from transformer.model import (
    DecoderOnlyTransformer,
    EncoderDecoderTransformer,
    make_tgt_mask,
)
from transformer.tokenizer import BOS_ID, EOS_ID, PAD_ID, get_print_seq

# --------------------------------------------------------------------------------------------------

def greedy_decode(
    model: EncoderDecoderTransformer,
    src: jnp.ndarray,
    src_mask: jnp.ndarray,
    max_length: int
) -> jnp.ndarray:
    """
    Encode a source sequence and then greedily (argmax on the logits) decode into a target
    sequence.
    """
    model.eval()
    encoder_output = model.encode(src, src_mask)

    # Start generating output, one token at a time
    tgt = jnp.array([[BOS_ID]])

    for step in range(max_length - 1):
        tgt_mask = make_tgt_mask(tgt)
        decoder_output = model.decode(encoder_output, src_mask, tgt, tgt_mask)
        logits = model.output_projection(decoder_output)

        # logits has shape (1, current_seq_len, vocab_size); we want to take best word, i.e. argmax
        # over vocab size. The argmax here is what makes this function "greedy".
        next_id = jnp.argmax(logits[0, -1, :])
        tgt = jnp.concatenate([tgt, next_id[None, None]], axis=1)

        if next_id == EOS_ID:
            break

    return tgt.squeeze()


# FIXME: This only works for batch_size == 1
def greedy_generate(
    model: DecoderOnlyTransformer,
    seq: jnp.ndarray,
    is_tgt: jnp.ndarray,
) -> jnp.ndarray:
    """
    Given a sequence so far ('src'), greedily generate the remainder ('tgt').
    """
    model.eval()

    # `seq` is the full sequence:
    #     <s> 1 2 3 + 4 5 6 = 5 7 9 </s>
    # we want to remove the answer:
    #     <s> 1 2 3 + 4 5 6 =
    # and see if the model can generate:
    #                         5 7 9 </s>
    src = np.where(is_tgt, PAD_ID, seq)
    src_working = src.copy()

    length = (src_working != PAD_ID).sum(axis=1).item()

    for i in range(length, src_working.shape[1]):

        # logits is shape (1, seq_len, vocab_size)
        logits = model.decode(src_working, make_tgt_mask(src_working))
        # pick the max logit, i.e. the next word
        next_id = jnp.argmax(logits[0, i - 1, :])
        # add next word
        src_working[:, i] = next_id

        if next_id == EOS_ID:
            break

    return src_working
