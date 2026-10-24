import jax.numpy as jnp

from transformer.model import Transformer
from transformer.tokenizer import BOS_ID, EOS_ID, make_tgt_mask

# --------------------------------------------------------------------------------------------------

def greedy_decode(
    model: Transformer,
    src: jnp.ndarray,
    src_mask: jnp.ndarray,
    max_length: int
) -> jnp.ndarray:
    """
    """
    model.eval()
    encoder_output = model.encode(src, src_mask)

    # Start generating output, one token at a time
    tgt = jnp.array([[BOS_ID]])

    for step in range(max_length - 1):

        tgt_mask = make_tgt_mask(tgt)
        
        decoder_output = model.decode(
            encoder_output,
            src_mask,
            tgt,
            tgt_mask
        )

        logits = model.output_projection(decoder_output)

        # logits has shape (1, current_seq_len, vocab_size); we want to take best word, i.e. argmax
        # over vocab size
        next_id = jnp.argmax(logits[0, -1, :])

        tgt = jnp.concatenate([tgt, next_id[None, None]], axis=1)

        if next_id == EOS_ID:
            break

    return tgt.squeeze()
