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


# FIXME: Better name?
"""
<s> 1 2 3 + 4 5 6 =         --> 5
<s> 1 2 3 + 4 5 6 = 5       --> 7
<s> 1 2 3 + 4 5 6 = 5 7     --> 9
<s> 1 2 3 + 4 5 6 = 5 7 9   --> </s>
"""
def greedy_generate(
    model: DecoderOnlyTransformer,
    seq: jnp.ndarray,
    is_tgt: jnp.ndarray,
    vocab: dict[int, str],
) -> jnp.ndarray:
    """
    """
    # FIXME: This only works for batch_size == 1

    model.eval()
    print_seq = get_print_seq(vocab)

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
        next_id = jnp.argmax(logits[0, -1, :])
        # add next word
        src_working[:, i] = next_id

        # debugging
        # print_seq(src_working)

        if next_id == EOS_ID:
            break

    return src_working


"""
  def _forward(model, seq):
      return model(seq, make_tgt_mask(seq))


  def greedy_generate(model: DecoderOnlyTransformer, prompts: np.ndarray) -> np.ndarray:
      model.eval()
      seq = prompts.copy()
      rows = np.arange(len(seq))
      lengths = (seq != PAD_ID).sum(axis=1)
      done = lengths >= seq.shape[1]
      while not done.all():
          logits = _forward(model, seq)
          next_ids = np.asarray(logits[rows, lengths - 1].argmax(-1))
          active = ~done
          seq[rows[active], lengths[active]] = next_ids[active]
          lengths[active] += 1
          done |= (next_ids == EOS_ID) | (lengths >= seq.shape[1])
      return seq
"""
