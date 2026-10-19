from datasets import Dataset
from flax import nnx
import optax

from transformer.tokenizer import build_vocab, prepare, make_tgt_mask, BOS_ID, EOS_ID, PAD_ID


# --------------------------------------------------------------------------------------------------

def batches(data: Dataset, batch_size: int) -> Dataset:
    """
    """
    for start in range(0, len(data), batch_size):
        yield data[start:start + batch_size]
        

def train_step(model, optimizer, batch):
    """
    """
    model.train()

    src = batch["src"]
    src_mask = batch["src_mask"]
    tgt_input = batch["tgt_input"]
    tgt_mask = batch["tgt_mask"]
    tgt_output = batch["tgt_output"]

    def loss_fn(model):
        logits = model(src, tgt_input, src_mask, tgt_mask)
        # FIXME: Explain teacher forcing
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, tgt_output)
        # FIXME: Explain predicting padding
        valid = tgt_output != PAD_ID
        return (loss * valid).sum() / valid.sum()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss.item()
