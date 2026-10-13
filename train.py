import json
import numpy as np

from flax import nnx
import jax.numpy as jnp
import optax

from dataset import load_iwslt_en_de
from transformer import Transformer
from tokenizer import build_vocab, prepare, make_tgt_mask, BOS_ID, EOS_ID, PAD_ID


# --------------------------------------------------------------------------------------------------

MAX_LENGTH : int = 100


def batches(data, batch_size: int) -> list:
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
        return loss.mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss.item()


def greedy_decode(model, batch):
    """
    """
    model.eval()

    src = batch["src"][0][None, :]
    src_mask = batch["src_mask"][0][None, :]
    encoder_output = model.encode(src, src_mask)

    # Start generating output, one token at a time
    tgt = jnp.array([[BOS_ID]])

    for _ in range(MAX_LENGTH - 1):

        tgt_mask = make_tgt_mask(tgt)
        decoder_output = model.decode(
            encoder_output,
            src_mask,
            tgt,
            tgt_mask
        )
        logits = model.output_projection(decoder_output)
        next_id = np.argmax(logits)
        tgt = jnp.concatenate([tgt, next_id[None, None]], axis=1)

        if next_id == EOS_ID:
            break

    return tgt.squeeze()


if __name__ == "__main__":
     data = load_iwslt_en_de()
     train_data = data["train"].to_list()

     # FIXME: Could be behind simple loaders.
     with open("vocab_de.json") as f:
         vocab_de = json.load(f)
     with open("vocab_en.json") as f:
         vocab_en = json.load(f)

     model = Transformer(
         src_vocab=vocab_de,
         tgt_vocab=vocab_en
     )

     optimizer = nnx.Optimizer(
         model,
         optax.adam(1e-4),
         wrt=nnx.Param,
     )

     # FIXME: What is the max length?
     train_data = data["train"].map(
         prepare,
         fn_kwargs={"vocab": {"de": vocab_de, "en": vocab_en}, "max_length": MAX_LENGTH},
         remove_columns=data["train"].column_names,
         load_from_cache_file=True,
     )

     # FIXME: Explain.
     train_data = train_data.with_format("jax")
     
     # for src, src_mask, tgt_input, tgt_mask, tgt_output in batches(train_data, batch_size=128):

     batch = next(batches(train_data, batch_size=32))
     y_true = batch["src"][0]
     inds = y_true != PAD_ID

     losses = []
     for step in range(1000):
         loss = train_step(model, optimizer, batch)
         print(f"{step}\t{loss}")
         losses.append(loss)
         if step % 10 == 0:
             y_hat = greedy_decode(model, batch)
             print(y_true[inds])
             print(y_hat[inds])
