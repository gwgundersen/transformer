"""
TODO:
- implement positional encoding
- understand nnx.Embed --- is this not just nnx.Linear?
- implement src_embed / tgt_embed
- implement multi-head attention
- get data to flow end-to-end
- implement batching
- build training loop
- debug + make fast
"""

import json
import numpy as np

from flax import nnx
import jax.numpy as jnp
import optax

from dataset import load_iwslt_en_de
from transformer import Transformer
from tokenizer import build_vocab, prepare # build_prepare, prepare


# --------------------------------------------------------------------------------------------------

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

train_data = data["train"].map(
    prepare,
    fn_kwargs={"vocab": {"de": vocab_de, "en": vocab_en}, "max_length": 100},
    remove_columns=data["train"].column_names,
)

# FIXME: Explain.
train_data = train_data.with_format("jax")


def batches(data, batch_size: int) -> list:
    """
    """
    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        yield batch["src"], batch["src_mask"], batch["tgt"], batch["tgt_mask"]
        

model.train()
for src, src_mask, tgt, tgt_mask in batches(train_data, batch_size=128):

    def loss_fn(model):
        logits = model(
            src,
            tgt,
            src_mask,
            tgt_mask,
        )
        # FIXME: Explain teacher forcing
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1],
            tgt[:, 1:],
        )
        return loss.mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    print(f"{loss.item()}")
