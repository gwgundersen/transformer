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

import numpy as np
import jax.numpy as jnp
from dataset import load_iwslt_en_de
from transformer import Transformer
from tokenizer import build_vocab, build_prepare


BOS_WORD = "<s>"
EOS_WORD = "</s>"
BLANK_WORD = "<blank>"
PADDING_ID = 0


# --------------------------------------------------------------------------------------------------

data = load_iwslt_en_de()

VOCAB_DE = build_vocab("de")(data["train"])
VOCAB_EN = build_vocab("en")(data["train"])

prepare_de = build_prepare(VOCAB_DE, "de")
prepare_en = build_prepare(VOCAB_EN, "en")

model = Transformer(
    src_vocab=VOCAB_DE,
    tgt_vocab=VOCAB_EN
)

model.train()

n_train = len(data["train"])
for x in data["train"]:
    src, src_mask = prepare_de(x)
    tgt, tgt_mask = prepare_en(x)
    model.forward(src, tgt, src_mask, tgt_mask)
