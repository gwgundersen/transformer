"""
FIXME: Explain the purpose of this test.
"""

import json
import numpy as np
import time

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from transformer.config import IWSLTConfig
from transformer.dataset import load_iwslt_en_de
from transformer.decode import greedy_decode
from transformer.model import Transformer
from transformer.tokenizer import (
    build_vocab, prepare, make_causal_mask, make_tgt_mask, BOS_ID, EOS_ID, PAD_ID
)
from transformer.train import batches, train_step
from transformer.vocab import load_vocab


# --------------------------------------------------------------------------------------------------

BATCH_SIZE = 128
DECODE_EVERY = 50


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")

    data = load_iwslt_en_de()
    train_data = data["train"]

    vocab_de = load_vocab("de")
    vocab_en = load_vocab("en")
    en_id_to_word = {i: w for w, i in vocab_en.items()}

    config = IWSLTConfig()

    model = Transformer(
        src_vocab=vocab_de,
        tgt_vocab=vocab_en,
        dropout_rate=0.0
    )

    optimizer = nnx.Optimizer(
        model,
        optax.adam(1e-4),
        wrt=nnx.Param,
    )

    train_data = data["train"].map(
        prepare,
        fn_kwargs={"vocab": {"de": vocab_de, "en": vocab_en}, "max_length": config.max_length},
        remove_columns=data["train"].column_names,
        load_from_cache_file=True,
    )

    # HuggingFace's API will return a list. Convert to a block of numpy arrays.
    train_data = train_data.with_format("numpy")
    src = np.asarray(train_data["src"])
    tgt = np.asarray(train_data["tgt"])

    # for src, src_mask, tgt_input, tgt_mask, tgt_output in batches(train_data, batch_size=128):
    rng = np.random.default_rng(seed=0)
    batch = {"src": src[:BATCH_SIZE], "tgt": tgt[:BATCH_SIZE]}
    y_true = batch["tgt"][0][:-1]

    def print_sentence(x):
        print("\t" + " ".join([en_id_to_word[int(i)] for i in x if i != PAD_ID]))

    losses = []
    for step in range(1000):

        loss = train_step(model, optimizer, batch)
        losses.append(loss)

        if step % DECODE_EVERY == 0:

            # Force a sync between Python and GPU before computing the time
            jax.block_until_ready(loss)
            elapsed = time.perf_counter() - t0
            print(f"{step}\t{loss} ({elapsed=:.2f})")

            src = batch["src"]
            tgt = batch["tgt"]
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            src_mask = (src != PAD_ID)[:, None, None, :]
            size = tgt_input.shape[1]
            causal_mask = make_causal_mask(size)
            tgt_mask = (tgt_input != PAD_ID)[:, None, None, :] & causal_mask

            y_hat = greedy_decode(
                model,
                src=src[:1],
                src_mask=src_mask[:1],
                max_length=config.max_length
            )
            print_sentence(y_true)
            print_sentence(y_hat)
