"""
FIXME: Explain the purpose of this test.
"""

import time

import jax
import numpy as np
import optax
from flax import nnx

from transformer.config import ONE_BATCH_EXPERIMENT
from transformer.batches import get_bucketed_batches
from transformer.model import Transformer
from transformer.tokenizer import PAD_ID, prepare
from transformer.train import train_step
from transformer.experiment import load_data, build_model, build_optimizer

# --------------------------------------------------------------------------------------------------

BATCH_SIZE = 128
DECODE_EVERY = 50


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")

    data = load_data(ONE_BATCH_EXPERIMENT.data)
    model = build_model(
        IWSLT_EXPERIMENT.model,
        vocabs={"src": data.src_vocab, "tgt": data.tgt_vocab}
    )
    optimizer = build_optimizer(ONE_BATCH_EXPERIMENT.train, model)

    rng = np.random.default_rng(seed=0)
    batches = get_bucketed_batches(
        data.src,
        data.tgt,
        buckets=data.buckets,
        token_budget=IWSLT_EXPERIMENT.data.token_budget,
        rng=rng,
    )

    # for src, src_mask, tgt_input, tgt_mask, tgt_output in batches(train_data, batch_size=128):
    rng = np.random.default_rng(seed=0)
    batch = {"src": src[:BATCH_SIZE], "tgt": tgt[:BATCH_SIZE]}
    y_true = batch["tgt"][0][:-1]

    def print_sentence(x):
        print("\t" + " ".join([en_id_to_word[int(i)] for i in x if i != PAD_ID]))

    losses = []
    for epoch in range(ONE_BATCH_EXPERIMENT.train.n_epochs):

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
