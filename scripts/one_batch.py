"""
FIXME: Explain the purpose of this test.
"""

import time

import jax
import numpy as np

from transformer.batches import get_bucketed_batches
from transformer.config import IWSLT_EXPERIMENT, ONE_BATCH_EXPERIMENT
from transformer.inputs import build_model, build_optimizer, load_data
from transformer.tokenizer import PAD_ID
from transformer.train import train_step

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

    batch = {"src": src[:BATCH_SIZE], "tgt": tgt[:BATCH_SIZE]}
    y_true = batch["tgt"][0][:-1]

    # Map English IDs to words to pretty translation.
    tgt_id_to_word = {i: w for w, i in data.tgt_vocab.items()}
    def print_sentence(x):
        print("\t" + " ".join([tgt_id_to_word[int(i)] for i in x if i != PAD_ID]))

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
