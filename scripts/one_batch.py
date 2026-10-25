"""
Overfit the transformer to a single batch, to verify training.
"""

import time

import jax
import numpy as np

from transformer.batches import get_batches
from transformer.config import ONE_BATCH_EXPERIMENT
from transformer.decode import greedy_decode
from transformer.inputs import build_model, build_optimizer, load_data
from transformer.model import make_src_mask, make_tgt_mask
from transformer.tokenizer import get_print_sentence
from transformer.train import train_step

# --------------------------------------------------------------------------------------------------

BATCH_SIZE = 128
DECODE_EVERY = 10


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")

    data = load_data(
        cfg=ONE_BATCH_EXPERIMENT.data
    )
    model = build_model(
        cfg=ONE_BATCH_EXPERIMENT,
        vocabs={"src": data.src_vocab, "tgt": data.tgt_vocab}
    )
    optimizer = build_optimizer(
        cfg=ONE_BATCH_EXPERIMENT.train,
        model=model
    )

    rng = np.random.default_rng(seed=0)
    batches = get_batches(
        data.src,
        data.tgt,
        batch_size=BATCH_SIZE,
        rng=rng,
    )

    batch = {"src": data.src[:BATCH_SIZE], "tgt": data.tgt[:BATCH_SIZE]}
    y_true = batch["tgt"][0][:-1]

    # Print sentences in target language (English for IWSLT)
    print_sentence = get_print_sentence(data.tgt_vocab)

    losses = []
    for step in range(200):

        t0 = time.perf_counter()
        loss = train_step(model, optimizer, batch)
        losses.append(loss)

        if step % DECODE_EVERY == 0:

            # Force a sync between Python and GPU before computing the time.
            jax.block_until_ready(loss)
            elapsed = time.perf_counter() - t0
            print(f"{step}\t{loss} ({elapsed=:.2f})")

            src = batch["src"]
            tgt = batch["tgt"]
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            src_mask = make_src_mask(src)
            tgt_mask = make_tgt_mask(tgt_input)

            y_hat = greedy_decode(
                model,
                src=src[:1],
                src_mask=src_mask[:1],
                max_length=ONE_BATCH_EXPERIMENT.data.max_length
            )
            print_sentence(y_true)
            print_sentence(y_hat)
