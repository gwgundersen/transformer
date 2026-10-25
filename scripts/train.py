"""
Train the original base transformer on the IWSLT dataset.

JAX_PLATFORMS=mps uv run python scripts/train.py
"""

import jax
import numpy as np

from transformer.batches import get_bucketed_batches
from transformer.checkpoints import save_checkpoint
from transformer.config import IWSLT_EXPERIMENT
from transformer.decode import greedy_decode
from transformer.inputs import build_model, build_optimizer, load_data
from transformer.model import make_src_mask, make_tgt_mask
from transformer.tokenizer import get_print_sentence
from transformer.train import train_step

# --------------------------------------------------------------------------------------------------

BATCH_SIZE = 128
SAVE_EVERY = 1


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")

    data = load_data(
        cfg=IWSLT_EXPERIMENT.data
    )
    model = build_model(
        cfg=IWSLT_EXPERIMENT,
        vocabs={"src": data.src_vocab, "tgt": data.tgt_vocab}
    )
    optimizer = build_optimizer(
        cfg=IWSLT_EXPERIMENT.train,
        model=model
    )
    print_sentence = get_print_sentence(data.tgt_vocab)

    rng = np.random.default_rng(seed=0)
    step = 0
    losses = []
    for epoch in range(1000):

        # Bucket and shuffle batches
        batches = get_bucketed_batches(
            data.src,
            data.tgt,
            buckets=data.buckets,
            token_budget=IWSLT_EXPERIMENT.data.token_budget,
            rng=rng,
        )

        for batch in batches:
            loss = train_step(model, optimizer, batch)
            losses.append(loss)
            step +=1 
        
        if epoch % SAVE_EVERY == 0:
            print(f"{epoch}\t{float(losses[-1]):.4f}")

            # Each checkpoint is ~1GB, so overwrite previous checkpoint
            save_checkpoint(
                "iwslt",
                model,
                optimizer,
                losses,
                step
            )

            tgt_input = batch["tgt"][:, :-1]
            src_mask = make_src_mask(batch["src"])
            tgt_mask = make_tgt_mask(tgt_input)

            y_true = tgt_input[0]
            y_hat = greedy_decode(
                model, 
                src=batch["src"][:1],
                src_mask=src_mask[:1],
                max_length=IWSLT_EXPERIMENT.data.max_length,
            )
            print_sentence(y_true)
            print_sentence(y_hat)
