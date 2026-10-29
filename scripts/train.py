"""
Train the original base transformer on the IWSLT dataset.

JAX_PLATFORMS=mps uv run python scripts/train.py
"""

import argparse

import jax
import numpy as np
from flax import nnx

from transformer import checkpoints as cp
from transformer.batches import get_bucketed_batches
from transformer.config import load_experiment_config
from transformer.decode import greedy_decode
from transformer.inputs import build_model, build_optimizer, load_data
from transformer.metrics import log_metrics
from transformer.model import make_src_mask, make_tgt_mask
from transformer.tokenizer import get_print_sentence
from transformer.train import train_step

# --------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="iwslt")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    # Load the experiment config and then model and optimizer.
    cfg = load_experiment_config(args.experiment)
    data = load_data(cfg=cfg.data)
    model = build_model(cfg=cfg, vocabs={"src": data.src_vocab, "tgt": data.tgt_vocab})
    optimizer = build_optimizer(cfg=cfg.train, model=model)

    if args.resume:
        # FIXME: Make the checkpoint path part of the config?
        state = cp.load_checkpoint(
            path=cfg.name,
            model=model,
            optimizer=optimizer
        )
        nnx.update(model, state["model"])
        nnx.update(optimizer, state["optimizer"])
        # step and epoch are from the last successful cycle; so increment
        step = 1 + int(state["step"])
        epoch_prev = int(state["epoch"])
        epoch_start = 1 + epoch_prev
        print(f"Model loaded at epoch={epoch_prev}")
    else:
        step = 0
        epoch_start = 0

    print_sentence = get_print_sentence(data.tgt_vocab)

    losses = []
    for epoch in range(epoch_start, cfg.train.n_epochs):

        rng = np.random.default_rng([cfg.train.seed, epoch])

        # Bucket and shuffle batches
        batches = get_bucketed_batches(
            data.src,
            data.tgt,
            buckets=data.buckets,
            token_budget=cfg.data.token_budget,
            rng=rng,
        )

        for batch in batches:
            loss = train_step(model, optimizer, batch)
            losses.append(loss)
            step +=1

        if epoch % cfg.train.metrics_freq == 0:
            print(f"{epoch}\t{float(losses[-1]):.4f}")
            log_metrics(cfg.name, step, loss)

            tgt_input = batch["tgt"][:, :-1]
            src_mask = make_src_mask(batch["src"])
            tgt_mask = make_tgt_mask(tgt_input)

            y_true = tgt_input[0]
            y_hat = greedy_decode(
                model, 
                src=batch["src"][:1],
                src_mask=src_mask[:1],
                max_length=cfg.data.max_length,
            )
            print_sentence(y_true)
            print_sentence(y_hat)

        if epoch % cfg.train.checkpoint_freq == 0:
            # Each checkpoint is ~1GB, so overwrite previous checkpoint
            cp.save_checkpoint(
                path=cfg.name,
                model=model,
                optimizer=optimizer,
                step=step,
                epoch=epoch
            )
