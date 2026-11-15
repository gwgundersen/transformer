"""
Train the original base transformer on the add3d dataset.

JAX_PLATFORMS=mps uv run python scripts/train_add3d.py"
"""

import argparse
from pathlib import Path

import jax
import numpy as np
from flax import nnx

from transformer import checkpoints as cp
from transformer.batches import get_batches
from transformer.config import load_experiment_config
from transformer.decode import greedy_generate
from transformer.inputs import build_model, build_optimizer, load_data
from transformer.metrics import log_metrics
from transformer.paths import CHECKPOINT_DIR
from transformer.tokenizer import get_print_seq
from transformer.train import train_step

# --------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suffix",
        default="",
        type=str,
        choices=["", "-zpad", "-zpad-oversample"]
    )
    args = parser.parse_args()
    print(f"{args.suffix=}")

    # Remove the log file, since we re-create it and then append to it
    try:
        Path.unlink(CHECKPOINT_DIR / f"add3d{args.suffix}.csv")
    except FileNotFoundError:
        pass

    # Load the experiment config and then model and optimizer.
    cfg = load_experiment_config(f"add3d{args.suffix}")
    print(cfg)

    data = load_data(cfg=cfg.data, decoder_only=True)
    model = build_model(cfg=cfg, vocabs=data.vocabs)
    optimizer = build_optimizer(cfg=cfg.train, data=data, model=model)
    print_seq = get_print_seq(data.vocabs["tgt"])

    # No resume functionality. These models are tiny
    step = 0
    epoch_start = 0

    losses = []
    for epoch in range(epoch_start, cfg.train.n_epochs):

        rng = np.random.default_rng([cfg.train.seed, epoch])

        # Bucket and shuffle batches
        batches = get_batches(
            {"seq": data.train.seq, "is_tgt": data.train.is_tgt},
            batch_size=cfg.train.batch_size,
            rng=rng,
        )

        losses_epoch = []
        for batch in batches:
            loss = train_step(model, optimizer, batch)
            losses.append(loss)
            losses_epoch.append(loss)
            step +=1

            if step % 250 == 0:
                print(f"{step}\t{float(loss):.4f}")

                # Evaluate training performance on held out data.
                eval_batch = list(get_batches(
                    {"seq": data.validation.seq, "is_tgt": data.validation.is_tgt},
                    batch_size=1,
                    rng=rng,
                ))[0]
                y_true = eval_batch["seq"][:1, :]
                y_hat = greedy_generate(
                    model,
                    seq=eval_batch["seq"][:1, :],
                    is_tgt=eval_batch["is_tgt"][:1, :]
                )
                print_seq(y_true)
                print_seq(y_hat)

        if epoch % cfg.train.metrics_freq == 0:
            print(f"{epoch=}\t{np.mean(losses_epoch):.4f}")
            log_metrics(cfg.name, step, loss, mode="a+")

        if epoch % cfg.train.checkpoint_freq == 0:
            # Each checkpoint is ~1GB, so overwrite previous checkpoint
            print(f"{epoch=}\tsaving checkpoint")
            cp.save_checkpoint(
                name=f"{cfg.name}-{epoch}",
                model=model,
                optimizer=optimizer,
                step=step,
                epoch=epoch
            )
