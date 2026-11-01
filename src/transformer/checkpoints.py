from pathlib import Path
import shutil

from flax import nnx
from jax import numpy as jnp
from orbax.checkpoint import v1 as ocp

from transformer.model import Model
from transformer.paths import CHECKPOINT_DIR

# --------------------------------------------------------------------------------------------------

def save_checkpoint(
    name: str | Path,
    model: Model,
    optimizer: nnx.Optimizer,
    step: int,
    epoch: int,
) -> None:
    """
    Save model and optimizer state at training step `step`.
    """
    path = CHECKPOINT_DIR / name
    backup = CHECKPOINT_DIR / "backup"

    # If a checkpoint already exists, cache it in case the next save fails
    if path.exists():
        if backup.exists():
            shutil.rmtree(backup)
        path.rename(backup)

    state = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "step": jnp.asarray(step),
        "epoch": jnp.asarray(epoch)
    }

    # Blocking save. Async is faster, but then we have to handle edge cases
    ocp.save(path, state)


def load_checkpoint(
    path: str | Path,
    model: Model,
    optimizer: nnx.Optimizer,
) -> dict:
    """
    Load model and optimizer state.
    """
    path = CHECKPOINT_DIR / path
    # We need to tell orbax the data layout
    state = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "step": jnp.asarray(0),
        "epoch": jnp.asarray(0),
    }
    return ocp.load(path, abstract_state=state)
