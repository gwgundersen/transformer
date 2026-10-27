from pathlib import Path

from flax import nnx
from jax import numpy as jnp
from orbax.checkpoint import v1 as ocp

from transformer.model import Transformer
from transformer.paths import CHECKPOINT_DIR

# --------------------------------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    model: Transformer,
    optimizer: nnx.Optimizer,
    step: int,
    epoch: int,
) -> None:
    """
    Save model and optimizer state at training step `step`.
    """
    path = CHECKPOINT_DIR / path
    state = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "step": jnp.asarray(step),
        "epoch": jnp.asarray(epoch)
    }
    ocp.save_async(path, state, overwrite=True)


def load_checkpoint(path: str | Path) -> dict:
    """
    Load model and optimizer state.
    """
    path = CHECKPOINT_DIR / path
    return ocp.load(path)
