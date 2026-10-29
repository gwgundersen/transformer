from pathlib import Path

from transformer.paths import CHECKPOINT_DIR

# --------------------------------------------------------------------------------------------------

def log_metrics(path: Path | str, step: int, loss: float) -> None:
    """
    Log training metrics alongside saved checkpoints.
    """
    path = CHECKPOINT_DIR / Path(path).with_suffix(".csv")
    with open(path, "a") as f:
        f.write(f"{step},{loss}\n")
