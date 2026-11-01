from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
DATASET_DIR = PROJECT_ROOT / "artifacts" / "datasets"
VOCAB_DIR = PROJECT_ROOT / "artifacts" / "vocab"
