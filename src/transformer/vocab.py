import json
from pathlib import Path


# --------------------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VOCAB_DIR = PROJECT_ROOT / "artifacts" / "vocab"


# FIXME: `load_vocab` should work for a path as well. For now, accept either "en" or "de".
def load_vocab(lang: str) -> dict[str, int]:
    """
    Return pre-computed vocabulary mapping words to IDs.
    """
    path = VOCAB_DIR / f"{lang}.json"
    with open(path) as f:
        return json.load(f)
