import json
from pathlib import Path

from transformer.dataset import load_iwslt_en_de
from transformer.tokenizer import build_vocab

# --------------------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOCAB_DIR = PROJECT_ROOT / "artifacts" / "vocab"


if __name__ == "__main__":
    # Only generate the vocab using the training data, as using the validation/testing data leaks
    # information
    data = load_iwslt_en_de()
    vocab_de = build_vocab("de")(data["train"])
    vocab_en = build_vocab("en")(data["train"])
    with open(VOCAB_DIR / "de.json", "w") as f:
        json.dump(vocab_de, f)
    with open(VOCAB_DIR / "en.json", "w") as f:
        json.dump(vocab_en, f)
    print("Vocabs generated")
