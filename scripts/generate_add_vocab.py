import json

from transformer.dataset import generate_add_dataset
from transformer.paths import VOCAB_DIR
from transformer.vocab import build_vocab


# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    sentences = generate_add_dataset()
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    vocab = build_vocab(sentences)
    with open(VOCAB_DIR / "add.json", "w") as f:
        json.dump(vocab, f)
    # 16 words: 10 digits, +/=, and BOS/EOS/UNK/PAD
    print(f"Vocab `add` generated: {len(vocab)} words")
