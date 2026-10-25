import json

from transformer.dataset import load_iwslt2017_en_de
from transformer.paths import VOCAB_DIR
from transformer.tokenizer import LANG_TO_TOKENIZER
from transformer.vocab import build_vocab

# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Only generate the vocab using the training data, as using the validation/testing data leaks
    # information
    data = load_iwslt2017_en_de()
    for lang in ("de", "en"):
        tokenize = LANG_TO_TOKENIZER[lang]
        tokens = [tokenize(pair["translation"][lang]) for pair in data["train"]]
        vocab = build_vocab(tokens)
        with open(VOCAB_DIR / "en.json", "w") as f:
            json.dump(vocab_en, f)
        print(f"Vocab {lang} generated")
