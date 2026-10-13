import json

from dataset import load_iwslt_en_de
from tokenizer import build_vocab


# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Only generate the vocab using the training data, as using the validation/testing data leaks
    # information
    data = load_iwslt_en_de()
    vocab_de = build_vocab("de")(data["train"])
    vocab_en = build_vocab("en")(data["train"])
    with open("vocab_de.json", "w") as f:
        json.dump(vocab_de, f, ensure_ascii=False, indent=2)
    with open("vocab_en.json", "w") as f:
        json.dump(vocab_en, f, ensure_ascii=False, indent=2)
