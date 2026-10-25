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
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    for lang in ("de", "en"):
        tokenize = LANG_TO_TOKENIZER[lang]
        docs = [tokenize(pair["translation"][lang]) for pair in data["train"]]
        # Convert spacy.tokens.doc.Doc into strings
        sentences = [t.text for t in docs]
        vocab = build_vocab(sentences)
        with open(VOCAB_DIR / f"{lang}.json", "w") as f:
            json.dump(vocab, f)
        print(f"Vocab {lang} generated: {len(vocab)} words")
