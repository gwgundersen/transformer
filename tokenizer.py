from collections import Counter
from typing import Callable

import jax.numpy as jnp
import numpy as np
from spacy.lang.de import German
from spacy.lang.en import English


BOS_WORD = "<s>"
EOS_WORD = "</s>"
BLANK_WORD = "<blank>"
UNK_WORD = "<unk>"
PADDING_ID = 0

LANG_TO_TOKENIZER = {
    "en": English().tokenizer,
    "de": German().tokenizer
}


def build_vocab(lang: str) -> Callable:

    def _build_vocab(data: list) -> dict:
        counts = Counter()
        tokenizer = LANG_TO_TOKENIZER[lang]
        for example in data:
            sentence = example["translation"][lang]
            counts.update([t.text for t in tokenizer(sentence)])
        filtered = [token for token, count in counts.items() if count >= 2]
        idx_to_str = [UNK_WORD, BLANK_WORD] + filtered
        # +1 to leave id=0 for PADDING
        str_to_idx = {token: i+1 for i, token in enumerate(idx_to_str)}
        return str_to_idx

    return _build_vocab


def build_prepare(vocab: dict, lang: str, max_length=10) -> Callable:

    def _prepare(pair: dict) -> jnp.array:
        tokenizer = LANG_TO_TOKENIZER[lang]
        doc = tokenizer(pair["translation"][lang])
        ids = [vocab[t.text] for t in doc]
        doc_length = len(doc)
        if doc_length > max_length:
            doc_padded = ids[:max_length]
        else:
            padding = [PADDING_ID] * (max_length - doc_length)
            doc_padded = ids + padding
        doc_padded = jnp.array(doc_padded)
        mask = doc_padded != PADDING_ID
        return doc_padded[None, :], mask[None, :]

    return _prepare
