from collections import Counter
from typing import Callable

import jax.numpy as jnp
import numpy as np
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer


BOS_WORD = "<s>"
EOS_WORD = "</s>"
BLANK_WORD = "<blank>"
UNK_WORD = "<unk>"
UNK_ID = 1
PADDING_ID = 0

LANG_TO_TOKENIZER = {
    "en": English().tokenizer,
    "de": German().tokenizer
}


def build_vocab(lang: str) -> Callable:
    """
    """

    def _build_vocab(data: list) -> dict:
        """
        """
        counts = Counter()
        tokenizer = LANG_TO_TOKENIZER[lang]
        for example in data:
            sentence = example["translation"][lang]
            counts.update([t.text for t in tokenizer(sentence)])
        filtered = [token for token, count in counts.items() if count >= 2]
        id_to_str = [UNK_WORD, BLANK_WORD] + filtered
        # +1 to leave id=0 for PADDING
        str_to_id = {token: i+1 for i, token in enumerate(id_to_str)}
        assert str_to_id[UNK_WORD] == UNK_ID
        return str_to_id

    return _build_vocab


def _prepare(
    sentence: str, vocab: dict, tokenizer: Tokenizer, max_length: int
) -> dict[str, np.array]:
    """
    """
    doc = tokenizer(sentence)
    ids = [vocab.get(t.text, UNK_ID) for t in doc]
    doc_length = len(doc)
    if doc_length > max_length:
        doc_padded = ids[:max_length]
    else:
        padding = [PADDING_ID] * (max_length - doc_length)
        doc_padded = ids + padding
    doc_padded = np.array(doc_padded)
    mask = doc_padded != PADDING_ID
    return {
        "doc": doc_padded,
        "mask": mask
    }


def prepare(x: dict, vocab: dict, max_length: int) -> dict:
    """
    """
    de = _prepare(
        x["translation"]["de"],
        vocab["de"],
        LANG_TO_TOKENIZER["de"],
        max_length,
    )
    en = _prepare(
        x["translation"]["en"],
        vocab["en"],
        LANG_TO_TOKENIZER["en"],
        max_length,
    )
    return {
        "src": de["doc"],
        "src_mask": de["mask"],
        "tgt": en["doc"],
        "tgt_mask": en["mask"],
    }
