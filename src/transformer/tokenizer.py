from collections.abc import Callable

import numpy as np
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer

from transformer.vocab import BOS_ID, EOS_ID, PAD_ID, UNK_ID

# --------------------------------------------------------------------------------------------------

LANG_TO_TOKENIZER = {
    "en": English().tokenizer,
    "de": German().tokenizer
}


def _encode(sentence: str, vocab: dict, tokenizer: Tokenizer, max_length: int) -> np.array:
    """
    Encode a single sentence using a vocab, tokenizer, and padded to a max length.
    """
    doc = tokenizer(sentence)
    ids = [vocab.get(t.text, UNK_ID) for t in doc]

    # Reserve two spots for BOS/EOS
    ids = ids[:max_length - 2]

    # Add BOS/EOS and then padding
    ids = [BOS_ID, *ids, EOS_ID]
    padding = [PAD_ID] * (max_length - len(ids))

    # Use int32. Using current tokenization, the German vocab has 61,841 IDs. Note:
    #   2^16 = 65,536
    #   2^32 = 4,294,967,296
    # So while int16 would fit this vocabularly, only just. If vocabulary grew, int16 would
    # fail silently by wrapping tokens
    return np.array(ids + padding, dtype=np.int32)


def encode_example(
    pair: dict,
    src_lang: str,
    tgt_lang: str,
    vocab: dict[str, dict[str, int]],
    max_length: int
) -> dict:
    """
    Encode a translation pair (src and tgt languages).
    """
    return {
        "src": _encode(
            pair["translation"][src_lang],
            vocab[src_lang],
            LANG_TO_TOKENIZER[src_lang],
            max_length,
        ),
        "tgt": _encode(
            pair["translation"][tgt_lang],
            vocab[tgt_lang],
            LANG_TO_TOKENIZER[tgt_lang],
            max_length,
        )
    }


def get_print_sentence(vocab: dict[str, int]) -> Callable:
    """
    Return a function which decode a sentence and then prints it in vocab's language.
    """
    id_to_word = {i: w for w, i in vocab.items()}

    def print_sentence(x):
        print("\t" + " ".join([id_to_word[int(i)] for i in x if i != PAD_ID]))

    return print_sentence
