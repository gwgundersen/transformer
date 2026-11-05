from collections.abc import Callable

import numpy as np
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer

from transformer.vocab import BOS_ID, EOS_ID, PAD_ID, UNK_ID

# --------------------------------------------------------------------------------------------------

TOKENIZERS = {
    "de": lambda s: [t.text for t in German().tokenizer(t)],
    "en": lambda s: [t.text for t in English().tokenizer(t)],
    "char": list
}


def _encode(sentence: str, vocab: dict, tokenizer: Tokenizer, max_length: int) -> np.array:
    """
    Encode a single sentence using a vocab, tokenizer, and padded to a max length.
    """
    doc = tokenizer(sentence)
    ids = [vocab.get(t, UNK_ID) for t in doc]

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
    row: dict,
    vocab: dict[str, dict[str, int]],
    tokenizers: dict[str, str],
    max_length: int,
) -> dict:
    """
    Encode a translation pair (src and tgt languages).
    """
    return {
        x: _encode(row[x], vocab[x], TOKENIZERS[tokenizers[x]], max_length)
        for x in ("src", "tgt")
    }


def get_print_seq(vocab: dict[str, int]) -> Callable:
    """
    Return a function which decodes a sequence and then prints it.
    """
    id_to_word = {i: w for w, i in vocab.items()}

    def print_seq(x):
        x = x.squeeze()
        print("\t" + " ".join([id_to_word[int(i)] for i in x if i != PAD_ID]))

    return print_seq
