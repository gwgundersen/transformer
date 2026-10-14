from collections import Counter
from typing import Callable

import jax.numpy as jnp
import numpy as np
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer


# FIXME: Explain why we need BOS/EOS
BOS_WORD = "<s>"
EOS_WORD = "</s>"
BLANK_WORD = "<blank>"
UNK_WORD = "<unk>"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

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

        special = [PAD_ID, UNK_ID, BOS_ID, EOS_ID]
        filtered = [t for t, c in counts.items() if c >= 2 and t not in special]
        id_to_str = special + filtered
        str_to_id = {token: i for i, token in enumerate(id_to_str)}
        return str_to_id

    return _build_vocab


# FIXME: Better name
def _prepare(sentence: str, vocab: dict, tokenizer: Tokenizer, max_length: int) -> np.array:
    """
    """
    doc = tokenizer(sentence)
    ids = [vocab.get(t.text, UNK_ID) for t in doc]

    # Reserve two spots for BOS/EOS
    ids = ids[:max_length - 2]

    # Add BOS/EOS and then padding
    ids = [BOS_ID, *ids, EOS_ID]
    padding = [PAD_ID] * (max_length - len(ids))
    return np.array(ids + padding)


def prepare(x: dict, vocab: dict, max_length: int) -> dict:
    """
    """
    src = _prepare(
        x["translation"]["de"],
        vocab["de"],
        LANG_TO_TOKENIZER["de"],
        max_length,
    )

    # The source mask only needs to remove invalid tokens.
    # FIXME: Explain None, None, which is reshaping to add n_head and broadcasting across (size, size)
    # dims
    src_mask = (src != PAD_ID)[None, None, :]

    tgt = _prepare(
        x["translation"]["en"],
        vocab["en"],
        LANG_TO_TOKENIZER["en"],
        max_length,
    )

    # FIXME: Explain
    tgt_input = tgt[:-1]
    tgt_output = tgt[1:]

    tgt_mask = make_tgt_mask(tgt_input)

    return {
        "src": src,
        "src_mask": src_mask,
        "tgt_input": tgt_input,
        "tgt_mask": tgt_mask,
        "tgt_output": tgt_output,
    }


def make_tgt_mask(tgt_input: np.ndarray | jnp.ndarray) -> np.ndarray | jnp.ndarray:
    """
    """
    mask = (tgt_input != PAD_ID) & causal_mask(len(tgt_input))
    # FIXME: Add new axis to allow broadcasting over n_heads
    return mask[None, :, :]


def causal_mask(size: int) -> np.ndarray:
    """
    """
    # FIXME: Explain
    score_shape = (size, size)
    return np.tril(np.ones(score_shape), k=0).astype(bool)
