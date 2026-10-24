from collections import Counter
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer

# FIXME: Explain why we need BOS/EOS
PAD_WORD = "<blank>"
UNK_WORD = "<unk>"
BOS_WORD = "<s>"
EOS_WORD = "</s>"

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
    Return partial function to build a word->ID mapping.
    """

    def _build_vocab(data: list) -> dict:
        """
        Return word->ID mapping for the embedding layer.
        """
        counts = Counter()
        tokenizer = LANG_TO_TOKENIZER[lang]
        for example in data:
            sentence = example["translation"][lang]
            counts.update([t.text for t in tokenizer(sentence)])

        special = [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD]
        filtered = [t for t, c in counts.items() if c >= 2 and t not in special]
        id_to_word = special + filtered
        word_to_id = {t: i for i, t in enumerate(id_to_word)}

        # Verify we have built the vocabulary correctly
        for w, i in zip(special, [PAD_ID, UNK_ID, BOS_ID, EOS_ID]):
            assert word_to_id[w] == i
        assert min(word_to_id.values()) == 0
        assert max(word_to_id.values()) == len(word_to_id) - 1
        assert set(word_to_id.values()) == set(range(len(word_to_id)))

        return word_to_id

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

    # Use int32. Using current tokenization, the German vocab has 61,841 IDs. Note:
    #   2^16 = 65,536
    #   2^32 = 4,294,967,296
    # So while int16 would fit this vocabularly, only just. If vocabulary grew, int16 would
    # fail silently by wrapping tokens
    return np.array(ids + padding, dtype=np.int32)


def prepare(x: dict, vocab: dict, max_length: int) -> dict:
    """
    """
    src = _prepare(
        x["translation"]["de"],
        vocab["de"],
        LANG_TO_TOKENIZER["de"],
        max_length,
    )
    tgt = _prepare(
        x["translation"]["en"],
        vocab["en"],
        LANG_TO_TOKENIZER["en"],
        max_length,
    )
    return {
        "src": src,
        "tgt": tgt
    }


def make_src_mask(src):
    """
    """
    # Attention has shape (B, H, Q, K); add dimensions so that src mask works.
    return (src != PAD_ID)[:, None, None, :]


def make_tgt_mask(tgt_input: np.ndarray | jnp.ndarray) -> np.ndarray | jnp.ndarray:
    """
    """
    # Add new axis to allow broadcasting over n_heads
    causal_mask = make_causal_mask(tgt_input.shape[1])
    return (tgt_input != PAD_ID)[:, None, None, :] & causal_mask


def make_causal_mask(size: int) -> np.ndarray:
    """
    Build the mask that enforces causality when decoding.
    """
    score_shape = (size, size)
    return np.tril(np.ones(score_shape), k=0).astype(bool)
