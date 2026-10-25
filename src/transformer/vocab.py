import json
from collections.abc import Callable

from transformer.paths import VOCAB_DIR
from transformer.tokenizer import LANG_TO_TOKENIZER

# --------------------------------------------------------------------------------------------------

# FIXME: Explain why we need BOS/EOS
PAD_WORD = "<blank>"
UNK_WORD = "<unk>"
BOS_WORD = "<s>"
EOS_WORD = "</s>"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


# FIXME: `load_vocab` should work for a path as well. For now, accept either "en" or "de".
def load_vocab(lang: str) -> dict[str, int]:
    """
    Return pre-computed vocabulary mapping words to IDs.
    """
    path = VOCAB_DIR / f"{lang}.json"
    with open(path) as f:
        return json.load(f)


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
