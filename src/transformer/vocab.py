import json
from collections import Counter

from transformer.paths import VOCAB_DIR

# --------------------------------------------------------------------------------------------------

BOS_WORD = "<s>"  # Give the model something to condition off-of to start the sequence
EOS_WORD = "</s>"  # Give the model something to generate in order to terminate the sequence
PAD_WORD = "<blank>"
UNK_WORD = "<unk>"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def load_vocab(name: str) -> dict[str, int]:
    """
    Return pre-computed vocabulary mapping words to IDs.
    """
    path = (VOCAB_DIR / name).with_suffix(".json")
    with open(path) as f:
        return json.load(f)


def build_vocab(sentences: list, min_freq: int = 2) -> dict[str, int]:
    """
    Return word->ID mapping for the embedding layer.
    """
    counts = Counter()
    for sent in sentences:
        for tok in sent:
            counts[tok] += 1

    special = [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD]
    filtered = [t for t, c in counts.items() if c >= min_freq and t not in special]
    id_to_word = special + filtered
    word_to_id = {t: i for i, t in enumerate(id_to_word)}

    # Verify we have built the vocabulary correctly
    for w, i in zip(special, [PAD_ID, UNK_ID, BOS_ID, EOS_ID]):
        assert word_to_id[w] == i
    assert min(word_to_id.values()) == 0
    assert max(word_to_id.values()) == len(word_to_id) - 1
    assert set(word_to_id.values()) == set(range(len(word_to_id)))

    return word_to_id
