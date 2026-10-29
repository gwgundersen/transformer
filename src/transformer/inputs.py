from dataclasses import dataclass

import numpy as np
import optax
from flax import nnx

from transformer.batches import assign_buckets
from transformer.config import DatasetConfig, ExperimentConfig, TrainConfig
from transformer.dataset import DATASETS
from transformer.model import Transformer
from transformer.tokenizer import encode_example
from transformer.vocab import load_vocab

# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Data:
    src: np.ndarray
    tgt: np.ndarray
    buckets: np.ndarray
    src_vocab: dict[str, int]
    tgt_vocab: dict[str, int]


def load_data(cfg: DatasetConfig) -> Data:
    """
    Load dataset and vocabularies, encode the sentences (from strings to tokens to IDs), compute
    bucket assignments for bucketed batching, and return bundle of metadata.
    """
    # Load 'raw' dataset.
    data = DATASETS[cfg.dataset]()

    # Load pre-computed vocabularies.
    vocabs = {lang: load_vocab(lang) for lang in (cfg.src_lang, cfg.tgt_lang)}

    # Preprocess training data.
    # FIXME: Make this train or test set?
    train_data = data["train"].map(
        encode_example,
        fn_kwargs={
            "src_lang": cfg.src_lang,
            "tgt_lang": cfg.tgt_lang,
            "vocab": vocabs,
            "max_length": cfg.max_length
        },
        remove_columns=data["train"].column_names,
        load_from_cache_file=True,
    )

    # HuggingFace's API will return a list. Convert to numpy arrays.
    train_data = train_data.with_format("numpy")
    src_data = np.asarray(train_data["src"])
    tgt_data = np.asarray(train_data["tgt"])

    # FIXME. Explain what this is doing here.
    buckets = assign_buckets(src_data, tgt_data, cfg.bucket_boundaries)

    return Data(
        src=src_data,
        tgt=tgt_data,
        buckets=buckets,
        src_vocab=vocabs[cfg.src_lang],
        tgt_vocab=vocabs[cfg.tgt_lang],
    )


def build_model(cfg: ExperimentConfig, vocabs: dict[str, dict[str, int]]) -> Transformer:
    """
    Build the transformer model using the config and vocabs.
    """
    return Transformer(
        src_vocab_size=len(vocabs["src"]),
        tgt_vocab_size=len(vocabs["tgt"]),
        model_dim=cfg.model.model_dim,
        ff_dim=cfg.model.ff_dim,
        n_heads=cfg.model.n_heads,
        n_modules=cfg.model.n_modules,
        dropout_rate=cfg.model.dropout_rate,
        rngs=nnx.Rngs(cfg.train.seed)
    )


# FIXME: Allow a richer config, e.g. rate scheduling.
def build_optimizer(cfg: TrainConfig, model: Transformer) -> nnx.Optimizer:
    """
    Build optimizer.
    """
    return nnx.Optimizer(
        model,
        optax.adam(cfg.learning_rate),
        wrt=nnx.Param,
    )
