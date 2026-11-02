from dataclasses import dataclass

import numpy as np
import optax
from datasets import DatasetDict
from flax import nnx

from transformer.batches import assign_buckets
from transformer.config import (
    DatasetConfig,
    DecoderOnlyConfig,
    EncoderDecoderConfig,
    ExperimentConfig,
    TrainConfig,
)
from transformer.dataset import DATASETS
from transformer.model import DecoderOnlyTransformer, EncoderDecoderTransformer, Model
from transformer.tokenizer import encode_example
from transformer.vocab import load_vocab

# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Split:
    src: np.ndarray
    tgt: np.ndarray
    buckets: np.ndarray


@dataclass(frozen=True)
class Data:
    train: Split
    validation: Split
    test: Split
    vocabs: dict[str, dict[str, int]]


def _load_split(
    ds: DatasetDict,
    cfg: DatasetConfig,
    vocabs: dict[str, dict[str, int]],
    split: str
) -> Split:
    """
    Load a single split (train, validation test) from a given HF dataset dict.
    """
    split_data = ds[split].map(
        encode_example,
        fn_kwargs={
            "vocab": vocabs,
            "tokenizers":  {
                "src": cfg.src_tokenizer,
                "tgt": cfg.tgt_tokenizer
            },
            "max_length": cfg.max_length,
        },
        remove_columns=ds[split].column_names,
        load_from_cache_file=True,
    )

    # HuggingFace's API will return a list. Convert to numpy arrays.
    split_data = split_data.with_format("numpy")
    src = np.asarray(split_data["src"])
    tgt = np.asarray(split_data["tgt"])

    # Assign each sample to a "bucket" so we can group batches based on sequence length to reduce
    # padding. 
    if cfg.bucket_boundaries:
        buckets = assign_buckets(src, tgt, cfg.bucket_boundaries)
    else:
        buckets = None

    return Split(
        src=src,
        tgt=tgt,
        buckets=buckets,
    )


def load_data(cfg: DatasetConfig) -> Data:
    """
    Load dataset and vocabularies, encode the sentences (from strings to tokens to IDs), compute
    bucket assignments for bucketed batching, and return bundle of metadata.
    """
    # Load 'raw' dataset.
    name = cfg.dataset
    data = DATASETS[name]()

    # Load pre-computed vocabularies.
    vocabs = {
        "src": load_vocab(cfg.src_vocab),
        "tgt": load_vocab(cfg.tgt_vocab)
    }

    # Load and process each dataset split.
    splits = {
        s: _load_split(data, cfg, vocabs, s)
        for s in ("train", "validation", "test")
    }
    return Data(**splits, vocabs=vocabs)


def build_model(cfg: ExperimentConfig, vocabs: dict[str, dict[str, int]]) -> Model:
    """
    Build the transformer model using the config and vocabs.
    """
    if isinstance(cfg.model, EncoderDecoderConfig):
        return EncoderDecoderTransformer(
            src_vocab_size=len(vocabs["src"]),
            tgt_vocab_size=len(vocabs["tgt"]),
            model_dim=cfg.model.model_dim,
            ff_dim=cfg.model.ff_dim,
            n_heads=cfg.model.n_heads,
            n_modules=cfg.model.n_modules,
            dropout_rate=cfg.model.dropout_rate,
            rngs=nnx.Rngs(cfg.train.seed)
        )
    elif isinstance(cfg.model, DecoderOnlyConfig):
        return DecoderOnlyTransformer(
            # src and tgt share a vocabulary in decoder-only experiments
            vocab_size=len(vocabs["src"]),
            model_dim=cfg.model.model_dim,
            ff_dim=cfg.model.ff_dim,
            n_heads=cfg.model.n_heads,
            n_modules=cfg.model.n_modules,
            dropout_rate=cfg.model.dropout_rate,
            rngs=nnx.Rngs(cfg.train.seed)
        )
    else:
        raise ValueError(f"Unknown config type {type(cfg.model)}")


# FIXME: Allow a richer config, e.g. rate scheduling.
def build_optimizer(cfg: TrainConfig, model: Model) -> nnx.Optimizer:
    """
    Build optimizer.
    """
    return nnx.Optimizer(
        model,
        optax.adam(cfg.learning_rate),
        wrt=nnx.Param,
    )
