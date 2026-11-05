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
from transformer.tokenizer import BOS_ID, EOS_ID, PAD_ID, encode_example
from transformer.vocab import load_vocab

# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Split:
    src: np.ndarray
    tgt: np.ndarray
    buckets: np.ndarray | None
    seq: np.ndarray | None
    is_tgt: np.ndarray | None


@dataclass(frozen=True)
class Data:
    train: Split
    validation: Split
    test: Split
    vocabs: dict[str, dict[str, int]]


def load_data(cfg: DatasetConfig, decoder_only: bool) -> Data:
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
        split: _load_split(data, cfg, vocabs, decoder_only, split)
        for split in ("train", "validation", "test")
    }
    return Data(**splits, vocabs=vocabs)


def _load_split(
    ds: DatasetDict,
    cfg: DatasetConfig,
    vocabs: dict[str, dict[str, int]],
    decoder_only: bool,
    split: str,
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

    if decoder_only:
        seq, is_tgt = _join_src_tgt(src, tgt, cfg.max_length)
    else:
        seq, is_tgt = None, None

    return Split(
        src=src,
        tgt=tgt,
        seq=seq,
        is_tgt=is_tgt,
        buckets=buckets,
    )


def _join_src_tgt(
    src: np.ndarray,
    tgt: np.ndarray,
    max_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Join encoded src/tgt pairs into a single sequence for decoder-only training.
    """
    # return seq and is_tgt
    assert src.shape[1] == max_length
    assert tgt.shape[1] == max_length

    seqs = np.full(src.shape, PAD_ID)
    masks = np.full(src.shape, False)
    n_samples = len(src)

    for i in range(n_samples):

        # Remove padding and end-of-seq ID
        s = src[i][src[i] != PAD_ID]
        assert s[-1] == EOS_ID
        s = s[:-1]

        # Remove padding and start-of-seq ID
        t = tgt[i][tgt[i] != PAD_ID]
        assert s[0] == BOS_ID
        t = t[1:]

        seq = np.concatenate([s, t])
        assert seq[0] == BOS_ID
        assert seq[-1] == EOS_ID

        seqs[i, :len(seq)] = seq
        # Mask-in only the target sequence. Used in computing the loss.
        masks[i, len(s):len(seq)] = True

    return seqs, masks


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


# FIXME: Allow a richer config, e.g. with learning rate scheduling.
def build_optimizer(cfg: TrainConfig, model: Model) -> nnx.Optimizer:
    """
    Build optimizer.
    """
    return nnx.Optimizer(
        model,
        optax.adam(cfg.learning_rate),
        wrt=nnx.Param,
    )
