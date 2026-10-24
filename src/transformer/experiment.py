import numpy as np

from flax import nnx
import optax

from transformer.batches import assign_buckets
from transformer.config import DatasetConfig, Data, ModelConfig, TrainConfig
from transformer.dataset import DATASETS
from transformer.tokenizer import prepare
from transformer.model import Transformer
from transformer.vocab import load_vocab


def load_data(cfg: DatasetConfig) -> Data:
    """
    loads, tokenizes and caches the dataset, then returns the padded src/tgt
    """
    # Load 'raw' dataset.
    data = DATASETS[cfg.dataset]()

    # Load pre-computed vocabularies.
    vocabs = {lang: load_vocab(lang) for lang in (cfg.src_lang, cfg.tgt_lang)}

    # Preprocess training data.
    # FIXME: Make this train or test set?
    train_data = data["train"].map(
        prepare,
        fn_kwargs={"vocab": vocabs, "max_length": cfg.max_length},
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


def build_model(cfg: ModelConfig, vocabs: dict[str, dict[str, int]]) -> Transformer:
    """
    Build the transformer model using the config and vocabs.
    """
    return Transformer(
        src_vocab=vocabs["src"],
        tgt_vocab=vocabs["tgt"],
        model_dim=cfg.model_dim,
        ff_dim=cfg.ff_dim,
        n_heads=cfg.n_heads,
        n_modules=cfg.n_modules,
        dropout_rate=cfg.dropout_rate
    )


# FIXME: Enrich this.
def build_optimizer(cfg: TrainConfig, model: Transformer) -> nnx.Optimizer:
    """
    """
    return nnx.Optimizer(
        model,
        optax.adam(cfg.learning_rate),
        wrt=nnx.Param,
    )
