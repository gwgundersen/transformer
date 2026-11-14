from dataclasses import dataclass


# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    dataset: str
    src_vocab: str
    tgt_vocab: str
    src_tokenizer: str | None
    tgt_tokenizer: str | None
    max_length: int
    bucket_boundaries: tuple[int, ...] | None
    token_budget: int


@dataclass(frozen=True)
class EncoderDecoderConfig:
    model_dim: int
    ff_dim: int
    n_heads: int
    n_modules: int
    dropout_rate: float


@dataclass(frozen=True)
class DecoderOnlyConfig:
    model_dim: int
    ff_dim: int
    n_heads: int
    n_modules: int
    dropout_rate: float


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    learning_rate: float
    batch_size: int
    n_epochs: int
    metrics_freq: int
    checkpoint_freq: int
    weight_decay: float = 0.0
    lr_schedule: str | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    data: DatasetConfig
    model: EncoderDecoderConfig | DecoderOnlyConfig
    train: TrainConfig


# --------------------------------------------------------------------------------------------------

ADD3D_DATA = [
    DatasetConfig(
        dataset=f"add3d{suffix}",
        src_vocab="add",
        tgt_vocab="add",
        src_tokenizer="char",
        tgt_tokenizer="char",
        max_length=14,  # [BOS]999+999=1998[EOS]
        bucket_boundaries=None,
        token_budget=4096
    )
    for suffix in ("", "-zpad", "-oversample", "-zpad-oversample")
]

IWSLT_DATA = DatasetConfig(
    dataset="iwslt2017-de-en",
    src_vocab="de",
    tgt_vocab="en",
    src_tokenizer="de",
    tgt_tokenizer="en",
    max_length=128,
    bucket_boundaries=(16, 24, 32, 48, 64, 128),
    token_budget=4096
)

TINY_STORIES_DATA = DatasetConfig(
    dataset="tiny-stories",
    src_vocab="tiny-stories",
    tgt_vocab="tiny-stories",
    src_tokenizer="char",
    tgt_tokenizer="char",
    max_length=512,
    bucket_boundaries=None,
    token_budget=4096
)


# --------------------------------------------------------------------------------------------------

TRANSFORMER_BASE = EncoderDecoderConfig(
    model_dim=512,
    ff_dim=2048,
    n_heads=8,
    n_modules=6,
    dropout_rate=0.1
)

TRANSFORMER_BIG = EncoderDecoderConfig(
    model_dim=1024,
    ff_dim=4096,
    n_heads=16,
    n_modules=6,
    dropout_rate=0.3
)

# This is the model from "Teaching arithmetic to small transformers". Quoting that paper:
#
# NanoGPT features a aecoder-only transformer architecture with six self-attention layers, six
# heads, and an embedding dimension of 384, resulting in approximately 10.6 million parameters.
#
TRANSFORMER_DECODER_SMALL = DecoderOnlyConfig(
    # FIXME: Check model_dim and ff_dim. My understanding is that ff_dim is fixed by NanoGPT
    model_dim=384,
    ff_dim=1536,
    n_heads=6,
    n_modules=6,
    dropout_rate=0.2
)

TRANSFORMER_DECODER_TINY = DecoderOnlyConfig(
    model_dim=126,
    ff_dim=512,
    n_heads=6,
    n_modules=1,
    # Dropout encourages the model to distribute the representations, while I want interpretable
    # attention heads.
    dropout_rate=0
)

# --------------------------------------------------------------------------------------------------

ONE_BATCH_EXPERIMENT = ExperimentConfig(
    name="one_batch",
    description="Verify model overfits to one batch",
    data=IWSLT_DATA,
    model=EncoderDecoderConfig(
        model_dim=512,
        ff_dim=2048,
        n_heads=8,
        n_modules=6,
        dropout_rate=0,
    ),
    train=TrainConfig(
        seed=0,
        learning_rate=1e-4,
        batch_size=128,
        n_epochs=100,
        metrics_freq=1,
        checkpoint_freq=1,
        weight_decay=0.0
    )
)

IWSLT_EXPERIMENT = ExperimentConfig(
    name="iwslt",
    description="Train model on IWSLT 2017 German to English translation task",
    data=IWSLT_DATA,
    model=TRANSFORMER_BASE,
    train=TrainConfig(
        seed=0,
        learning_rate=1e-4,
        batch_size=128,
        n_epochs=100,
        metrics_freq=1,
        checkpoint_freq=1,
        weight_decay=0.0
    )
)

ADD3D_EXPERIMENTS = [
    ExperimentConfig(
        name=cfg.dataset,
        description="Train tiny transformer to do up-to-3-digit addition",
        data=cfg,
        model=TRANSFORMER_DECODER_TINY,
        train=TrainConfig(
            seed=0,
            learning_rate=2e-3,
            batch_size=1024,
            n_epochs=100,
            metrics_freq=1,
            checkpoint_freq=1,
            weight_decay=0.1,
            lr_schedule="cosine"
        )
    )
    for cfg in ADD3D_DATA
]


EXPERIMENT_CONFIGS = (
    [
        ONE_BATCH_EXPERIMENT,
        IWSLT_EXPERIMENT,
    ]
    +
    ADD3D_EXPERIMENTS
)

def load_experiment_config(name: str) -> ExperimentConfig:
    """
    Given a string name, return the requested experiment config. Raise if no match.
    """
    matches = [c for c in EXPERIMENT_CONFIGS if c.name == name]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise ValueError(f"Duplicate experiments named '{name}' found")
    else:
        raise ValueError(f"Experiment config '{name}' not found")
