from dataclasses import dataclass

# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    dataset: str
    src_lang: str
    tgt_lang: str
    max_length: int
    bucket_boundaries: tuple[int, ...]
    token_budget: int


@dataclass(frozen=True)
class ModelConfig:
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


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    data: DatasetConfig
    model: ModelConfig
    train: TrainConfig


# --------------------------------------------------------------------------------------------------

IWSLT_DATA = DatasetConfig(
    dataset="iwslt2017-de-en",
    src_lang="de",
    tgt_lang="en",
    max_length=128,
    bucket_boundaries=(16, 24, 32, 48, 64, 128),
    token_budget=4096
)

TRANSFORMER_BASE = ModelConfig(
    model_dim=512,
    ff_dim=2048,
    n_heads=8,
    n_modules=6,
    dropout_rate=0.1
)

TRANSFORMER_BIG = ModelConfig(
    model_dim=1024,
    ff_dim=4096,
    n_heads=16,
    n_modules=6,
    dropout_rate=0.3
)

ONE_BATCH_EXPERIMENT = ExperimentConfig(
    name="one_batch",
    description="Verify model overfits to one batch",
    data=IWSLT_DATA,
    model=ModelConfig(
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
        checkpoint_freq=1
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
        checkpoint_freq=1
    )
)

EXPERIMENT_CONFIGS = [
    ONE_BATCH_EXPERIMENT,
    IWSLT_EXPERIMENT,
]

def load_experiment_config(name: str) -> ExperimentConfig:
    """
    Given a string name, return the requested experiment config. Raise if no match.
    """
    matches = [c for c in EXPERIMENT_CONFIGS if c.name == name]
    if len(matches) == 1:
        return matches[0]
    else:
        raise ValueError(f"Experiment config {name} not found")
