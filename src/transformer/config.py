from dataclasses import dataclass


@dataclass(frozen=True)
class IWSLTConfig:
    max_length: int = 50
    model_dim: int = 512
    ff_dim: int = 2048
    n_heads: int = 8
    n_layers: int = 6
    dropout_rate: float = 0.1
