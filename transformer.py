import copy
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax


"""
EncoderDecoder: core object

Encoder:
- multiple EncoderLayer layers + LayerNorm
"""

def clones(module: nnx.Module, N: int) -> list[nnx.Module]:
    """Return N identical layers."""
    return [copy.deepcopy(module) for _ in range(N)]


class LayerNorm(nnx.Module):
    """ FIXME: Add docstring"""

    def __init__(self, features, eps=1e-6):
        """ """
        super(LayerNorm, self).__init__()
        self.a = nnx.Param(features)
        self.b = nnx.Param(features)
        self.eps = eps

    def forward(self, x):
        # FIXME: Is reshaping mean/std in this way the right way to do it?
        mean = x.mean(-1)[:, None]
        std = x.std(-1)[:, None]
        z = (x - mean) / (std + self.eps)
        return self.a * z + self.b


class SublayerConnection(nnx.Module):
    """
    LayerNorm(x + Sublayer(x))
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nnx.Dropout(dropout)

    def forward(self, x, sublayer):
        pass


class Encoder(nnx.Module):

    def __init__(self, layer, N):
        """ """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(...)  # FIXME: What is layer.size? N? Something else?
    
    def forward(self, x, mask):
        """ """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


NUM_SAMPLES = 100
X_DIM = 10
rng = jax.random.PRNGKey(0)
x = jax.random.normal(rng, (NUM_SAMPLES, X_DIM))

norm = LayerNorm(X_DIM)
print(norm)
norm.forward(x)
