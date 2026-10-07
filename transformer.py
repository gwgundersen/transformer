import copy
from flax import nnx
from flax.nnx import softmax
import jax
import jax.numpy as jnp
import numpy as np
import optax


# FIXME.
def attention(Q, K, V):
    """
    attention(q, k, v) = softmax[(Q @ K.T) / sqrt(d_k)] @ V
    """
    d_k = K.size(-1)
    return softmax((Q @ K.T) / np.sqrt(d_k)) @ V


# FIXME.
class MultiheadAttention(nnx.Module):

    def __init__(self, n_heads: int, model_dim: int):
        pass

    def forward(self, query, key, value, mask):
        """
        """
        pass


# FIXME.
class Embeddings(nnx.Module):

    def __init__(self, model_dim, vocab, rngs):
        self.lut = nnx.Embed(
            num_embeddings=len(vocab),
            features=model_dim,
            rngs=rngs
        )

    def __call__(self, x):
        # FIXME: In the AT, this is `lut(x) * math.sqrt(model_dim)`. Why?
        return self.lut(x)


# FIXME.
class PositionalEncoding(nnx.Module):
    
    def __init__(self, model_dim: int, dropout_rate: float):
        """
        """
        self.dropout = nnx.Dropout(dropout_rate)

    def __call__(self, x):
        return x


class EncoderLayer(nnx.Module):

    def __init__(self, model_dim: int, n_attn_heads: int, dropout_rate: float, rngs: nnx.Rngs):
        """
        """
        self.norm1 = nnx.LayerNorm(model_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(model_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(dropout_rate)
        self.dropout2 = nnx.Dropout(dropout_rate)
        self.self_attention = MultiheadAttention(n_attn_heads, model_dim)

    def __call__(self, x, mask):
        """
        """
        # sublayer 1
        x = self.norm1(x)
        x = self.self_attention(x, x, x, mask)
        x = x + self.dropout1(x)

        # sublayer 2
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + self.dropout2(x)

        return x


class DecoderLayer(nnx.Module):

    def __init__(self, model_dim: int, n_attn_heads: int, dropout_rate: float, rngs: nnx.Rngs):
        """
        """
        self.norm1 = nnx.LayerNorm(model_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(model_dim, rngs=rngs)
        self.norm3 = nnx.LayerNorm(model_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(dropout_rate)
        self.dropout2 = nnx.Dropout(dropout_rate)
        self.dropout3 = nnx.Dropout(dropout_rate)
        self.self_attention = MultiheadAttention(n_attn_heads, model_dim)
        self.src_attention = MultiheadAttention(n_attn_heads, model_dim)

    def __call__(sef, x, memory, src_mask, tgt_mask):
        """
        """
        x = self.norm1(x)
        x = self.self_attention(x, x, x, tgt_mask)
        x = x + self.dropout1(x)

        x = self.norm2(x)
        x = self.source_attention(x, memory, memory, src_mask)
        x = x + self.dropout2(x)

        x = self.norm3(x)
        x = self.feed_forward(x)
        x = x + self.dropout3(x)

        return x


class Transformer(nnx.Module):
    """
    Encoder-decoder transformer with multi-head self-attention.
    """

    def __init__(
        self,
        src_vocab: dict[str, int],
        tgt_vocab: dict[str, int],
        model_dim: int = 512,
        n_attn_heads: int = 8,
        dropout_rate: float = 0.1,
        num_modules: int = 6,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.encoder_layers = nnx.List([
            EncoderLayer(model_dim, n_attn_heads, dropout_rate, rngs) for _ in range(num_modules)
        ])
        self.decoder_layers = nnx.List([
            DecoderLayer(model_dim, n_attn_heads, dropout_rate, rngs) for _ in range(num_modules)
        ])

        self.src_embed = nnx.Sequential(
            Embeddings(model_dim, src_vocab, rngs),
            PositionalEncoding(model_dim, dropout_rate)
        )
        self.tgt_embed = nnx.Sequential(
            Embeddings(model_dim, tgt_vocab, rngs),
            PositionalEncoding(model_dim, dropout_rate)
        )

    def forward(self, src, tgt, src_mask, tgt_mask):
        import ipdb; ipdb.set_trace()
        z = self.encode(src, src_mask)
        return self.decode(z, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """
        """
        x = self.src_embed(src)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_mask)
        return x

    def decode(self, memory, src_mask, tgt, tgt_mask):
        x = self.tgt_embed(tgt)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, mask, memory, src_mask, tgt_mask)
        return x
