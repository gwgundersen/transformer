import copy
from flax import nnx
from flax.nnx import softmax
import jax
import jax.numpy as jnp
import numpy as np
import optax


MAX_LEN = 5000


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

    def __call__(self, query, key, value, mask):
        """
        """
        return query


class PositionwiseFeedForward(nnx.Module):

    def __init__(self, model_dim, ff_dim, dropout_rate, rngs):
        self.w1 = nnx.Linear(model_dim, ff_dim, rngs=rngs)
        self.w2 = nnx.Linear(ff_dim, model_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.w1(x))
        x = self.dropout(x)
        return self.w2(x)


"""
PE(pos, 2i)   = sin(pos/10000^{2i/dmodel})
PE(pos, 2i+1) = cos(pos/10000^{2i/dmodel})
"""
def make_positional_encoding(model_dim: int) -> jnp.array:
    """
    """
    # positions has shape (seq_len, 1)
    positions = jnp.arange(MAX_LEN)[:, None]

    # i has shape (model_dim // 2,), since duplicated
    i = jnp.arange(model_dim // 2)

    # scale has shape (model_dim // 2,)
    scale = 1 / jnp.pow(10000.0, (2*i) / model_dim)

    # theta has shape (seq_len, model_dim // 2)
    theta = positions * scale

    # fill in pe matrix; Jax arrays are immutable so use at/set
    pe = jnp.zeros((MAX_LEN, model_dim))
    pe = pe.at[:, 0::2].set(jnp.sin(theta))
    pe = pe.at[:, 1::2].set(jnp.cos(theta))

    return pe


# FIXME: Undestand this line of reasoning better
#
# sqrt(d) here does not make sense without understanding Xavier-initialized embeddings
# This initialization plus sqrt(d) makes the embeddings roughly the same scale as positional
# encodings.
class EmbeddingLayer(nnx.Module):

    def __init__(self, model_dim, vocab, dropout_rate, rngs):
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.embedding = nnx.Embed(num_embeddings=len(vocab), features=model_dim, rngs=rngs)
        self.scale = np.sqrt(model_dim)
        self.pe = make_positional_encoding(model_dim)

    def __call__(self, x):
        # The input to the transformer is:
        #
        #   scale(dim) * Embedding(tok) + PositionalEncoding(pos)
        #
        seq_len = x.shape[1]

        # pe has shape (MAX_LEN, model_dim); slice it to (batch_size, seq_len, model_dim)
        x = self.embedding(x) * self.scale + self.pe[None, :seq_len, :]
        return self.dropout(x)


class EncoderLayer(nnx.Module):

    def __init__(
        self, model_dim: int, ff_dim: int, n_attn_heads: int, dropout_rate: float, rngs: nnx.Rngs
    ):
        """
        """
        self.norm1 = nnx.LayerNorm(model_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(model_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.self_attention = MultiheadAttention(n_attn_heads, model_dim)
        self.feed_forward = PositionwiseFeedForward(model_dim, ff_dim, dropout_rate, rngs)

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

    def __init__(
        self, model_dim: int, ff_dim: int, n_attn_heads: int, dropout_rate: float, rngs: nnx.Rngs
    ):
        """
        """
        self.norm1 = nnx.LayerNorm(model_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(model_dim, rngs=rngs)
        self.norm3 = nnx.LayerNorm(model_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.dropout3 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.self_attention = MultiheadAttention(n_attn_heads, model_dim)
        self.src_attention = MultiheadAttention(n_attn_heads, model_dim)
        self.feed_forward = PositionwiseFeedForward(model_dim, ff_dim, dropout_rate, rngs)

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
        ff_dim: int = 2048,
        n_attn_heads: int = 8,
        dropout_rate: float = 0.1,
        num_modules: int = 6,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """
        """
        self.src_embed = EmbeddingLayer(model_dim, src_vocab, dropout_rate, rngs)
        self.tgt_embed = EmbeddingLayer(model_dim, tgt_vocab, dropout_rate, rngs)
        self.encoder_layers = nnx.List([
            EncoderLayer(model_dim, ff_dim, n_attn_heads, dropout_rate, rngs) for _ in range(num_modules)
        ])
        self.decoder_layers = nnx.List([
            DecoderLayer(model_dim, ff_dim, n_attn_heads, dropout_rate, rngs) for _ in range(num_modules)
        ])

    def __call__(self, src, tgt, src_mask, tgt_mask):
        """
        """
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
        """
        """
        x = self.tgt_embed(tgt)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, memory, src_mask, tgt_mask)
        return x
