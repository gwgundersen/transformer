import jax.numpy as jnp
import numpy as np
from flax import nnx

"""
Encoder:
  source self-attention

Decoder:
  target self-attention
  target-source cross-attention
"""

# --------------------------------------------------------------------------------------------------

MAX_LEN = 5000


def attention(
    query: jnp.array,
    key: jnp.array,
    value: jnp.array,
    mask: jnp.array
) -> jnp.array:
    """
    attention(q, k, v) = softmax[(Q @ K.T) / sqrt(d_k)] @ V

    query shape = (query_len, d_k)
    key shape   = (key_len,   d_k)
    value shape = (value_len, d_v)
   
    so scores have shape (query_len, key_len)
    but then attention mask will have shape (query_len, key_len)
    because it's masking the scores
    """
    d_k = query.shape[-1]

    # query and key have shapes (batch, head, seq, model)
    # we want do the matrix multiplicataion: (batch, head, seq, model) @ (batch, head, model, seq)
    # the resulting vector has dimension (batch, head, seq, seq)
    # so each head has its own (seq, seq) attention layer, and we run this over tbatches
    scores = (
        query @ jnp.transpose(key, (0, 1, 3, 2))
    ) / np.sqrt(d_k)

    # FIXME: Explain why masking. Answer:
    # 1) padding
    # 2) causlity
    scores = jnp.where(mask, scores, -jnp.inf)

    # qk_scaled has shape (batch, head, seq, seq)
    # value has shape (batch, head, seq, d_k)
    # output has shape (batch, head, seq, d_k)
    return jnp.matmul(nnx.softmax(scores), value)


def make_src_mask(src):
    """
    """
    # Attention has shape (B, H, Q, K); add dimensions so that src mask works.
    return (src != PAD_ID)[:, None, None, :]


def make_tgt_mask(tgt_input: np.ndarray | jnp.ndarray) -> np.ndarray | jnp.ndarray:
    """
    """
    # Add new axis to allow broadcasting over n_heads
    causal_mask = make_causal_mask(tgt_input.shape[1])
    return (tgt_input != PAD_ID)[:, None, None, :] & causal_mask


def make_causal_mask(size: int) -> np.ndarray:
    """
    Build the mask that enforces causality when decoding.
    """
    score_shape = (size, size)
    return np.tril(np.ones(score_shape), k=0).astype(bool)


# --------------------------------------------------------------------------------------------------

class MultiheadAttention(nnx.Module):

    def __init__(self, n_heads: int, model_dim: int, rngs):
        """
        """
        # FIXME: This surprises me. I thought multi-head attention meant attention was repeated over
        # same input window
        self.d_k = model_dim // n_heads
        self.n_heads = n_heads
        self.model_dim = model_dim
        self.linear_q = nnx.Linear(model_dim, model_dim, rngs=rngs)
        self.linear_k = nnx.Linear(model_dim, model_dim, rngs=rngs)
        self.linear_v = nnx.Linear(model_dim, model_dim, rngs=rngs)
        self.linear_out = nnx.Linear(model_dim, model_dim, rngs=rngs)

        heads = {}
        for i in range(n_heads):
            pass

    def __call__(self, query, key, value, mask):
        """
        """
        batch_size, seq_len, model_dim = query.shape

        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        # Note: d_k * n_heads = model_dim, so:
        # Take 512-dimensional vector (model_dim = 512)
        # Split it into 8 chunks (n_heads = 8)
        query = query.reshape(batch_size, -1, self.n_heads, self.d_k)
        key = key.reshape(batch_size, -1, self.n_heads, self.d_k)
        value = value.reshape(batch_size, -1, self.n_heads, self.d_k)

        # FIXME: Explain
        query = jnp.transpose(query, (0, 2, 1, 3))
        key = jnp.transpose(key, (0, 2, 1, 3))
        value = jnp.transpose(value, (0, 2, 1, 3))

        # Apply attention across the heads in batch, rather than:
        #
        #     attn = []
        #     for _ in range(self.heads):
        #         attn.append(attention(query, key, value, mask)
        #.    attn = concat(attn)
        #
        # One could argue that this operation here is one of a few reasons transformers are so
        # successful. The operations are embarrassingly parallel.
        #
        x = attention(query, key, value, mask)

        # This effectively concatenates the head outputs. First put back dimensions, then reshape
        # (B, H, S, Dh) -> (B, S, H, Dh)
        x = jnp.transpose(x, (0, 2, 1, 3))
        x = x.reshape(batch_size, seq_len, model_dim)

        return self.linear_out(x)


class PositionwiseFeedForward(nnx.Module):

    def __init__(self, model_dim, ff_dim, dropout_rate, rngs):
        """
        """
        self.w1 = nnx.Linear(model_dim, ff_dim, rngs=rngs)
        self.w2 = nnx.Linear(ff_dim, model_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x):
        """
        """
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
        self, model_dim: int, ff_dim: int, n_heads: int, dropout_rate: float, rngs: nnx.Rngs
    ):
        """
        """
        self.norm1 = nnx.LayerNorm(model_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(model_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.self_attention = MultiheadAttention(n_heads, model_dim, rngs=rngs)
        self.feed_forward = PositionwiseFeedForward(model_dim, ff_dim, dropout_rate, rngs=rngs)

    def __call__(self, x, mask):
        """
        """
        # FIXME: explain sublayer 1
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x, x, x, mask)
        x = residual + self.dropout1(x)

        # FIXME: explain sublayer 2
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + self.dropout2(x)

        return x


class DecoderLayer(nnx.Module):

    def __init__(
        self, model_dim: int, ff_dim: int, n_heads: int, dropout_rate: float, rngs: nnx.Rngs
    ):
        """
        """
        self.norm1 = nnx.LayerNorm(model_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(model_dim, rngs=rngs)
        self.norm3 = nnx.LayerNorm(model_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.dropout3 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.self_attention = MultiheadAttention(n_heads, model_dim, rngs=rngs)
        self.cross_attention = MultiheadAttention(n_heads, model_dim, rngs=rngs)
        self.feed_forward = PositionwiseFeedForward(model_dim, ff_dim, dropout_rate, rngs)

    def __call__(self, x, encoder_output, src_mask, tgt_mask):
        """
        x has shape (batch_dim, seq_len-1, model_dim)
        """
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x, x, x, tgt_mask)
        x = residual + self.dropout1(x)

        residual = x
        x = self.norm2(x)
        x = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = residual + self.dropout2(x)

        residual = x
        x = self.norm3(x)
        x = self.feed_forward(x)
        x = residual + self.dropout3(x)

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
        n_heads: int = 8,
        dropout_rate: float = 0.1,
        n_modules: int = 6,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """
        """
        self.src_embed = EmbeddingLayer(model_dim, src_vocab, dropout_rate, rngs)
        self.tgt_embed = EmbeddingLayer(model_dim, tgt_vocab, dropout_rate, rngs)
        self.encoder_layers = nnx.List([
            EncoderLayer(model_dim, ff_dim, n_heads, dropout_rate, rngs)
            for _ in range(n_modules)
        ])
        self.decoder_layers = nnx.List([
            DecoderLayer(model_dim, ff_dim, n_heads, dropout_rate, rngs)
            for _ in range(n_modules)
        ])
        self.encoder_norm = nnx.LayerNorm(model_dim, rngs=rngs)
        self.decoder_norm = nnx.LayerNorm(model_dim, rngs=rngs)
        self.output_projection = nnx.Linear(model_dim, len(tgt_vocab), rngs=rngs)

    def __call__(self, src, tgt, src_mask, tgt_mask):
        """
        """
        x = self.encode(src, src_mask)
        x = self.decode(x, src_mask, tgt, tgt_mask)
        return self.output_projection(x)

    def encode(self, src, src_mask):
        """
        """
        x = self.src_embed(src)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_mask)
        return self.encoder_norm(x)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        """
        x = self.tgt_embed(tgt)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, memory, src_mask, tgt_mask)
        return self.decoder_norm(x)
