import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jax.Array
ArrayLike = jax.typing.ArrayLike

class FFN(nn.Module):
    dim: int
    ff_dim: int
    dropout: float

    deterministic: bool=False
    
    @nn.compact
    def __call__(self, x, *, deterministic=None):
        if deterministic is None:
            deterministic = self.deterministic
        
        shortcut = x

        x = nn.Dense(self.ff_dim)(x)
        x = jax.nn.gelu(x)
        x = nn.Dropout(self.dropout)(x, deterministic=deterministic)
        x = nn.Dense(self.dim)(x)
        x = nn.Dropout(self.dropout)(x, deterministic=deterministic)
        x = nn.LayerNorm()(x + shortcut)
        
        return x

class Block(nn.Module):
    n_heads: int
    dim: int
    dropout: float
    ff_dropout: float

    deterministic: bool=False
    
    @nn.compact
    def __call__(self, x, y, mask, *, deterministic=None):
        if deterministic is None:
            deterministic = self.deterministic
        
        x, y, mask = x

        if y is None:
            y = x

        shortcut = x

        x = nn.MultiHeadDotProductAttention(
            self.n_heads, dropout_rate=self.dropout
        )(x, y, y, mask=mask, deterministic=deterministic)

        x = nn.LayerNorm()(x+shortcut)

        x = FFN(self.dim, self.dim * 4, self.ff_dropout)(x, deterministic=deterministic)
        
        return x


class Encoder(nn.Module):
    """ A Transformer encoder model for MS data.
    """
    n_out_tokens: int = 1
    dim: int = 64

    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0
    ff_dropout: float = 0

    deterministic: bool = False

    @nn.compact
    def __call__(self, tokens:ArrayLike, mask:ArrayLike|None, *, deterministic:bool|None=None)->Array:
        """ Encoder
        Args:
            tokens: (batch, n_tokens, d)
            mask: (batch, n_tokens) a mask array where True indicates valid tokens        
        Retuns:
            (batch, n_out_tokens, d)
        """
        if deterministic is None:
            deterministic = self.deterministic

        output_tokens = nn.Embed(self.n_out_tokens, self.dim)(jnp.arange(self.n_out_tokens))

        x = nn.Dense(self.dim)(tokens)
        x = jnp.concatenate([x, output_tokens[None, :, :]], axis=1)

        mask = jnp.r_[mask, jnp.asarray([True] * self.n_out_tokens)]

        for _ in range(self.n_layers):
            x = Block(
                self.n_heads,
                self.dim,
                self.dropout,
                self.ff_dropout,
            )((x, None, mask), deterministic=deterministic)

        out = x[-self.n_out_tokens:]

        self.sow("intermediates", "state", out)
        
        return out


class Decoder(nn.Module):
    """ A Transformer decoder model for MS data.
    """
    dim: int = 64
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0
    ff_dropout: float = 0

    deterministic: bool = False

    @nn.compact
    def __call__(self, tokens:ArrayLike, enc_tokens:ArrayLike, mask:ArrayLike|None = None, *, deterministic=None)->Array:
        """ Decoder
        Args:
            tokens: (batch, n_tokens, d) query tokens
            enc_tokens: (batch, n_enc_tokens, d) output of the encoder
            mask: (batch, n_enc_tokens) a mask array where True indicates valid tokens        
        Retuns:
            (batch, n_out_tokens, d)
        """
        if deterministic is None:
            deterministic = self.deterministic

        x = nn.Dense(self.dim)(tokens)
        y = nn.Dense(self.dim)(enc_tokens)

        for _ in range(self.n_layers):
            x = Block(
                self.n_heads,
                self.dim,
                self.dropout,
                self.ff_dropout,
            )((x, y, mask), deterministic=deterministic)

        out = x[-self.n_out_tokens:]

        self.sow("intermediates", "state", out)
        
        return out


class PhosphoNet(nn.Module):
    """ A autoencoder model for MS data.
    """
    n_out_tokens: int = 1
    dim: int = 64

    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0
    ff_dropout: float = 0

    deterministic: bool = False

    @nn.compact
    def __call__(
        self, tokens:jax.ArrayLike, 
        mask: jax.ArrayLike, 
        queries: jax.typing.ArrayLike, 
        query_mask: jax.ArrayLike|None = None,
        *, 
        deterministic=None,
    )->jax.Array:
        """ Encoder
        Args:
            tokens: (batch, n_tokens, d)
            mask: (batch, n_tokens) a mask array where True indicates valid tokens
            queries: (batch, n_query_tokens, d)
            query_mask: (batch, n_query_tokens) a mask array where True indicates valid query
        Retuns:
            (batch, n_query_tokens, 1)
        """
        if deterministic is None:
            deterministic = self.deterministic
        
        enc_tokens = Encoder(
            self.n_out_tokens,
            self.dim,
            self.n_heads,
            self.n_layers,
            self.dropout,
            self.ff_dropout,
        )(tokens, mask, deterministic=deterministic)

        self.sow("intermediates", "enc_tokens", enc_tokens)

        query_out = Decoder(
            self.dim,
            self.n_heads,
            self.n_layers,
            self.dropout,
            self.ff_dropout,
        )(queries, enc_tokens, query_mask, deterministic=deterministic)

        self.sow("intermediates", "query_out", query_out)

        out = nn.Dense(1)(query_out)

        return out
