import logging

import torch
import torch.nn as nn
from einops import einsum, rearrange

logger = logging.getLogger(__name__)


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        self.W = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty((out_features, in_features), device=device, dtype=dtype),
                mean=0,
                std=std,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(nn.Module):
    """Token embedding table mapping integer token IDs to dense vectors.

    Args:
        num_embeddings: Vocabulary size — the number of distinct tokens.
        embedding_dim: Embedding dimension (d_model).
        device: Device to store the parameters on.
        dtype: Data type of the parameters
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            ## default implementation of nn.Embedding uses .normal_ only -
            nn.init.trunc_normal_(
                torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype),
                mean=0,
                std=1,
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones((d_model), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(
            torch.float32
        )  # since we will square numbers - to avoid overflow we upcast in case dtype has smaller headroom
        rms_a = ((x**2).mean(dim=-1, keepdim=True) + self.eps) ** 0.5  # keepdim keeps the last dimension
        return einsum(
            (x / rms_a).to(in_dtype),
            self.gain.to(in_dtype),
            "... d_model, d_model -> ... d_model",
        )


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_ff,
        d_model,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        kaiming = lambda out_features, in_features: nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty((out_features, in_features), device=device, dtype=dtype),
                mean=0,
                std=1 / in_features**0.5,
            )
        )
        self.W1 = kaiming(d_ff, d_model)
        self.W2 = kaiming(d_model, d_ff)
        self.W3 = kaiming(d_ff, d_model)
        # lore is that d_ff/d_model is roughly 4 (consensus hyparam)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu = lambda x: x * torch.sigmoid(x)
        hadamard = silu(einsum(self.W1, x, "d_ff d_model, ... d_model -> ... d_ff")) * einsum(
            self.W3, x, "d_ff d_model, ... d_model -> ... d_ff"
        )
        return einsum(self.W2, hadamard, "d_model d_ff, ... d_ff -> ... d_model")


def SPDA(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
    return_weights: bool = False,
):
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
        return_weights (bool): if True, also return the pre-softmax attention
            scores (post-mask); otherwise that element of the pair is None.
    Returns:
        tuple of (Float[Tensor, " ... queries d_v"], Float[Tensor, " ... queries keys"] | None):
            the SDPA output and the pre-softmax attention weights (or None).
    """
    d_k = K.shape[-1]
    attention_weights = einsum(Q, K, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k") / (d_k**0.5)
    if mask is not None:
        attention_weights += torch.where(mask, 0, float("-inf"))
    softmax = torch.softmax(attention_weights, dim=-1)
    return softmax @ V, (attention_weights if return_weights else None)


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        i = torch.arange(0, max_seq_len, device=device)
        k = torch.arange(1, d_k // 2 + 1, device=device)
        freq = 1 / theta ** ((2.0 * k - 2.0) / d_k)
        theta_ik = i[:, None] * freq[None, :]
        self.register_buffer("cos", theta_ik.cos(), persistent=False)
        self.register_buffer("sin", theta_ik.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # here x is the K or Q with dims ... x sequence x d_k (rep'n dim)
        # separate them in to even and odd indices
        evens, odds = rearrange(x, "... (blocks pair) -> ... pair blocks", pair=2).unbind(dim=-2)
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]
        even_out = cos * evens - sin * odds
        odd_out = sin * evens + cos * odds
        x_rope = rearrange(
            torch.stack((even_out, odd_out), dim=-1),
            "... blocks pair -> ... (blocks pair)",
        )
        return x_rope


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        max_seq_len,
        rope: RoPE | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        store_cache: bool = False,
    ):
        super().__init__()
        self.rope = rope  # positional embedding submodule
        # debug toggle: when True, forward() stashes detached intermediates in self._cache
        self.store_cache = store_cache
        self._cache: dict[str, torch.Tensor] = {}

        ## need to pass the diagonal mask to pass the test - create
        diagonal_mask = torch.tril(
            torch.ones((max_seq_len, max_seq_len), device=device, dtype=torch.bool),
            diagonal=0,
        )
        self.register_buffer("mask", diagonal_mask, persistent=False)

        assert d_model % num_heads == 0
        mean, std = 0, 0.02  ## why make these choices
        self.num_heads = num_heads
        d_Q = d_K = d_V = d_model // num_heads
        ## convention from the spec, note WO dimensions are flipped (out, in)
        self.WQ = nn.Parameter(torch.empty(d_Q * num_heads, d_model, device=device, dtype=dtype))
        self.WK = nn.Parameter(torch.empty(d_K * num_heads, d_model, device=device, dtype=dtype))
        self.WV = nn.Parameter(torch.empty(d_V * num_heads, d_model, device=device, dtype=dtype))
        self.WO = nn.Parameter(torch.empty(d_model, d_V * num_heads, device=device, dtype=dtype))
        for p in (self.WQ, self.WK, self.WV, self.WO):
            nn.init.normal_(p, mean=mean, std=std)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[-2]
        mask = self.mask[:seq_len, :seq_len]
        Q = einsum(x, self.WQ, "batch ... seq d_model, d_qkv d_model -> batch ... seq d_qkv")
        K = einsum(x, self.WK, "batch ... seq d_model, d_qkv d_model -> batch ... seq d_qkv")
        V = einsum(x, self.WV, "batch ... seq d_model, d_qkv d_model -> batch ... seq d_qkv")
        head_rep = lambda M: rearrange(
            M,
            "... seq (n_heads d_head) -> ... n_heads seq d_head",
            n_heads=self.num_heads,
        )
        hQ, hK, hV = (
            head_rep(Q),
            head_rep(K),
            head_rep(V),
        )  # introduced the head dimension
        if self.rope is not None:
            assert token_positions is not None
            hK = self.rope(hK, token_positions)
            hQ = self.rope(hQ, token_positions)
        attn, attn_weights = SPDA(hQ, hK, hV, mask, return_weights=self.store_cache)
        mult_head_attention = rearrange(attn, "... n_heads seq d_head -> ... seq (n_heads d_head)")
        out = einsum(mult_head_attention, self.WO, "... d_v, ... d_model d_v -> ... d_model")

        if self.store_cache:
            self._cache = {
                "Q": Q.detach(),
                "K": K.detach(),
                "V": V.detach(),  # pre-head-split projections
                "hQ": hQ.detach(),
                "hK": hK.detach(),
                "hV": hV.detach(),  # head-split (hQ/hK post-rope if rope is set)
                "attn_weights": attn_weights.detach(),  # pre-softmax per-head attention scores
                "attn": attn.detach(),  # per-head attention output
                "out": out.detach(),  # final MHA output
            }
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.rope = RoPE(
            theta=theta,
            d_k=d_model // num_heads,
            max_seq_len=max_seq_len,
            device=device,
        )
        self.norm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope=self.rope,
            device=device,
            dtype=dtype,
        )
        self.norm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_ff=d_ff, d_model=d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(x.shape[-2], device=x.device)
        y = x + self.attention(self.norm1(x), token_positions)
        z = y + self.ffn(self.norm2(y))
        return z


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # torch submodules
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.linear = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)
        self.tblocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        x = self.embedding(in_features)
        for block in self.tblocks:
            x = block(x)
        z = self.linear(self.norm(x))
        return z
