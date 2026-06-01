import torch
import torch.nn as nn
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        self.W = nn.Parameter(
            nn.init.trunc_normal_(torch.empty((out_features, in_features), device=device, dtype=dtype), mean=0, std=std)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, d_in)
        return einsum(self.W, x, "d_out d_in, batch ... d_in -> batch ... d_out")


class Embedding(nn.Module):
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
                torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype), mean=0, std=1
            )
        )
        pass

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
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
        return torch.einsum("...d,d->...d", (x / rms_a).to(in_dtype), self.gain.to(in_dtype))


class SwiGLU(nn.Module):
    def __init__(self, d_ff, d_model):
        super().__init__()
        self.W1 = nn.Parameter(torch.ones((d_ff, d_model)))
        self.W2 = nn.Parameter(torch.ones((d_model, d_ff)))
        self.W3 = nn.Parameter(torch.ones((d_ff, d_model)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        SiLU = lambda x: x * torch.sigmoid(x)
        hadamard = SiLU(torch.einsum("f m, ... m -> ... f", self.W1, x)) * torch.einsum(
            "f m, ... m -> ... f", self.W3, x
        )
        return torch.einsum("m f, ... f -> ... m", self.W2, hadamard)


def SPDA(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_K = K.shape[-1]
    attention_weights = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (d_K**0.5)
    if mask is not None:
        attention_weights += torch.where(mask, 0, float("-inf"))
    softmax = torch.softmax(attention_weights, dim=-1)
    return softmax @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        ## need to pass the diagonal mask to pass the test.
        diagonal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=0)
        Q = einsum(x, self.WQ, "batch ... sequence model, query model -> batch ... sequence query")
        K = einsum(x, self.WK, "batch ... sequence model, key model -> batch ... sequence key")
        V = einsum(x, self.WV, "batch ... sequence model, value model -> batch ... sequence value")
        head_rep = lambda M: rearrange(M, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads)
        hQ, hK, hV = head_rep(Q), head_rep(K), head_rep(V)
        mult_head_attention = rearrange(SPDA(hQ, hK, hV, diagonal_mask), "... h seq_len d_val -> ... seq_len (h d_val)")
        return einsum(mult_head_attention, self.WO, "... hdval, ... model hdval -> ... model")


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        i = torch.arange(0, max_seq_len)
        k = torch.arange(1, d_k // 2 + 1)
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
        x_rope = rearrange(torch.stack((even_out, odd_out), dim=-1), "... blocks pair -> ... (blocks pair)")
        return x_rope
