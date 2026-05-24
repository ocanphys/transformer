import torch
from einops import einsum


class Linear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        self.W = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty((out_features, in_features), device=device, dtype=dtype), mean=0, std=std
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, d_in)
        return einsum(self.W, x, "d_out d_in, batch ... d_in -> batch ... d_out")


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            ## default implementation of torch.nn.Embedding uses .normal_ only -
            torch.nn.init.trunc_normal_(
                torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype), mean=0, std=1
            )
        )
        pass

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
