from einops import einsum
import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """Construct a linear transformation module. This function should accept the following parameters:

        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        w = torch.empty(self.in_features, self.out_features, device=device, dtype=dtype)
        std = 2.0/(self.out_features + self.in_features)
        nn.init.trunc_normal_(
            w, mean=0.0, std=std, a=-3 * std**0.5, b=3 * std**0.5
        )

        self.W = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input"""

        return x @ self.W.T


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.d_ff =  d_ff

        w1 = torch.zeros(self.d_ff, self.d_model, device=device, dtype=dtype)
        w2 = torch.zeros(self.d_model, self.d_ff, device=device, dtype=dtype)
        w3 = torch.zeros(self.d_ff, self.d_model, device=device, dtype=dtype)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w3 = nn.Parameter(w3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: d_model
        w1: d_ff x d_model
        w2: d_model x d_ff
        w3: d_ff x d_model

        w1_x: ... d_ff
        silu: ... d_ff
        w3_x: ... d_ff
        """
        w1_x = einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = w1_x / (1 + torch.exp(-w1_x))

        w3_x = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu_w3_x = silu * w3_x

        swiglu = einsum(
            silu_w3_x, self.w2,
            "batch sequence d_ff, d_model d_ff -> batch sequence d_model",
        )

        return swiglu
