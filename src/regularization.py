from einops import einsum, rearrange
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """ Construct the RMSNorm module. This function should accept the following parameters:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        gain = torch.zeros(self.d_model, device=device, dtype=dtype)
        self.gain = nn.Parameter(gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape."""
        out_dtype = x.dtype
        x = x.to(dtype=torch.float32)

        squared_add_eps = einsum(
            x, x, "batch seq d_model, batch seq d_model -> batch seq"
        ) / self.d_model + self.eps
        rms = torch.sqrt(squared_add_eps)

        result = einsum(
            x, self.gain, "batch seq d_model, d_model -> batch seq d_model"
        ) / rearrange(rms, "batch (seq v) -> batch seq v", v=1)

        return result.to(dtype=out_dtype)
