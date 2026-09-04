import torch

class Linear(torch.nn.Module):
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

        w = torch.empty(self.in_features, self.out_features)
        std = 2.0/(self.out_features + self.in_features)
        torch.nn.init.trunc_normal_(
            w, mean=0.0, std=std, a=-3 * std**0.5, b=3 * std**0.5
        )

        self.W = torch.nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input"""

        return x @ self.W.T
