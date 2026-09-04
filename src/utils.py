import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)

    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
