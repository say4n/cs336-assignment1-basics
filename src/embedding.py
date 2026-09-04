from einops import rearrange
import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """Construct an embedding module. 
        This function should accept the following parameters:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., d_model
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        embedding_matrix = torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        self.embedding_matrix = nn.Parameter(embedding_matrix)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs."""
        return self.embedding_matrix[token_ids]

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """Construct the RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be input
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        token_positions = torch.arange(max_seq_len, device=self.device).unsqueeze(1)
        exponent = (2 * torch.arange(1, d_k // 2 + 1) - 2) / d_k
        denominator = self.theta ** exponent
        inverse_denominator = 1 / denominator

        angles = token_positions * inverse_denominator

        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (..., seq_len, d_k) and return a tensor 
        of the same shape. Note that you should tolerate x with an arbitrary number of 
        batch dimensions. You should assume that the token positions are a tensor of 
        shape (..., seq_len) specifying the token positions of x along the sequence 
        dimension.
        """
        cos_values = self.cos[token_positions]
        sin_values = self.sin[token_positions]

        even_x = x[..., 0::2]
        odd_x = x[..., 1::2]

        rotated_even_x = even_x * cos_values - odd_x * sin_values
        rotated_odd_x = even_x * sin_values + odd_x * cos_values

        rotated_x = rearrange([rotated_even_x, rotated_odd_x], "list ... -> ... list")
        interleaved_x = rearrange(
            rotated_x, "... angle_dim even_odd_dim -> ... (angle_dim even_odd_dim)"
        )

        return interleaved_x
