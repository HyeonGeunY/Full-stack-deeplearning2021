import math
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(torch.nn.Module):
    """
    Classic Attention-is-all-you-need positional encoding
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = self.make_pe(d_model=d_model, max_len=max_len)  # positional encoding??
        self.register_buffer("pe", pe) # 업데이트 하지 않는 layer 지정

    @staticmethod
    def make_pe(d_model: int, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # (max_len,) -> (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # (max_len, d_model // 2)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (S, B, d_model)
        assert x.shape[2] == self.pe.shape[2]
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def generate_square_subsequent_mask(size: int) -> torch.Tensor:
    """
    Generate a triangular (size, size) mask.
    """

    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask
