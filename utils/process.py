import torch
from torch import Tensor

def normalize_data(x: Tensor, eps=1e-8) -> Tensor:

    mean = torch.mean(x, dim=0, keepdim=True)
    std = torch.std(x, dim=0, keepdim=True, unbiased=False)
    std = torch.clamp(std, min=eps)  
    return (x - mean) / std
