import torch

def grad_stats(module):
    total_elems = 0
    nonzero_elems = 0
    l2_sq_sum = 0.0
    max_abs = 0.0
    nan_count = 0
    inf_count = 0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        # Check for NaN and Inf values
        finite_mask = torch.isfinite(g)
        nan_count += (g != g).sum().item()  # NaN
        inf_count += (~finite_mask).sum().item() - (g != g).sum().item()
        g = g[finite_mask]
        if g.numel() == 0:
            continue
        total_elems += g.numel()
        nonzero_elems += (g != 0).sum().item()
        l2_sq_sum += float((g.double() * g.double()).sum().item())
        max_abs = max(max_abs, float(g.abs().max().item()))
    l2 = (l2_sq_sum ** 0.5)
    sparsity = 1.0 - (nonzero_elems / total_elems) if total_elems > 0 else 0.0
    return {
        "global_grad_norm": l2,
        "grad_max_abs": max_abs,
        "grad_sparsity": sparsity,
        "grad_nan_count": nan_count,
        "grad_inf_count": inf_count,
        # "grad_total_elems": total_elems,
    }

import random
import numpy as np
import torch

def set_seed(np_seed: int, torch_seed: int, deterministic: bool = False) -> None:

    random.seed(np_seed)
    np.random.seed(np_seed)

    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False