import warnings
import os
import sys
from typing import Dict, Tuple, Union, Optional, Any
import json
import time
import torch
from pathlib import Path

from torch import Tensor
from torch.utils.data import IterableDataset

from utils.save import SliceNestedTensor, sparse2dense, cat_slice_nested_tensors
from configs.prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP
from prior.mix_prior import MixPrior
from prior.base import DummyPrior


warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)

class PriorDataset(IterableDataset):
    """
    Main dataset class that provides an infinite iterator over synthetic tabular datasets.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch

    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics

    batch_size_per_subgp : int, default=None
        Number of datasets per subgroup, with more similar causal structures
        If None, defaults to batch_size_per_gp

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    seq_len_per_gp : bool = False
        If True, sample sequence length per group, allowing variable-sized datasets

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets

    prior_type : str, default="mlp_scm"
        Type of prior: 'mlp_scm' (default), 'tree_scm', 'mix_scm', 'gm_scm', 'dummy', or "gm".

        1. SCM-based: Structural causal models with complex feature relationships
         - 'mlp_scm': MLP-based causal models
         - 'tree_scm': Tree-based causal models
         - 'mix_scm': Probabilistic mix of the above models
         - 'gm_scm': Gaussian Mixture Model combined with Structural Causal Model

        2. Dummy & Gaussian: Randomly generated datasets for debugging

    fixed_hp : dict, default=DEFAULT_FIXED_HP
        Fixed parameters for SCM-based priors

    sampled_hp : dict, default=DEFAULT_SAMPLED_HP
        Parameters sampled during generation

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors)

    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda')
    """

    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        replay_small: bool = False,
        prior_type: str = "mix_prior",
        fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
        sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
        device: str = "cpu",
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.generator = generator
        if prior_type == "dummy":
            self.prior = DummyPrior(
                batch_size=batch_size,
                min_features=min_features,
                max_features=max_features,
                max_classes=max_classes,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                device=device,
            )
        elif prior_type in ["gm",  "mix_prior", "gmires"]:
            self.prior = MixPrior(
                batch_size=batch_size,
                batch_size_per_gp=batch_size_per_gp,
                batch_size_per_subgp=batch_size_per_subgp,
                min_features=min_features,
                max_features=max_features,
                max_classes=max_classes,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                seq_len_per_gp=seq_len_per_gp,
                replay_small=replay_small,
                prior_type=prior_type,
                fixed_hp=fixed_hp,
                sampled_hp=sampled_hp,
                n_jobs=n_jobs,
                num_threads_per_generate=num_threads_per_generate,
                device=device,
            )
        else:
            raise ValueError(
                f"Unknown prior type '{prior_type}'. Available options: 'mlp_scm', 'tree_scm', 'mix_scm', 'gm_scm', 'gmires', 'gmnf', 'gmatt', 'dummy', 'gaussian', or 'gmm_scm'."
            )

        self.batch_size = batch_size
        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.min_features = min_features
        self.max_features = max_features
        self.max_classes = max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len
        self.seq_len_per_gp = seq_len_per_gp
        self.device = device
        self.prior_type = prior_type

    def set_anneal_config(self, max_steps: int, anneal_proportion: float):
        """
        Configure prior annealing. Effective only when the underlying prior is a MixPrior instance.

        Parameters
        ----------
        max_steps : int
            Total number of training steps.
        anneal_proportion : float
            Proportion of training steps during which annealing is applied (0.0-1.0).
        """
        if isinstance(self.prior, MixPrior):
            self.prior.set_anneal_config(max_steps, anneal_proportion)

    def set_current_step(self, step: int):
        """
        Set the current training step. Effective only when the underlying prior is a MixPrior instance.

        Parameters
        ----------
        step : int
            Current training step.
        """
        if isinstance(self.prior, MixPrior):
            self.prior.set_current_step(step)

    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict]:
        """
        Generate a new batch of datasets.

        Parameters
        ----------
        batch_size : int, optional
            If provided, overrides the default batch size for this call

        Returns
        -------
        X : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is (batch_size, seq_len, max_features).
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random Gaussian values of (batch_size, seq_len, max_features).

        y : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is (batch_size, seq_len).
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random class labels of (batch_size, seq_len).

        d : Tensor
            Number of active features per dataset of shape (batch_size,).

        seq_lens : Tensor
            Sequence length for each dataset of shape (batch_size,).

        num_classes : Tensor
            Number of classes for each dataset of shape (batch_size,).
            
        extra_data : dict
            Dictionary containing optional data:
            - "raw_y": Raw scores for score-based training (when using SCMScorePrior)
        """
        seed = torch.empty((), dtype=torch.int64, device='cpu').random_(generator=self.generator).item()
        
        batch_generator = torch.Generator()
        batch_generator.manual_seed(seed)

        result = self.prior.get_batch(batch_size, generator=batch_generator)
        
        # Handle different return formats based on result length and prior type
        if len(result) == 5:
            X, y, d, seq_lens, num_classes = result # (X, y, d, seq_lens, num_classes)
            extra_data = {}
        else:
            raise ValueError(f"Unexpected result length from prior.get_batch(): {len(result)}")
        
        # If the prior supports annealing, automatically increment the step counter
        # so that each newly generated batch uses the correct, up-to-date step value.
        if isinstance(self.prior, MixPrior):
            self.prior.increment_step()
        
        return X, y, d, seq_lens, num_classes, extra_data

    def __iter__(self) -> "PriorDataset":
        """
        Returns an iterator that yields batches indefinitely.

        Returns
        -------
        self
            Returns self as an iterator
        """
        return self

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict]:
        """
        Returns the next batch from the iterator. Since this is an infinite
        iterator, it never raises StopIteration and instead continuously generates
        new synthetic data batches.
        """
        with DisablePrinting():
            return self.get_batch()

    def __repr__(self) -> str:
        """
        Returns a string representation of the dataset.

        Provides a detailed view of the dataset configuration for debugging
        and logging purposes.

        Returns
        -------
        str
            A formatted string with dataset parameters
        """
        return (
            f"PriorDataset(\n"
            f"  prior_type: {self.prior_type}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  batch_size_per_gp: {self.batch_size_per_gp}\n"
            f"  features: {self.min_features} - {self.max_features}\n"
            f"  max classes: {self.max_classes}\n"
            f"  seq_len: {self.min_seq_len or 'None'} - {self.max_seq_len}\n"
            f"  sequence length varies across groups: {self.seq_len_per_gp}\n"
            f"  device: {self.device}\n"
            f")"
        )

class DisablePrinting:
    """Context manager to temporarily suppress printed output."""

    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.original_stdout
