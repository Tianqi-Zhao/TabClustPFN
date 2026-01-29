"""
The module offers a flexible framework for creating diverse, realistic tabular datasets
with controlled properties, which can be used for training and evaluating in-context
learning models. Key features include:

- Controlled feature relationships and causal structures via multiple generation methods
- Customizable feature distributions with mixed continuous and categorical variables
- Flexible train/test splits optimized for in-context learning evaluation
- Batch generation capabilities with hierarchical parameter sharing
- Memory-efficient handling of variable-length datasets

The main class is PriorDataset, which provides an iterable interface for generating
an infinite stream of synthetic datasets with diverse characteristics.
"""

from __future__ import annotations

import warnings
import math
import inspect
from typing import Dict, Tuple, Union, Optional, Any
import os

import numpy as np
from scipy.stats import loguniform
import joblib

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nested import nested_tensor

from utils.process import normalize_data

from abc import ABC, abstractmethod

# Import R_HOME configuration
from configs.env_config import R_HOME_PATH

# Set R_HOME environment variable
os.environ['R_HOME'] = R_HOME_PATH


warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)

class Prior:
    """
    Abstract base class for dataset prior generators.

    Defines the interface and common functionality for different types of
    synthetic dataset generators.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets
    """

    def __init__(
        self,
        batch_size: int = 256,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        replay_small: bool = False,
    ):
        self.batch_size = batch_size

        assert min_features <= max_features, "Invalid feature range"
        self.min_features = min_features
        self.max_features = max_features

        self.max_classes = max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len

        self.replay_small = replay_small

    @staticmethod
    def sample_seq_len(
        min_seq_len: Optional[int], max_seq_len: int, log: bool = False, replay_small: bool = False, generator=None
    ) -> int:
        """
        Selects a random sequence length within the specified range.

        This method provides flexible sampling strategies for dataset sizes, including
        occasional re-sampling of smaller sequence lengths for better training diversity.

        Parameters
        ----------
        min_seq_len : int, optional
            Minimum sequence length. If None, returns max_seq_len directly.

        max_seq_len : int
            Maximum sequence length

        log : bool, default=False
            If True, sample from a log-uniform distribution to better
            cover the range of possible sizes

        replay_small : bool, default=False
            If True, occasionally sample smaller sequence lengths with
            specific distributions to ensure model robustness on smaller datasets

        Returns
        -------
        int
            The sampled sequence length
        """
        if min_seq_len is None:
            return max_seq_len

        # Create a numpy random number generator from the torch generator for scipy/numpy operations
        seed = torch.empty((), dtype=torch.int64, device='cpu').random_(generator=generator).item()
        rng = np.random.default_rng(seed)

        if log:
            seq_len = int(loguniform.rvs(min_seq_len, max_seq_len, random_state=rng))
        else:
            seq_len = rng.integers(min_seq_len, max_seq_len)

        if replay_small:
            p = rng.random()
            if p < 0.05:
                return rng.integers(50, 150)
            elif p < 0.3:
                return int(loguniform.rvs(150, 500, random_state=rng))
            else:
                return seq_len
        else:
            return seq_len

    @staticmethod
    def adjust_max_features(seq_len: int, max_features: int) -> int:
        """
        Adjusts the maximum number of features based on the sequence length.

        This method implements an adaptive feature limit that scales inversely
        with sequence length. Longer sequences are restricted to fewer features
        to prevent memory issues and excessive computation times while still
        maintaining dataset diversity and learning difficulty.

        Parameters
        ----------
        seq_len : int
            Sequence length (number of samples)

        max_features : int
            Original maximum number of features

        Returns
        -------
        int
            Adjusted maximum number of features, ensuring computational feasibility
        """
        if seq_len <= 100:
            return min(512, max_features)
        elif 100 < seq_len <= 400:
            return min(256, max_features)
        elif 400 < seq_len <= 600:
            return min(128, max_features)
        elif 600 < seq_len <= 1000:
            return min(64, max_features)
        else:
            return 32

    @staticmethod
    def delete_unique_features(X: Tensor, d: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Removes features that have only one unique value across all samples.

        Single-value features provide no useful information for learning since they
        have zero variance. This method identifies and removes such constant features
        to improve model training efficiency and stability. The removed features are
        replaced with zero padding to maintain tensor dimensions.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape (B, T, H) where:
            - B is batch size
            - T is sequence length
            - H is feature dimensionality

        d : Tensor
            Number of features per dataset of shape (B,), indicating how many
            features are actually used in each dataset (rest is padding)

        Returns
        -------
        tuple
            (X_new, d_new) where:
            - X_new is the filtered tensor with non-informative features removed
            - d_new is the updated feature count per dataset
        """

        def filter_unique_features(xi: Tensor, di: int) -> Tuple[Tensor, Tensor]:
            """Filters features with only one unique value from a single dataset."""
            num_features = xi.shape[-1]
            # Only consider actual features (up to di, ignoring padding)
            xi = xi[:, :di]
            
            # Check for NaN and replace with 0 before unique check
            # This prevents NaN from being treated as multiple unique values
            xi_clean = torch.where(torch.isnan(xi), torch.zeros_like(xi), xi)
            
            # Identify features with more than one unique value (informative features)
            unique_mask = [len(torch.unique(xi_clean[:, j])) > 1 for j in range(di)]
            di_new = sum(unique_mask)
            
            # Create new tensor with only informative features, padding the rest
            # Use cleaned xi for output to ensure no NaN propagation
            xi_new = F.pad(xi_clean[:, unique_mask], pad=(0, num_features - di_new), mode="constant", value=0)
            return xi_new, torch.tensor(di_new, device=xi.device)

        # Process each dataset in the batch independently
        filtered_results = [filter_unique_features(xi, di) for xi, di in zip(X, d)]
        X_new, d_new = [torch.stack(res) for res in zip(*filtered_results)]

        return X_new, d_new

class StructuredPrior(Prior, ABC):
    """
    Abstract base class for structured dataset priors that use hierarchical parameter sampling
    and dataset generation. Provides common functionality for SCM-based and mixture-based priors.
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
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
        device: str = "cpu",
    ):
        super().__init__(
            batch_size=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            replay_small=replay_small,
        )
        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.seq_len_per_gp = seq_len_per_gp
        self.n_jobs = n_jobs
        self.num_threads_per_generate = num_threads_per_generate
        self.device = device

    @abstractmethod
    def hp_sampling(self) -> Dict[str, Any]:
        """Sample hyperparameters for dataset generation."""
        raise NotImplementedError

    @abstractmethod
    def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, ...]:
        """Generate a single dataset based on parameters."""
        raise NotImplementedError

    @abstractmethod
    def get_prior(self, generator: Optional[torch.Generator] = None) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def fixed_hp(self) -> Dict[str, Any]:
        """Fixed hyperparameters that must be defined by subclasses."""
        raise NotImplementedError

    
    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None, generator: Optional[torch.Generator] = None) -> Tuple[Tensor, ...]:
        """
        Generates a batch of datasets by first creating a parameter list and then processing it.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override. If None, uses self.batch_size

        Returns
        -------
        tuple
            Dataset tensors (X, y, d, seq_lens, num_classes, [optional extras])
        """
        batch_size = batch_size or self.batch_size

        # Calculate number of groups and subgroups
        size_per_gp = min(self.batch_size_per_gp, batch_size)
        num_gps = math.ceil(batch_size / size_per_gp)

        size_per_subgp = min(self.batch_size_per_subgp, size_per_gp)

        # Generate parameters list for all datasets, preserving group and subgroup structure
        param_list = []
        global_seq_len = None

        # Determine global seq_len if not per-group
        if not self.seq_len_per_gp:
            global_seq_len = self.sample_seq_len(
                self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small, generator=generator
            )
            
        # Generate parameters for each group
        for gp_idx in range(num_gps):
            # Determine actual size for this group (may be smaller for the last group)
            actual_gp_size = min(size_per_gp, batch_size - gp_idx * size_per_gp)
            if actual_gp_size <= 0:
                break

            # Create an independent numpy RNG for this group to ensure randomness independence
            gp_seed = torch.empty((), dtype=torch.int64, device='cpu').random_(generator=generator).item()
            gp_rng = np.random.default_rng(gp_seed)

            group_sampled_hp = self.hp_sampling(generator=generator)
            # If per-group, sample seq_len for this group. Otherwise, use global ones
            if self.seq_len_per_gp:
                gp_seq_len = self.sample_seq_len(
                    self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small, generator=generator
                )
                # Adjust max features based on seq_len for this group
                gp_max_features = self.adjust_max_features(gp_seq_len, self.max_features)
            else:
                gp_seq_len = global_seq_len
                gp_max_features = self.max_features

            # Calculate number of subgroups for this group
            num_subgps_in_gp = math.ceil(actual_gp_size / size_per_subgp)

            # Generate parameters for each subgroup
            for subgp_idx in range(num_subgps_in_gp):
                # Determine actual size for this subgroup
                actual_subgp_size = min(size_per_subgp, actual_gp_size - subgp_idx * size_per_subgp)
                if actual_subgp_size <= 0:
                    break

                # Subgroups share prior type, number of features, and sampled HPs
                subgp_prior_type = self.get_prior(generator=generator)
                # Use group-specific RNG instead of global RNG to ensure independence
                subgp_num_features = round(gp_rng.uniform(self.min_features, gp_max_features))
                
                # Evaluate hyperparameters: callable values are called, with context if they accept it
                # Other values remain unchanged from group-level sampling
                subgp_sampled_hp = {}
                for k, v in group_sampled_hp.items():
                    if callable(v):
                        # Check if the function accepts parameters (like meta_uniform)
                        sig = inspect.signature(v)
                        if len(sig.parameters) > 0:
                            # Function accepts parameters, pass context
                            subgp_sampled_hp[k] = v({"num_features": subgp_num_features})
                        else:
                            # Function doesn't accept parameters (like meta_choice_mixed)
                            subgp_sampled_hp[k] = v()
                    else:
                        subgp_sampled_hp[k] = v

                # Generate parameters for each dataset in this subgroup
                for ds_idx in range(actual_subgp_size):
                    # Each dataset has its own number of classes
                    # Use group-specific RNG instead of global RNG to ensure independence
                    if gp_rng.random() > 0.3:
                        ds_num_classes = gp_rng.integers(2, self.max_classes + 1)
                    else:
                        ds_num_classes = 2

                    # Create a new, independent generator for each dataset to ensure
                    # that parallel jobs receive unique generators.
                    ds_seed = torch.empty((), dtype=torch.int64, device='cpu').random_(generator=generator).item()
                    ds_generator = torch.Generator(device=self.device)
                    ds_generator.manual_seed(ds_seed)

                    # Create parameters dictionary for this dataset
                    params = {
                        **self.fixed_hp,  # Fixed HPs
                        "seq_len": gp_seq_len,
                        # If per-gp setting, use adjusted max features for this group because we use nested tensors
                        # If not per-gp setting, use global max features to fix size for concatenation
                        "max_features": gp_max_features if self.seq_len_per_gp else self.max_features,
                        **subgp_sampled_hp,  # sampled HPs for this group
                        "prior_type": subgp_prior_type,
                        "num_features": subgp_num_features,
                        "num_classes": ds_num_classes,
                        "device": self.device,
                        "generator": ds_generator, # Pass the new, independent generator
                    }
                    param_list.append(params)

        # Use joblib to generate datasets in parallel.
        # Note: the 'loky' backend does not support nested parallelism during DDP, whereas the 'threading' backend does.
        # However, 'threading' does not respect `inner_max_num_threads`.
        # Therefore, we stick with the 'loky' backend for parallelism, but this requires generating
        # the prior datasets separately from the training process and loading them from disk,
        # rather than generating them on-the-fly.
        if self.n_jobs > 1 and self.device == "cpu":
            with joblib.parallel_config(
                n_jobs=self.n_jobs, backend="loky", inner_max_num_threads=self.num_threads_per_generate
            ):
                results = joblib.Parallel()(joblib.delayed(self.generate_dataset)(params) for params in param_list)
        else:
            results = [self.generate_dataset(params) for params in param_list]

        # Process results, handling potentially mixed-length tuples
        X_list, y_list, d_list = [], [], []
        extra_list = []
        # Check if any result in the batch contains extra data.
        has_extra_data = any(len(r) == 4 for r in results)

        for result in results:
            if len(result) == 4:
                X, y, d, extra = result
                X_list.append(X)
                y_list.append(y)
                d_list.append(d)
                if has_extra_data:
                    extra_list.append(extra)
            elif len(result) == 3:
                X, y, d = result
                X_list.append(X)
                y_list.append(y)
                d_list.append(d)
                if has_extra_data:
                    # Use a placeholder with the same shape as X to maintain alignment
                    extra_list.append(torch.zeros_like(X))
            else:
                raise ValueError(f"Unexpected result length from generate_dataset: {len(result)}")

        # If no item in the batch had extra data, extra_list will be empty.
        if not has_extra_data:
            extra_list = None

        # Apply normalization to each dataset's features
        X_list = [normalize_data(x) for x in X_list]

        # Process labels: remap to contiguous [0, n-1] and compute actual num_classes
        # This handles cases where GMM fitting results in fewer classes than specified,
        # or when label values are not contiguous (e.g., [0, 2, 5] instead of [0, 1, 2])
        # Must be done BEFORE stacking/nesting tensors
        num_classes_list = []
        for i, y_i in enumerate(y_list):
            unique_labels, inverse_indices = torch.unique(y_i, return_inverse=True)
            # Remap labels to contiguous range [0, n-1]
            y_list[i] = inverse_indices
            # Report actual num_classes (trainer should handle num_classes < 2 cases)
            n_unique = len(unique_labels)
            num_classes_list.append(n_unique)

        # Combine Results
        if self.seq_len_per_gp:
            # Use nested tensors for variable sequence lengths
            X = nested_tensor([x.to(self.device) for x in X_list], device=self.device)
            y = nested_tensor([y.to(self.device) for y in y_list], device=self.device)
            if extra_list is not None:
                extra = nested_tensor([extra.to(self.device) for extra in extra_list], device=self.device)
        else:
            # Stack into regular tensors for fixed sequence length
            X = torch.stack(X_list).to(self.device)  # (B, T, H)
            y = torch.stack(y_list).to(self.device)  # (B, T)
            if extra_list is not None:
                extra = torch.stack(extra_list).to(self.device)  # (B, T) or (B, T, H)

        # Metadata (always regular tensors)
        d = torch.stack(d_list).to(self.device)  # Actual number of features after filtering
        seq_lens = torch.tensor([params["seq_len"] for params in param_list], device=self.device, dtype=torch.long)
        num_classes = torch.tensor(num_classes_list, device=self.device, dtype=torch.long)

        if extra_list is not None:
            return X, y, d, seq_lens, num_classes, extra
        else:
            return X, y, d, seq_lens, num_classes

class DummyPrior(Prior):
    """This class creates purely random data. This is useful for testing and debugging
    without the computational overhead of SCM-based generation.

    Parameters
    ----------
    batch_size : int, default=256
        Number of datasets to generate

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

    device : str, default="cpu"
        Computation device
    """

    def __init__(
        self,
        batch_size: int = 256,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        device: str = "cpu",
    ):
        super().__init__(
            batch_size=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
        )
        self.device = device

    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None, generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generates a batch of random datasets for testing purposes.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override, if None, uses self.batch_size

        Returns
        -------
        X : Tensor
            Features tensor of shape (batch_size, seq_len, max_features).
            Contains random Gaussian values for all features.

        y : Tensor
            Labels tensor of shape (batch_size, seq_len).
            Contains randomly assigned class labels.

        d : Tensor
            Number of features per dataset of shape (batch_size,).
            Always set to max_features for DummyPrior.

        seq_lens : Tensor
            Sequence length for each dataset of shape (batch_size,).
            All datasets share the same sequence length.

        num_classes : Tensor
            Number of classes of shape (batch_size,).
        """

        batch_size = batch_size or self.batch_size
        seq_len = self.sample_seq_len(self.min_seq_len, self.max_seq_len, log=self.log_seq_len, generator=generator)

        X = torch.randn(batch_size, seq_len, self.max_features, device=self.device, generator=generator)

        # Create a numpy random number generator from the torch generator for numpy operations
        seed = torch.empty((), dtype=torch.int64, device='cpu').random_(generator=generator).item()
        rng = np.random.default_rng(seed)
        expected_num_classes = rng.integers(2, self.max_classes + 1)
        y = torch.randint(0, expected_num_classes, (batch_size, seq_len), device=self.device, generator=generator)

        d = torch.full((batch_size,), self.max_features, device=self.device)
        seq_lens = torch.full((batch_size,), seq_len, device=self.device)
        # Use actual number of classes from y, not the expected number
        # For small seq_len, not all classes may appear in the generated data
        num_classes = torch.tensor([len(torch.unique(y[i])) for i in range(batch_size)], device=self.device)

        extra_data = torch.zeros_like(X)

        return X, y, d, seq_lens, num_classes, extra_data
