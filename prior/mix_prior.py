"""
Mixed Prior class: Mixes SCMPrior and GaussianMixturePrior dataset generation methods by proportion.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from typing import Dict, Tuple, Optional, Any
from multiprocessing import Value
import ctypes

from .base import StructuredPrior
from .hp_sampling import HpSamplerList
from .gaussian_mixture_prior import GaussianMixturePrior
from .gmires_prior import GMIResPrior
from configs.prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP

class MixPrior(StructuredPrior):
    """
    Mixes SCMPrior and GaussianMixturePrior dataset generation methods by proportion.
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
    ):
        super().__init__(
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
            n_jobs=n_jobs,
            num_threads_per_generate=num_threads_per_generate,
            device=device,
        )
        self._fixed_hp = fixed_hp
        self._sampled_hp = sampled_hp
        self.prior_type = prior_type
        # Prior annealing related - use shared memory to support multi-process DataLoader
        # Use multiprocessing.Value to create variables shareable across processes
        self._shared_current_step = Value(ctypes.c_int, 0)
        self.max_steps = None
        self.anneal_proportion = 0.0
        # Pre-initialize GaussianMixturePrior instance
        self.gm_prior = GaussianMixturePrior(
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
            fixed_hp=fixed_hp,
            sampled_hp=sampled_hp,
            n_jobs=n_jobs,
            num_threads_per_generate=num_threads_per_generate,
            device=device,
        )
        self.gmires_prior = GMIResPrior(
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
            fixed_hp=fixed_hp,
            sampled_hp=sampled_hp,
            n_jobs=n_jobs,
            num_threads_per_generate=num_threads_per_generate,
            device=device,
        )

    @property
    def fixed_hp(self) -> Dict[str, Any]:
        return self._fixed_hp

    @property
    def sampled_hp(self) -> Dict[str, Any]:
        return self._sampled_hp

    def set_anneal_config(self, max_steps: int, anneal_proportion: float):
        """
        Set prior annealing configuration parameters.
        
        Parameters
        ----------
        max_steps : int
            Total training steps
        anneal_proportion : float
            Proportion of annealing duration (0.0-1.0). If 0, no annealing is performed.
        """
        self.max_steps = max_steps
        self.anneal_proportion = anneal_proportion

    def set_current_step(self, step: int):
        """
        Set current training step for calculating annealing progress.
        Uses shared memory to ensure worker processes in multi-process DataLoader can read correct step.
        
        Parameters
        ----------
        step : int
            Current training step
        """
        with self._shared_current_step.get_lock():
            self._shared_current_step.value = step
    
    def get_current_step(self) -> int:
        """
        Get current training step.
        
        Returns
        -------
        int
            Current training step
        """
        with self._shared_current_step.get_lock():
            return self._shared_current_step.value
    
    def increment_step(self) -> None:
        """
        Atomically increment current step.
        Automatically called when generating each batch to ensure step matches actual generated batches.
        """
        with self._shared_current_step.get_lock():
            self._shared_current_step.value += 1

    def get_annealed_probas(self, step: Optional[int] = None) -> list:
        """
        Calculate annealed prior type probabilities based on training progress.
        
        Parameters
        ----------
        step : int, optional
            Specified training step. If None, read current step from shared memory.
        
        Returns
        -------
        list
            Annealed probability distribution
        """
        # If not using annealing, return fixed probabilities
        if self.anneal_proportion <= 0.0 or self.max_steps is None:
            return self.fixed_hp.get("mix_prior_probas", [0.33, 0.33, 0.34])
        
        # Get step
        if step is None:
            current_step = self.get_current_step()
        else:
            current_step = step
        
        # Calculate annealing progress (0.0 to 1.0)
        anneal_steps = int(self.max_steps * self.anneal_proportion)
        if current_step >= anneal_steps:
            # Annealing finished, use final probabilities
            progress = 1.0
        else:
            # During annealing, linear interpolation
            progress = current_step / anneal_steps if anneal_steps > 0 else 1.0
        
        # Get initial and final probabilities
        probas_initial = self.fixed_hp.get("mix_prior_probas_initial", [0.05, 0.90, 0.05])
        probas_final = self.fixed_hp.get("mix_prior_probas_final", [0.3, 0.2, 0.5])
        
        # Linear interpolation to calculate current probabilities
        probas = []
        for p_init, p_final in zip(probas_initial, probas_final):
            p = p_init + (p_final - p_init) * progress
            probas.append(p)
        
        # Normalize to ensure sum is 1 (handle floating point precision issues)
        total = sum(probas)
        if total > 0:
            probas = [p / total for p in probas]
        
        return probas

    def hp_sampling(self, generator: Optional[torch.Generator] = None) -> Dict[str, Any]:
        """
        Sample hyperparameters for dataset generation.
        Returns
        -------
        dict
            Dictionary with sampled hyperparameters merged with fixed ones
        """
        hp_sampler = HpSamplerList(self.sampled_hp, device=self.device, generator=generator)
        return hp_sampler.sample()

    @torch.no_grad()
    def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generate dataset by selecting appropriate prior based on type returned by get_prior().
        """
        prior_type = params["prior_type"]
        if prior_type == "gm":
            X, y, d = self.gm_prior.generate_dataset(params)
        elif prior_type == "gmires":
            X, y, d = self.gmires_prior.generate_dataset(params)
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")
        return X, y, d

    def get_prior(self, generator: Optional[torch.Generator] = None) -> str:
        """
        Randomly select prior type, similar to SCMPrior implementation.
        Prefer getting types and probabilities from fixed_hp['mix_prior_types'] and fixed_hp['mix_prior_probas'].
        If annealing is configured, use annealed probabilities.
        """
        prior_type = self.prior_type
        if prior_type == "mix_prior":
            types = self.fixed_hp.get("mix_prior_types", ["gm", "gmires"])
            # Use annealed probabilities (if not using annealing, get_annealed_probas returns fixed probabilities)
            probas = self.get_annealed_probas()
            seed = torch.empty((), dtype=torch.int64, device='cpu').random_(generator=generator).item()
            rng = np.random.default_rng(seed)
            prior_type =  rng.choice(types, p=probas)
    
        return prior_type
