"""Learning rate scheduler and optimizer utilities."""

from __future__ import annotations

import torch.nn as nn
from torch import optim
from transformers import (
    get_constant_schedule,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)


import math
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from collections import deque


def _get_cosine_with_restarts_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int,
    amplitude_decay: float,
    lr_end: float = 0.0,
    lr_init: float = 1.0,
):
    """
    Compute the learning rate factor for a cosine schedule with warmup, hard restarts, and amplitude scaling.
    """
    if current_step < num_warmup_steps:
        # Warmup phase: Linearly increase learning rate
        return float(current_step) / float(max(1, num_warmup_steps))

    # After warmup: Apply cosine schedule with hard restarts and amplitude scaling
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return lr_end / lr_init  # as LambdaLR multiplies by lr_init

    # Determine which cycle the current step is in
    cycle_progress = (float(num_cycles) * progress) % 1.0
    current_cycle = int(float(num_cycles) * progress)
    amplitude = amplitude_decay**current_cycle  # Exponentially decay amplitude per cycle

    # Calculate the current learning rate with proper scaling
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
    current_lr = lr_end + (lr_init - lr_end) * cosine_factor * amplitude
    return current_lr / lr_init  # as LambdaLR multiplies by lr_init


def get_cosine_with_restarts(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    amplitude_decay: float = 1.0,
    lr_end: float = 0.0,
    last_epoch: int = -1,
):
    """
    Create a learning rate scheduler with warmup, cosine decay, hard restarts, and amplitude scaling.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        num_warmup_steps (int): Number of warmup steps.
        num_training_steps (int): Total number of training steps.
        num_cycles (int, optional): Number of hard restarts. Defaults to 1.
        amplitude_decay (float, optional): Factor to exponentially decay the max LR per cycle. Defaults to 1.0.
        lr_end (float, optional): Minimum learning rate at the end of each cycle. Defaults to 0.0.
        last_epoch (int, optional): The index of the last epoch. Defaults to -1.

    Returns:
        LambdaLR: A learning rate scheduler.
    """
    lr_init = optimizer.defaults["lr"]
    if lr_end > lr_init:
        raise ValueError(f"lr_end ({lr_end}) must be smaller than initial lr ({lr_init})")

    lr_lambda = partial(
        _get_cosine_with_restarts_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        amplitude_decay=amplitude_decay,
        lr_end=lr_end,
        lr_init=lr_init,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(config, optimizer):
    """Get the learning rate scheduler based on configuration."""

    if config.warmup_proportion >= 0:
        warmup_steps = config.max_steps * config.warmup_proportion
    else:
        warmup_steps = config.warmup_steps

    if config.scheduler == "constant":
        scheduler = get_constant_schedule(optimizer=optimizer)
    elif config.scheduler == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=config.max_steps
        )
    elif config.scheduler == "cosine_warmup":
        if hasattr(config, 'min_lr') and config.min_lr is not None:
            final_lr = config.min_lr
        else:
            final_lr = optimizer.defaults['lr'] * 0.1 

        scheduler = get_cosine_with_restarts(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=config.max_steps,
            num_cycles=1,
            amplitude_decay=1.0,
            lr_end=final_lr,
        )
    elif config.scheduler == "cosine_with_restarts":
        scheduler = get_cosine_with_restarts(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=config.max_steps,
            num_cycles=config.cosine_num_cycles,
            amplitude_decay=config.cosine_amplitude_decay,
            lr_end=config.cosine_lr_end,
        )
    elif config.scheduler == "polynomial_decay_warmup":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=config.max_steps,
            lr_end=config.poly_decay_lr_end,
            power=config.poly_decay_power,
        )
    else:
        raise NotImplementedError

    return scheduler

NORM_CLASSES = (
    nn.LayerNorm,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.GroupNorm,
)

def get_optimizer(config, model):
    """Get the optimizer based on configuration.
    
    Parameters
    ----------
    config : argparse.Namespace
        Configuration object containing optimizer parameters
    model : nn.Module
        The model whose parameters are to be optimized.
        
    Returns
    -------
    torch.optim.Optimizer or AdaptiveStageSGDWrapper
        The configured optimizer. For 'stagesgd', returns an AdaptiveStageSGDWrapper
        instance that wraps the SGD optimizer.
        
    Examples
    --------
    >>> optimizer = get_optimizer(config, model)
    >>> # For adaptive stage SGD:
    >>> config.optimizer = 'stagesgd'
    >>> optimizer = get_optimizer(config, model)
    """
    named_modules = dict(model.named_modules())

    decay_params = []
    no_decay_params = []

    for full_name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        module_name, _, param_name = full_name.rpartition('.')
        module = named_modules.get(module_name, model)

        if isinstance(module, NORM_CLASSES) or param_name.endswith("bias") or param.ndim == 1:
            no_decay_params.append(param)
            continue

        decay_params.append(param)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": config.weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    # Common parameters for all optimizers
    base_params = {
        'lr': config.lr,
    }
    
    optimizer_name = config.optimizer.lower()
    
    if optimizer_name == 'adamw':
        # AdamW optimizer (recommended for transformers)
        optimizer_params = {
            **base_params,
            'betas': (config.beta1, config.beta2),
            'eps': config.eps,
        }
        optimizer = optim.AdamW(optimizer_grouped_parameters, **optimizer_params)
        
    elif optimizer_name == 'adam':
        # Adam optimizer  
        optimizer_params = {
            **base_params,
            'betas': (config.beta1, config.beta2),
            'eps': config.eps,
        }
        optimizer = optim.Adam(optimizer_grouped_parameters, **optimizer_params)
        
    elif optimizer_name == 'sgd':
        # SGD optimizer
        optimizer_params = {
            **base_params,
            'momentum': config.momentum,
            'dampening': config.dampening,
            'nesterov': config.nesterov,
        }
        optimizer = optim.SGD(optimizer_grouped_parameters, **optimizer_params)
        
    elif optimizer_name == 'stagesgd':
        # Adaptive Stage SGD: SGD with automatic momentum reset and LR decay
        # First create the base SGD optimizer
        sgd_params = {
            **base_params,
            'momentum': config.momentum,
            'dampening': config.dampening,
            'nesterov': config.nesterov,
        }
        base_optimizer = optim.SGD(optimizer_grouped_parameters, **sgd_params)
        
        # Then wrap it with AdaptiveStageSGDWrapper
        # Note: scheduler will be set later via set_scheduler()
        optimizer = AdaptiveStageSGDWrapper(
            optimizer=base_optimizer,
            scheduler=None,  # Will be set after scheduler is created
            use_ema=getattr(config, 'adaptive_sgd_use_ema', True),
            ema_beta=getattr(config, 'adaptive_sgd_ema_beta', 0.99),
            loss_window=getattr(config, 'adaptive_sgd_loss_window', 100),
            oscillation_threshold=getattr(config, 'adaptive_sgd_oscillation_threshold', 0.5),
            lr_decay_factor=getattr(config, 'adaptive_sgd_lr_decay_factor', 0.5),
            min_lr=getattr(config, 'adaptive_sgd_min_lr', 1e-6),
            check_interval=getattr(config, 'adaptive_sgd_check_interval', 50),
            cooldown_steps=getattr(config, 'adaptive_sgd_cooldown_steps', 200),
            verbose=getattr(config, 'adaptive_sgd_verbose', True)
        )
        
    elif optimizer_name == 'rmsprop':
        # RMSprop optimizer
        optimizer_params = {
            **base_params,
            'alpha': config.rmsprop_alpha,
            'eps': config.eps,
            'momentum': config.momentum,
        }
        optimizer = optim.RMSprop(optimizer_grouped_parameters, **optimizer_params)
        
    elif optimizer_name == 'adagrad':
        # Adagrad optimizer
        optimizer_params = {
            **base_params,
            'eps': config.eps,
        }
        optimizer = optim.Adagrad(optimizer_grouped_parameters, **optimizer_params)
        
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. "
                        f"Supported optimizers: adamw, adam, sgd, stagesgd, rmsprop, adagrad")
    
    return optimizer


class AdaptiveStageSGDWrapper:
    """
    Adaptive Stage SGD optimizer wrapper.

    This wrapper monitors loss oscillation during training and, when sustained large
    oscillations are detected, automatically clears SGD momentum and decays the learning rate.

    Parameters
    ----------
    optimizer : torch.optim.SGD
        The underlying SGD optimizer to wrap.
    scheduler : torch.optim.lr_scheduler, optional
        Learning rate scheduler whose base_lrs will be synchronized after LR decay.
    use_ema : bool, default=True
        Whether to use exponential moving average (EMA) to measure oscillation.
        True: use EMA (recommended when training on generated / naturally noisy data).
        False: use a sliding loss window (better when loss itself is relatively stable).
    ema_beta : float, default=0.99
        EMA decay coefficient. Larger values give smoother estimates and more sensitivity
        to long-term trends. Recommended range: 0.98–0.995.
    loss_window : int, default=100
        Window size for monitoring loss (used only when use_ema=False).
    oscillation_threshold : float, default=0.5
        Oscillation threshold. When std(loss) / mean(loss) exceeds this value,
        the loss is considered oscillating.
    lr_decay_factor : float, default=0.5
        Learning rate decay factor. On each reset, lr is multiplied by this factor.
    min_lr : float, default=1e-6
        Minimum learning rate; lr will not decay below this value.
    check_interval : int, default=50
        Interval (in steps) between oscillation checks.
    cooldown_steps : int, default=200
        Cooldown steps after each reset before another reset can be triggered.
    verbose : bool, default=True
        If True, print detailed information when a reset occurs.

    Examples
    --------
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> scheduler = get_scheduler(config, optimizer)
    >>> # Use EMA-based monitoring (recommended for generated data)
    >>> adaptive_optimizer = AdaptiveStageSGDWrapper(
    ...     optimizer, scheduler,
    ...     use_ema=True,
    ...     ema_beta=0.99,
    ...     oscillation_threshold=0.5,
    ...     lr_decay_factor=0.5,
    ...     min_lr=1e-6
    ... )
    >>> # In the training loop
    >>> loss = compute_loss(...)
    >>> adaptive_optimizer.record_loss(loss.item())
    >>> if adaptive_optimizer.should_reset():
    ...     adaptive_optimizer.reset_momentum_and_lr()
    """
    
    def __init__(
        self,
        optimizer: optim.SGD,
        scheduler=None,
        use_ema: bool = True,
        ema_beta: float = 0.99,
        loss_window: int = 100,
        oscillation_threshold: float = 0.5,
        lr_decay_factor: float = 0.5,
        min_lr: float = 1e-6,
        check_interval: int = 50,
        cooldown_steps: int = 200,
        verbose: bool = True
    ):
        if not isinstance(optimizer, optim.SGD):
            raise ValueError("AdaptiveStageSGDWrapper only supports SGD optimizer")
        
        self.optimizer = optimizer
        self.scheduler = scheduler  # keep a reference for synchronizing base_lrs
        
        # Proxy common optimizer attributes so the wrapper can be used like an optimizer
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self.defaults = optimizer.defaults
        self.use_ema = use_ema
        self.ema_beta = ema_beta
        self.loss_window = loss_window
        self.oscillation_threshold = oscillation_threshold
        self.lr_decay_factor = lr_decay_factor
        self.min_lr = min_lr
        self.check_interval = check_interval
        self.cooldown_steps = cooldown_steps
        self.verbose = verbose
        
        if self.use_ema:
            self.ema_loss = None  # EMA of loss
            self.ema_loss_sq = None  # EMA of loss^2, for computing variance
        else:
            self.loss_history = deque(maxlen=loss_window)
        
        self.step_count = 0
        self.last_reset_step = -cooldown_steps  # negative so the first reset is allowed
        self.reset_count = 0
        
        self.initial_lr = optimizer.param_groups[0]['lr']
        
    def record_loss(self, loss: float):
        """Record the loss value for the current step."""
        if self.use_ema:
            # 使用EMA模式
            if self.ema_loss is None:
                # 第一次记录，初始化EMA
                self.ema_loss = loss
                self.ema_loss_sq = loss ** 2
            else:
                # 更新EMA
                self.ema_loss = self.ema_beta * self.ema_loss + (1 - self.ema_beta) * loss
                self.ema_loss_sq = self.ema_beta * self.ema_loss_sq + (1 - self.ema_beta) * (loss ** 2)
        else:
            # 使用滑动窗口模式
            self.loss_history.append(loss)
        
        self.step_count += 1
        
    def _calculate_oscillation_metric(self) -> float:
        """
        Compute the oscillation metric.
        
        In EMA mode, uses EMA-based mean and std.
        In sliding-window mode, uses mean and std within the window.
        
        Returns
        -------
        float
            Oscillation metric defined as std(loss) / mean(loss).
        """
        if self.use_ema:
            if self.ema_loss is None:
                return 0.0
            
            mean_loss = self.ema_loss
            variance = self.ema_loss_sq - (self.ema_loss ** 2)
            
            variance = max(0.0, variance)
            std_loss = np.sqrt(variance)
            
            if abs(mean_loss) < 1e-10:
                return 0.0
            
            oscillation_metric = std_loss / abs(mean_loss)
        else:
            if len(self.loss_history) < self.loss_window // 2:
                return 0.0
            
            losses = np.array(self.loss_history)
            mean_loss = np.mean(losses)
            std_loss = np.std(losses)
            
            if mean_loss < 1e-10:
                return 0.0
            
            oscillation_metric = std_loss / mean_loss
        
        return oscillation_metric
    
    def should_reset(self) -> bool:
        """
        Decide whether momentum and learning rate should be reset.
        
        Returns
        -------
        bool
            True if a reset should be triggered, otherwise False.
        """
        if self.step_count % self.check_interval != 0:
            return False
        
        if self.step_count - self.last_reset_step < self.cooldown_steps:
            return False
        
        if self.use_ema:
            if self.step_count < self.check_interval * 2:
                return False
        else:
            if len(self.loss_history) < self.loss_window // 2:
                return False
        
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr <= self.min_lr:
            return False
        
        oscillation_metric = self._calculate_oscillation_metric()
        
        return oscillation_metric > self.oscillation_threshold
    
    def reset_momentum_and_lr(self):
        """Clear SGD momentum and decay the learning rate."""
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                param_state = self.optimizer.state[p]
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].zero_()
        
        old_lr = self.optimizer.param_groups[0]['lr']
        new_lr = max(old_lr * self.lr_decay_factor, self.min_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        if self.scheduler is not None and hasattr(self.scheduler, 'base_lrs'):
            lr_scale = new_lr / old_lr
            self.scheduler.base_lrs = [base_lr * lr_scale for base_lr in self.scheduler.base_lrs]
            if self.verbose:
                print(f"  Synchronized scheduler base_lrs: {self.scheduler.base_lrs}")
        
        self.last_reset_step = self.step_count
        self.reset_count += 1
        
        if self.verbose:
            oscillation_metric = self._calculate_oscillation_metric()
            if self.use_ema:
                mean_loss = self.ema_loss
                std_loss = np.sqrt(max(0, self.ema_loss_sq - self.ema_loss ** 2))
            else:
                mean_loss = np.mean(self.loss_history)
                std_loss = oscillation_metric * mean_loss
            
            mode_str = "EMA" if self.use_ema else f"window({self.loss_window})"
            print(f"\n[AdaptiveSGD-{mode_str}] reset #{self.reset_count} at step {self.step_count}")
            print(f"  Oscillation metric: {oscillation_metric:.4f} (threshold: {self.oscillation_threshold})")
            print(f"  Mean loss: {mean_loss:.6f}, Std: {std_loss:.6f}")
            print(f"  Learning rate: {old_lr:.6e} -> {new_lr:.6e}")
            print(f"  Momentum buffers cleared\n")
    
    def get_stats(self) -> dict:
        """
        Get current statistics for logging and monitoring.
        
        Returns
        -------
        dict
            Dictionary containing the current adaptive SGD state.
        """
        oscillation_metric = self._calculate_oscillation_metric()
        
        if self.use_ema:
            mean_loss = self.ema_loss if self.ema_loss is not None else 0.0
        else:
            mean_loss = np.mean(self.loss_history) if len(self.loss_history) > 0 else 0.0
        
        steps_since_reset = self.step_count - self.last_reset_step
        
        return {
            'adaptive_sgd_step': self.step_count,
            'adaptive_sgd_reset_count': self.reset_count,
            'adaptive_sgd_oscillation_metric': oscillation_metric,
            'adaptive_sgd_mean_loss': mean_loss,
            'adaptive_sgd_steps_since_reset': steps_since_reset,
            'adaptive_sgd_current_lr': self.optimizer.param_groups[0]['lr'],
            'adaptive_sgd_use_ema': self.use_ema,
        }
    
    def set_scheduler(self, scheduler):
        """Set or update the scheduler reference (called after scheduler creation)."""
        self.scheduler = scheduler
    
    
    def step(self, closure=None):
        """Perform an optimization step (with optional automatic reset)."""
        if self.should_reset():
            self.reset_momentum_and_lr()
        
        return self.optimizer.step(closure)
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none)
    
    def state_dict(self) -> dict:
        """Return state dict including both optimizer and adaptive wrapper state."""
        base_state = {
            'use_ema': self.use_ema,
            'step_count': self.step_count,
            'last_reset_step': self.last_reset_step,
            'reset_count': self.reset_count,
            'initial_lr': self.initial_lr,
            'optimizer_state': self.optimizer.state_dict(), 
        }
        
        if self.use_ema:
            base_state['ema_loss'] = self.ema_loss
            base_state['ema_loss_sq'] = self.ema_loss_sq
        else:
            base_state['loss_history'] = list(self.loss_history)
        
        return base_state
    
    def load_state_dict(self, state_dict: dict):
        """Load state for both optimizer and adaptive wrapper."""
        if 'optimizer_state' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer_state'])
        
        self.step_count = state_dict['step_count']
        self.last_reset_step = state_dict['last_reset_step']
        self.reset_count = state_dict['reset_count']
        self.initial_lr = state_dict['initial_lr']
        
        use_ema = state_dict.get('use_ema', False)  # backward compatibility
        if use_ema and self.use_ema:
            # EMA模式
            self.ema_loss = state_dict.get('ema_loss')
            self.ema_loss_sq = state_dict.get('ema_loss_sq')
        elif not use_ema and not self.use_ema:
            self.loss_history = deque(state_dict.get('loss_history', []), maxlen=self.loss_window)
        else:
            # Mode mismatch: warn and fall back to default initialization
            print(
                f"[AdaptiveSGD] Warning: checkpoint used "
                f"{'EMA' if use_ema else 'window'} mode, but current config uses "
                f"{'EMA' if self.use_ema else 'window'} mode. "
                f"Adaptive statistics will be reinitialized."
            )
