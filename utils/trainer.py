from __future__ import annotations

import os
import sys
import logging
from typing import Tuple
import warnings
import functools
from contextlib import nullcontext
import json
import copy
import math
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.utils.data import DataLoader, get_worker_info
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .timer import Timer
from .utils import grad_stats, set_seed
from .checkpoints import get_latest_checkpoint
from .metrics import calculate_metrics
from utils.optim import get_scheduler, get_optimizer
from configs.train_config import HPARAM_KEYS, METRIC_KEYS_FOR_HPARAMS
from utils.prior_config_loader import load_prior_config
from utils.optim import AdaptiveStageSGDWrapper

warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)


def ddp_cleanup(func):
    """Decorator to clean up DDP process group after method execution.

    Ensures that destroy_process_group() is called if DDP is enabled,
    even if an exception occurs during method execution.
    Also closes rank log file if it exists.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            # 关闭rank日志文件
            if hasattr(self, 'rank_log_file_handle') and self.rank_log_file_handle is not None:
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    self.rank_log_file_handle.close()
                except Exception as e:
                    try:
                        original_stderr = sys.__stderr__
                        print(f"Warning: Failed to close rank log file: {e}", file=original_stderr)
                    except:
                        pass
            
            if self.ddp:
                destroy_process_group()

    return wrapper


class BaseTrainer(ABC):
    """This class handles the complete training lifecycle for TabCluster, including:

    - Environment setup and distributed training configuration
    - Model building and initialization
    - Optimizer, scheduler, and dataloader configuration
    - Checkpoint management and recovery
    - Training loop execution with gradient accumulation
    - Metrics tracking and logging using wandb
    - Test dataset evaluation during training

    Parameters
    ----------
    config : argparse.Namespace
        Training configuration parameters containing all settings for model,
        optimizer, distributed training, and data generation.
        
        Additional test evaluation parameters:
        - test_eval_every : int, optional
            Frequency of test evaluation (in training steps). If 0 or not set, 
            no test evaluation is performed.
        - test_num_batches : int, optional
            Number of test batches to evaluate each time. If None, evaluate 
            all available test batches.
    """

    # Subclasses should override this to specify their model class
    MODEL_CLASS = None

    def __init__(self, config):
        self.config = config
        self.configure_ddp()
        self._fixed_hp, self._sampled_hp, self.prior_config_source = load_prior_config(
            getattr(self.config, "prior_config", None)
        )
        if self.master_process:
            print(f"Using prior_config: {self.prior_config_source}")
        self.configure_logger()
        self.init_model()
        self.configure_optimizer()
        self.configure_amp()
        self.load_checkpoint()
        self.configure_prior()

    def configure_ddp(self):
        """Set up distributed training and system configuration.

        This method:
        1. Configures distributed data parallel (DDP) if enabled
        2. Sets up device and process information
        3. Adjusts batch size for multi-GPU training
        4. Sets random seeds for reproducibility
        """
        # Setup distributed training
        self.ddp = int(os.environ.get("RANK", -1)) != -1

        if self.ddp:
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.master_process = self.ddp_rank == 0
            self.config.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.config.device)

            # Adjust batch size for distributed training
            original_batch_size = self.config.batch_size
            self.config.batch_size = math.ceil(original_batch_size / self.ddp_world_size)

            if self.master_process:
                print(f"DDP training with {self.ddp_world_size} processes")
                if original_batch_size % self.ddp_world_size == 0:
                    print(f"Per-GPU batch size: {self.config.batch_size}")
                else:
                    self.config.effective_batch_size = self.config.batch_size * self.ddp_world_size
                    print(
                        f"Original batch size ({original_batch_size}) cannot be divided by world size ({self.ddp_world_size}).\n"
                        f"Use ceiling division for equal per-GPU batch size: {self.config.batch_size}.\n"
                        f"Effective batch size is {self.config.effective_batch_size}.\n"
                    )
        else:
            self.master_process = True
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.ddp_local_rank = 0
            print("No DDP training")

        self.curr_step = 0  # Initialize current step for training

        # Set random seeds
        seed_offset = self.ddp_rank if self.ddp else 0
        self.DDP_GLOBAL_SEED = self.config.torch_seed + seed_offset
        set_seed(self.config.np_seed + seed_offset, self.DDP_GLOBAL_SEED)
        # Use the new API for TF32 precision (PyTorch 2.0+)
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'

    def configure_logger(self):
        """Set up TensorBoard logging and configure version-based checkpoint directory."""
        version_num = 0
        if self.master_process:
            # Only master process determines the version number
            version_num = self._get_or_create_version()

        if self.ddp:
            # Broadcast the version number from master to all other processes
            version_tensor = torch.tensor([version_num], dtype=torch.int64, device=self.config.device)
            torch.distributed.broadcast(version_tensor, src=0)
            version_num = version_tensor.item()

        # All processes now have the same version number
        self.run_checkpoint_dir = os.path.join(self.config.checkpoint_dir, f"version_{version_num}")
        
        if self.ddp:
            rank_log_dir = os.path.join(self.run_checkpoint_dir, "rank_logs")
            os.makedirs(rank_log_dir, exist_ok=True)
            
            rank_log_file = os.path.join(rank_log_dir, f"rank_{self.ddp_rank}.log")
            
            self.rank_log_file_handle = open(rank_log_file, 'a', buffering=1)
            
            sys.stdout = self.rank_log_file_handle
            sys.stderr = self.rank_log_file_handle
            
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            
            logger.handlers.clear()
            
            file_handler = logging.FileHandler(rank_log_file, mode='a')
            file_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                f'[Rank {self.ddp_rank}] %(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
            
            print(f"[Rank {self.ddp_rank}] Rank log file: {rank_log_file}", flush=True)
            logging.info(f"Rank {self.ddp_rank} logging initialized")
        else:
            self.rank_log_file_handle = None
        
        # Master process creates directories and saves config
        if self.master_process:
            os.makedirs(self.run_checkpoint_dir, exist_ok=True)
            print(f"Experiment version: {version_num}")
            print(f"Checkpoints will be saved to: {self.run_checkpoint_dir}")

            # Save config to a json file
            base_config_path = os.path.join(self.run_checkpoint_dir, "config")
            config_path = f"{base_config_path}.json"
            counter = 1
            while os.path.exists(config_path):
                config_path = f"{base_config_path}_{counter}.json"
                counter += 1

            with open(config_path, "w") as f:
                json.dump(vars(self.config), f, indent=4)
            print(f"Configuration saved to: {config_path}")

            # Save prior_config to a separate json file, handling existing files
            base_prior_config_path = os.path.join(self.run_checkpoint_dir, "prior_config")
            prior_config_path = f"{base_prior_config_path}.json"
            counter = 1
            while os.path.exists(prior_config_path):
                prior_config_path = f"{base_prior_config_path}_{counter}.json"
                counter += 1

            # Create serializable copies of the configs, removing the problematic key
            fixed_hp_copy = self.get_fixed_hp()
            sampled_hp_copy = self.get_sampled_hp()

            # Remove the non-serializable 'mlp_activations' key before saving
            if "mlp_activations" in sampled_hp_copy:
                del sampled_hp_copy["mlp_activations"]
                print("Removed 'mlp_activations' from prior_config for JSON serialization.")

            # Remove the non-serializable 'min_fn' and 'max_fn' keys from MaxOmega before saving
            if "MaxOmega" in sampled_hp_copy and isinstance(sampled_hp_copy["MaxOmega"], dict):
                max_omega_copy = sampled_hp_copy["MaxOmega"].copy()
                if "min_fn" in max_omega_copy:
                    del max_omega_copy["min_fn"]
                if "max_fn" in max_omega_copy:
                    del max_omega_copy["max_fn"]
                sampled_hp_copy["MaxOmega"] = max_omega_copy
                print("Removed 'min_fn' and 'max_fn' from MaxOmega in prior_config for JSON serialization.")

            prior_config_dict = {
                "DEFAULT_FIXED_HP": fixed_hp_copy,
                "DEFAULT_SAMPLED_HP": sampled_hp_copy,
            }
            with open(prior_config_path, "w") as f:
                json.dump(prior_config_dict, f, indent=4)
            print(f"Prior configuration saved to: {prior_config_path}")


        if self.config.log_to_tensorboard and self.master_process:
            # Find a unique log directory for this run to avoid overwriting previous logs on resume.
            base_log_dir = os.path.join(self.run_checkpoint_dir, "tensorboard_logs")
            log_dir = base_log_dir
            counter = 1
            # If the base directory exists, find a new one by appending a counter.
            while os.path.exists(log_dir):
                log_dir = f"{base_log_dir}_{counter}"
                counter += 1
            
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logs will be saved to: {log_dir}")
        else:
            self.writer = None

    def _get_or_create_version(self):
        """
        Get or create version number.
        
        Returns
        -------
        int
            Version number
        """
        if self.config.version is not None:
            # Use specified version
            version_num = self.config.version
            version_dir = os.path.join(self.config.checkpoint_dir, f"version_{version_num}")
            
            if os.path.exists(version_dir):
                print(f"Resuming existing version {version_num}")
            else:
                print(f"Starting new run with specified version {version_num}")
            
            return version_num
        else:
            # Auto-increment version number
            version_num = self._get_next_version_number()
            print(f"Creating new version {version_num}")
            return version_num
    
    def _get_next_version_number(self):
        """Find the next available version number by scanning existing version directories."""
        if not os.path.exists(self.config.checkpoint_dir):
            return 0
        
        existing_versions = []
        for item in os.listdir(self.config.checkpoint_dir):
            if item.startswith("version_") and os.path.isdir(os.path.join(self.config.checkpoint_dir, item)):
                try:
                    version_num = int(item.replace("version_", ""))
                    existing_versions.append(version_num)
                except ValueError:
                    continue
        
        return max(existing_versions) + 1 if existing_versions else 0

    def init_model(self):
        """initialize the model and wrap with DDP if needed."""
        model = self.build_model() 

        if self.master_process:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model has {num_params} parameters.")

        # to(device)
        model.to(device=self.config.device)

        # Compile model if requested
        if self.config.model_compile:
            model = torch.compile(model, dynamic=True)
            if self.master_process:
                print("Model compiled successfully.")

        # DDP wrapping
        if self.ddp:
            # find_unused_parameters=True is needed when:
            # 1. Some parameters may not participate in loss in certain micro-batches
            # 2. k_prediction uses detached logits (by design), so col_embedder/row_interactor
            #    gradients only flow through the main cluster_learner path
            self.model = DDP(model, device_ids=[self.ddp_local_rank], broadcast_buffers=False)
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Constructs the model architecture."""
        raise NotImplementedError

    def configure_prior(self) -> None:
        """Configures the prior dataset and dataloader."""
        # 创建dataset的generator（注意：在worker中会被_worker_init_fn覆盖, num_workers=1）
        generator = torch.Generator().manual_seed(self.DDP_GLOBAL_SEED)
        dataset = self.build_dataset(generator=generator)
        if self.master_process:
            print(dataset)

        # 配置prior退火（如果启用）
        if hasattr(self.config, 'prior_anneal_proportion') and self.config.prior_anneal_proportion > 0:
            dataset.set_anneal_config(
                max_steps=self.config.max_steps,
                anneal_proportion=self.config.prior_anneal_proportion
            )
            
            # 设置初始步数（resume时恢复到curr_step，正常训练时为0）
            from prior.mix_prior import MixPrior
            if isinstance(dataset.prior, MixPrior):
                dataset.set_current_step(self.curr_step)
            
            if self.master_process:
                if self.curr_step > 0:
                    print(f"Prior annealing resumed at step {self.curr_step}")
                else:
                    print(f"Prior annealing enabled: {self.config.prior_anneal_proportion:.2%} of training")
                
                # 打印当前概率分布
                if isinstance(dataset.prior, MixPrior):
                    current_probas = dataset.prior.get_annealed_probas()
                    probas_str = ", ".join([f"{p:.3f}" for p in current_probas])
                    print(f"Current prior probas: [{probas_str}]")

        # Create dataloader for efficient loading and prefetching
        if self.curr_step > 0:
            large_prime = 2**31 - 1  
            base_offset = 1000000  
            loader_seed_offset = (self.curr_step * base_offset + self.curr_step) % large_prime
        else:
            loader_seed_offset = 0
        final_seed = self.DDP_GLOBAL_SEED + loader_seed_offset
        loader_generator = torch.Generator().manual_seed(final_seed) 
        self.dataloader = self.build_dataloader(dataset, loader_generator)
        
        self.dataset = dataset

    def get_fixed_hp(self):
        return copy.deepcopy(self._fixed_hp)

    def get_sampled_hp(self):
        return copy.deepcopy(self._sampled_hp)

    @abstractmethod
    def build_dataset(self, generator: torch.Generator):
        """Constructs the dataset for training."""
        raise NotImplementedError

    def build_dataloader(self, dataset, loader_generator: torch.Generator):
        # Create dataloader for efficient loading and prefetching
        self.num_workers = 1
        return DataLoader(
            dataset,
            batch_size=None,  # No additional batching since PriorDataset handles batching internally
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=self._worker_init_fn,
            prefetch_factor=4,
            pin_memory=True if self.config.prior_device == "cpu" else False,
            multiprocessing_context='fork',  # Use 'fork' to avoid /dev/shm issues in Docker
            generator=loader_generator,
        )
    
    def _worker_init_fn(self, worker_id):
        """
        Ensures each worker has a unique, deterministic, and independent random state
        by seeding it with a globally unique seed.
        """
        worker_info = get_worker_info()
        dataset = worker_info.dataset  # The dataset instance for this worker
        dataset.generator.manual_seed(worker_info.seed)

    def build_optimizer(self):
        """Configure optimizer and scheduler."""
        
        # Use the unified get_optimizer function
        optimizer = get_optimizer(config=self.config, model=self.raw_model)
        
        # For AdaptiveStageSGD, scheduler should use the underlying optimizer
        # to avoid TypeError (scheduler requires a torch.optim.Optimizer instance)
        if isinstance(optimizer, AdaptiveStageSGDWrapper):
            scheduler = get_scheduler(config=self.config, optimizer=optimizer.optimizer)
            optimizer.set_scheduler(scheduler)
        else:
            scheduler = get_scheduler(config=self.config, optimizer=optimizer)

        return optimizer, scheduler

    def configure_optimizer(self) -> None:
        """Configures the optimizer and learning rate scheduler."""
        self.optimizer, self.scheduler = self.build_optimizer()

    def _scheduler_step(self):
        if self.scheduler is None:
            return
        else:
            self.scheduler.step()

    def _get_current_lr(self):
        """Get the current learning rate from the optimizer."""
        # The learning rate is stored in the 'param_groups' of the optimizer
        if self.optimizer and self.optimizer.param_groups:
            return self.optimizer.param_groups[0]['lr']
        return 0.0

    def configure_amp(self):
        """Configure automatic mixed precision (AMP) for training."""

        self.amp = self.config.amp and "cuda" in self.config.device
        self.scaler = torch.GradScaler("cuda", enabled=self.amp)
        if self.amp:
            if self.master_process:
                print(f"Automatic Mixed Precision is enabled.")
            self.amp_ctx = torch.autocast(
                device_type="cuda", dtype=torch.float16 if self.config.dtype == "float16" else torch.float32
            )
        else:
            self.amp_ctx = nullcontext()

    def load_checkpoint(self):
        """Load model and training state from checkpoint.

        Handles three scenarios:
        1. Fresh Start: No checkpoint found.
        2. Resume (Default): Restores full state (Model, Opt, Sched, Step) to continue training.
        3. Phase Change (config.phase_change=True): Restores Model & Optimizer (for momentum),
           but resets Scheduler and Step to allow Re-warmup and new LR schedules.
        """
        # 1. Determine Checkpoint Path
        if (
            hasattr(self.config, "checkpoint_path")
            and self.config.checkpoint_path
            and self.config.version is not None
            and self.master_process
        ):
            warnings.warn(
                f"Both 'checkpoint_path' and 'version' are specified. "
                "This might lead to unexpected behavior."
            )

        checkpoint_path = None
        strict = True
        
        if hasattr(self.config, "checkpoint_path") and self.config.checkpoint_path:
            strict = False
            checkpoint_path = self.config.checkpoint_path
        else:
            # Default: load from current version's directory
            checkpoint_path = get_latest_checkpoint(self.run_checkpoint_dir)

        # --- Scenario 1: Fresh Start ---
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            if self.master_process:
                print("No checkpoint found, starting from scratch (Fresh Start).")
            return

        if self.master_process:
            print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        if "state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain model state")

        # Load Model Weights (Always happen if checkpoint exists)
        self.raw_model.load_state_dict(checkpoint["state_dict"], strict=strict)

        # --- Logic Branching based on Config ---
        
        # Flags handling
        only_load_model = getattr(self.config, "only_load_model", False)
        is_phase_change = getattr(self.config, "phase_change", False)

        if only_load_model:
            # --- Scenario: Transfer Learning (Weights Only) ---
            if self.master_process:
                print("Mode: Weights Only (only_load_model=True). Optimizer/Scheduler/Step NOT loaded.")
            # Do nothing else, start from step 0 with fresh optimizer
            return

        elif is_phase_change:
            # --- Scenario 3: Phase Change / Re-warmup ---
            if self.master_process:
                print("Mode: Phase Change (Preserving Optimizer Momentum, Resetting Scheduler).")

            # A. Load Optimizer State (Crucial: Keep v_t/m_t)
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            
            # B. Reset Optimizer LR (Crucial for Re-warmup)
            current_target_lr = self.config.lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_target_lr
                if 'initial_lr' in param_group:
                    param_group['initial_lr'] = current_target_lr
            
            self.curr_step = 0
            
            # D. Load Scaler (Optional but recommended to avoid initial overflow)
            ckpt_scaler_state = checkpoint.get("scaler_state", None)
            if self.scaler is not None and ckpt_scaler_state is not None:
                try:
                    self.scaler.load_state_dict(ckpt_scaler_state)
                except Exception as e:
                    print(f"Warning: failed to load scaler_state in phase change: {e}")

            if self.master_process:
                print(f"Phase Change prepared: Scheduler reset, Step set to {self.curr_step}, Optimizer state loaded.")

        else:
            # --- Scenario 2: Resume (Default) ---
            if self.master_process:
                print("Mode: Resume (Restoring Full State).")
            
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.curr_step = checkpoint["curr_step"]

            # Load Scheduler
            ckpt_sched_state = checkpoint.get("scheduler_state", None)
            if self.scheduler is not None and ckpt_sched_state is not None:
                try:
                    self.scheduler.load_state_dict(ckpt_sched_state)
                except Exception as e:
                    print(f"Warning: failed to load scheduler_state: {e}.")
            
            # Load Scaler
            ckpt_scaler_state = checkpoint.get("scaler_state", None)
            if self.scaler is not None and ckpt_scaler_state is not None:
                try:
                    self.scaler.load_state_dict(ckpt_scaler_state)
                except Exception as e:
                    print(f"Warning: failed to load scaler_state: {e}.")

            if self.master_process:
                print(f"Resumed training at step {self.curr_step}")
    
    def save_checkpoint(self, name: str):
        """Save model and training state to version-specific checkpoint file.

        Parameters
        ----------
        name : str
            Filename for the checkpoint
        """

        os.makedirs(self.run_checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.run_checkpoint_dir, name)
        checkpoint = {
            "config": self.model_config,
            "state_dict": self.raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "curr_step": self.curr_step,
        }

        if self.scheduler is not None:
            try:
                checkpoint["scheduler_state"] = self.scheduler.state_dict()
            except Exception as e:
                print(f"Warning: failed to serialize scheduler_state: {e}")

        # Save GradScaler state (important for AMP training continuity)
        if self.scaler is not None:
            try:
                checkpoint["scaler_state"] = self.scaler.state_dict()
            except Exception as e:
                print(f"Warning: failed to serialize scaler_state: {e}")

        torch.save(checkpoint, checkpoint_path)

    def manage_checkpoint(self):
        """
        Manages the number of temporary checkpoints by deleting the oldest ones
        if the count exceeds `max_checkpoints`. Permanent checkpoints are ignored.
        Uses version-specific checkpoint directory to avoid conflicts between experiments.
        """
        ckpt_dir = self.run_checkpoint_dir
        limit = self.config.max_checkpoints

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]
        temp_checkpoints = []
        for ckpt in checkpoints:
            try:
                step = int(ckpt.split("-")[1].split(".")[0])
                # Consider a checkpoint temporary if its step is not divisible by save_perm_every
                if step % self.config.save_perm_every != 0:
                    temp_checkpoints.append((step, ckpt))
            except:
                continue  # Ignore files that don't match the format

        # Sort temporary checkpoints by step number (ascending)
        temp_checkpoints.sort(key=lambda x: x[0])

        # Remove oldest temporary checkpoints if limit is exceeded
        num_to_delete = len(temp_checkpoints) - limit
        if num_to_delete > 0:
            for step, ckpt_name in temp_checkpoints[:num_to_delete]:
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                try:
                    os.remove(ckpt_path)
                except Exception as e:
                    print(f"Error removing checkpoint {ckpt_path}: {e}")

    @ddp_cleanup
    def train(self):
        """Main training loop.

        Iterates through batches, processes them, updates model parameters,
        and handles checkpoint saving and metric logging.
        """

        if self.master_process:
            step_progress = tqdm(total=self.config.max_steps - self.curr_step, desc="Step", leave=True, initial=0)
            step_iterator = range(self.curr_step, self.config.max_steps)
        else:
            step_progress = None
            step_iterator = range(self.curr_step, self.config.max_steps)

        dataloader = iter(self.dataloader)
        for step in step_iterator:
            
            # Get the next batch (uses the current step's probability)
            try:
                with Timer() as prior_timer:
                    batch = next(dataloader) 
            except StopIteration:
                dataloader = iter(self.dataloader) # TODO For finite datasets tests
                with Timer() as prior_timer:
                    batch = next(dataloader)

            prior_time = prior_timer.elapsed

            # Update current step AFTER generating batch (for logging and checkpointing)
            self.curr_step = step + 1

            # Train the model on the batch
            with Timer() as train_timer:
                results = self.run_batch(batch)
            train_time = train_timer.elapsed

            # Clear CUDA cache to free memory
            torch.cuda.empty_cache()

            if self.master_process:
                # Add timing information to results
                results.update({"prior_time": prior_time, "train_time": train_time})

                # Add extra log metrics
                extra_metrics = self.get_extra_log_metrics()
                results.update(extra_metrics)

                # Save checkpoints
                is_temp_save = self.curr_step % self.config.save_temp_every == 0
                is_perm_save = self.curr_step % self.config.save_perm_every == 0

                if is_temp_save or is_perm_save:
                    ckpt_name = f"step-{self.curr_step}.ckpt"
                    self.save_checkpoint(name=ckpt_name)

                    # Manage checkpoint limit only for temporary checkpoints
                    if is_temp_save and not is_perm_save and self.config.max_checkpoints > 0:
                        self.manage_checkpoint()
                

            # Logging to TensorBoard
            if self.writer is not None:
                # Add learning rate to results
                results["lr"] = self._get_current_lr()
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        if key.startswith("test_"):
                            # Remove "test_" prefix and put it under "test" tag
                            self.writer.add_scalar(f"test/{key[5:]}", value, self.curr_step)
                        else:
                            # Put other metrics under "train" tag
                            self.writer.add_scalar(f"train/{key}", value, self.curr_step)
            
            # Update progress bar with step and metrics
            if self.master_process and step_progress is not None:
                # First update the postfix with metrics
                step_progress.set_postfix(**{k: f"{v:.3f}" if isinstance(v, float) else v for k, v in results.items()}, refresh=False)
                # Then update the progress
                step_progress.update()

                # Log HParams to TensorBoard at the end of the run
        if self.master_process and self.writer is not None:
            hparams_to_log = {key: getattr(self.config, key, "N/A") for key in HPARAM_KEYS}
            
            # Filter for valid hparam types
            hparams_to_log = {
                k: v for k, v in hparams_to_log.items() 
                if isinstance(v, (int, float, str, bool))
            }

            final_metrics = {
                f"hparam/{key}": results.get(key)
                for key in METRIC_KEYS_FOR_HPARAMS
                if key in results and isinstance(results.get(key), (int, float))
            }

            try:
                self.writer.add_hparams(hparams_to_log, final_metrics)
                print("Hyperparameters logged to TensorBoard.")
            except Exception as e:
                print(f"Warning: Failed to log hyperparameters to TensorBoard: {e}")
                print("This may be due to a change in the TensorBoard API or an unsupported hparam value.")
                print("HParams to log:", hparams_to_log)
                print("Final metrics:", final_metrics)

        
    def get_extra_log_metrics(self) -> dict:
        """
        Returns a dictionary of extra metrics to be logged.
        This method can be overridden by subclasses to provide custom metrics.
        
        Returns
        -------
        dict
            A dictionary of metrics to be logged (e.g., {"gumbel_tau": 0.5}).
        """
        metrics = {}
        
        if hasattr(self, 'dataset'):
            from prior.mix_prior import MixPrior
            if isinstance(self.dataset.prior, MixPrior):
                batch_step = self.curr_step - 1 if self.curr_step > 0 else 0
                current_probas = self.dataset.prior.get_annealed_probas(step=batch_step)
                prior_types = self.dataset.prior.fixed_hp.get("mix_prior_types", ["gmires", "gm"])
                for i, prior_type in enumerate(prior_types):
                    if i < len(current_probas):
                        metrics[f"prior_prob_{prior_type}"] = current_probas[i]
        
        return metrics

    def validate_micro_batch(self, micro_seq_len):
        """
        Validate consistent sequence length within a micro batch.

        Ensures all datasets in a micro batch share the same sequence length, 
        required for efficient batch processing during gradient accumulation.

        Parameters
        ----------
        micro_seq_len : Tensor (micro_batch_size,)
            Sequence lengths for each dataset.

        Returns
        -------
        int
            The common seq_len for the micro batch.

        Raises
        ------
        ValueError
            If sequence lengths are inconsistent.
        """
        if len(torch.unique(micro_seq_len)) > 1:
            raise ValueError("All datasets in the micro batch must have the same sequence length.")

        seq_len = micro_seq_len[0].item()

        return seq_len

    def align_micro_batch(self, micro_X, micro_y, micro_d, seq_len, micro_extra_data):
        """
        Truncate micro batch tensors to required dimensions.

        Truncates sequence length and feature dimensions to the validated `seq_len`
        and the maximum active features (`micro_d.max()`) respectively. This optimizes
        memory and computation by removing unused tensor elements. Also handles
        truncation for tensors in `micro_extra_data`.

        Parameters
        ----------
        micro_X : Tensor (B, T, H)
            Input features per dataset.

        micro_y : Tensor (B, T)
            Target labels per dataset.

        micro_d : Tensor (B,)
            Number of active features per dataset.

        seq_len : int
            Validated sequence length for this micro batch.
            
        micro_extra_data : dict
            Dictionary of extra tensors.

        Returns
        -------
        tuple (Tensor, Tensor, dict)
            Truncated (micro_X, micro_y, micro_extra_data).
        """
        # Truncate sequence length
        if micro_X.shape[1] > seq_len:
            micro_X = micro_X[:, :seq_len]

        if micro_y.shape[1] > seq_len:
            micro_y = micro_y[:, :seq_len]
            
        # Also truncate tensors in extra_data that have a sequence dimension
        for key, value in micro_extra_data.items():
            if isinstance(value, torch.Tensor) and value.dim() > 1 and value.shape[1] > seq_len:
                micro_extra_data[key] = value[:, :seq_len]

        # Truncate feature dimension
        max_features = micro_d.max().item()
        if micro_X.shape[-1] > max_features:
            micro_X = micro_X[..., :max_features]

        return micro_X, micro_y, micro_extra_data

    @abstractmethod
    def compute_loss(self, 
                     micro_X: torch.Tensor, 
                     micro_y: torch.Tensor, 
                     micro_d: torch.Tensor,
                     micro_num_classes: torch.Tensor,
                     micro_extra_data: dict) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Configures the optimizer and learning rate scheduler."""
        raise NotImplementedError

    @staticmethod
    @torch.no_grad()
    @abstractmethod
    def predict_labels(model_output: torch.Tensor, micro_num_classes: torch.Tensor, micro_extra_data: dict = None, config=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert model output to predicted labels for evaluation.
        This method should be implemented by each trainer to define how
        the raw model output should be converted to final predicted labels.
        Different experiments may use different prediction strategies
        (e.g., argmax for multi-class, threshold for binary, etc.).
        
        Parameters
        ----------
        model_output : torch.Tensor
            Raw output from the model. Shape varies by model type:
            - For multi-class: (B, T, max_classes) with -inf for invalid classes
            - For ranking/scoring: (B, T, 1) or (B, T)
        micro_num_classes : torch.Tensor
            Number of valid classes for each sample in the batch, shape (B,).
            Used for masking invalid class predictions.
        micro_extra_data : dict, optional
            Additional data that might be needed for prediction
            (e.g., raw_scores for threshold-based prediction)
        config : argparse.Namespace, optional
            
        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            - pred: Predicted labels of shape (B, T) with integer class indices
            - predicted_num_classes: Number of unique predicted classes per sample, shape (B,)
        """
        raise NotImplementedError

    def parse_batch(self, batch: tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Parses a batch from the dataloader into standard components.

        Since PriorDataset and LoadPriorDataset have been unified to use the same format,
        this method now expects exactly 6 elements with extra_data as a dictionary.

        Parameters
        ----------
        batch : tuple
            The batch from the dataloader. Must be exactly 6 elements:
            (X, y, d, seq_lens, num_classes, extra_data)

        Returns
        -------
        tuple
            A tuple containing:
            - X: Features tensor
            - y: Labels tensor  
            - d: Number of active features per dataset
            - seq_lens: Sequence length for each dataset
            - num_classes: Number of classes for each dataset
            - extra_data: A dictionary for any additional data from the batch.

        Raises
        ------
        ValueError
            If batch length is not 6 or if extra_data is not a dictionary.
        """
        if len(batch) != 6:
            raise ValueError(f"Expected a batch of length 6: (X, y, d, seq_lens, num_classes, extra_data), "
                             f"but got {len(batch)}. Both PriorDataset and LoadPriorDataset should now return "
                             f"the unified 6-element format.")
        
        X, y, d, seq_lens, num_classes, extra_data = batch
        
        # Ensure extra_data is a dictionary
        if not isinstance(extra_data, dict):
            raise ValueError(f"Expected extra_data to be a dict, but got {type(extra_data)}. "
                             f"The unified format should always have a dict as the 6th element.")
        
        return X, y, d, seq_lens, num_classes, extra_data

    def run_micro_batch(self, micro_batch, micro_batch_idx, num_micro_batches):
        """Process a micro batch for gradient accumulation.

        Parameters
        ----------
        micro_batch : tuple
            (micro_X, micro_y, micro_d, micro_seq_len, micro_num_classes, micro_extra_data) tensors for the micro batch

        micro_batch_idx : int
            Index of the current micro batch

        num_micro_batches : int
            Total number of micro batches

        Returns
        -------
        dict
            Result dictionary
        """
        micro_X, micro_y, micro_d, micro_seq_len, micro_num_classes, micro_extra_data = micro_batch
        seq_len = self.validate_micro_batch(micro_seq_len)
        micro_X, micro_y, micro_extra_data = self.align_micro_batch(micro_X, micro_y, micro_d, seq_len, micro_extra_data)

        # Move to device
        micro_X = micro_X.to(self.config.device)
        micro_y = micro_y.to(self.config.device)
        micro_d = micro_d.to(self.config.device)
        micro_num_classes = micro_num_classes.to(self.config.device)
        for key, value in micro_extra_data.items():
            micro_extra_data[key] = value.to(self.config.device)

        if torch.isnan(micro_X).any() or torch.isnan(micro_y.float()).any():
            print(f"[Step {self.curr_step}] NaN detected in input data before forward pass")
            print(f"  micro_X has NaN: {torch.isnan(micro_X).any().item()}")
            print(f"  micro_y has NaN: {torch.isnan(micro_y.float()).any().item()}")

        param_has_nan = any(torch.isnan(p).any() for p in self.model.parameters())
        if param_has_nan:
            print(f"[Step {self.curr_step}] NaN detected in model parameters BEFORE forward pass")
            nan_params = [(name, torch.isnan(param).sum().item()) for name, param in self.model.named_parameters() if torch.isnan(param).any()]
            total_nan_elements = sum(count for _, count in nan_params)
            print(f"  {len(nan_params)} layers affected, {total_nan_elements} NaN parameters total")

        # Set DDP gradient sync for last micro batch only
        if self.ddp:
            self.model.require_backward_grad_sync = micro_batch_idx == num_micro_batches - 1

        with self.amp_ctx:
            loss, model_output, loss_dict = self.compute_loss(micro_X, micro_y, micro_d, micro_num_classes, micro_extra_data)
        
        if torch.isnan(loss).any():
            print(f"[Step {self.curr_step}] NaN detected in LOSS after forward pass")
            print(f"  Loss value: {loss.item()}")
        
        # Handle both single tensor and tuple of tensors for model_output
        if isinstance(model_output, tuple):
            if any(torch.is_tensor(t) and torch.isnan(t).any() for t in model_output):
                print(f"[Step {self.curr_step}] NaN detected in MODEL OUTPUT after forward pass")
                for i, t in enumerate(model_output):
                    if torch.is_tensor(t) and torch.isnan(t).any():
                        print(f"  Output tensor {i} has NaN: {torch.isnan(t).sum().item()}/{t.numel()} elements")
        elif torch.is_tensor(model_output) and torch.isnan(model_output).any():
            print(f"[Step {self.curr_step}] NaN detected in MODEL OUTPUT after forward pass")
            print(f"  Model output has NaN: {torch.isnan(model_output).sum().item()}/{model_output.numel()} elements")
        
        # Convert model output to predicted labels using the customizable method
        pred, predicted_num_classes = self.__class__.predict_labels(model_output, micro_num_classes, micro_extra_data, self.config)

        # Scale loss for gradient accumulation and backpropagate
        scaled_loss = loss / num_micro_batches
        
        if torch.isnan(scaled_loss).any():
            print(f"[Step {self.curr_step}] NaN detected in SCALED_LOSS before backward pass")
            print(f"  Scaled loss value: {scaled_loss.item()}")
        
        self.scaler.scale(scaled_loss).backward()

        with torch.no_grad():
            # Calculate metrics using the unified function
            # This includes num_classes_accuracy and num_classes_mae when pred_num_classes and true_num_classes are provided
            metrics = calculate_metrics(pred, micro_y, micro_seq_len, 
                                       pred_num_classes=predicted_num_classes, 
                                       true_num_classes=micro_num_classes)
            
            micro_results = {}
            # Store original loss for monitoring (will be averaged in run_batch)
            micro_results["loss"] = loss.item()
            # Add metrics (already averaged within micro_batch)
            # This includes: ari, accuracy, splitting_entropy, merging_entropy, 
            # splitting_perplexity, merging_perplexity, num_classes_accuracy, num_classes_mae
            micro_results.update(metrics)
            micro_results.update(loss_dict)

        return micro_results

    def run_batch(self, batch):
        """
        Trains the model on a batch of datasets. Handles gradient accumulation by
        splitting the batch into micro-batches. Supports variable-sized datasets
        by padding. Skips micro-batches on CUDA OOM errors. Updates model
        parameters and returns loss and accuracy metrics.

        Parameters
        ----------
        batch: tuple
            Contains tensors for the batch, as returned by the dataloader.

        Returns
        ------
        dict
            Dictionary containing 'loss' and any other metrics from the model.

        Raises
        ------
        RuntimeError
            If more than 10% of micro-batches fail due to OOM errors.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Parse batch data using the (potentially overridden) method
        X, y, d, seq_lens, num_classes, extra_data = self.parse_batch(batch)

        # Pad nested tensors to the same size
        padded_batch_main = [t.to_padded_tensor(padding=0.0) if hasattr(t, 'is_nested') and t.is_nested else t for t in [X, y, d, seq_lens, num_classes]]
        
        # Also pad extra data if they are nested tensors
        for key, value in extra_data.items():
            if hasattr(value, 'is_nested') and value.is_nested:
                extra_data[key] = value.to_padded_tensor(padding=0.0)

        # Get actual batch size from the first tensor (after padding)
        actual_batch_size = padded_batch_main[0].shape[0]
        
        # Split the batch into micro-batches along the first dimension
        # Use actual batch size instead of configured batch size to avoid index errors
        num_micro_batches = math.ceil(actual_batch_size / self.config.micro_batch_size)
        micro_batches_main = [torch.split(t, self.config.micro_batch_size, dim=0) for t in padded_batch_main]
        
        # Split extra data as well
        micro_batches_extra = {key: torch.split(value, self.config.micro_batch_size, dim=0) for key, value in extra_data.items()}

        # Zip main and extra micro-batches together
        micro_batches = []
        for i in range(num_micro_batches):
            main_data = [mb[i] for mb in micro_batches_main]
            extra_dict = {key: mb[i] for key, mb in micro_batches_extra.items()}
            micro_batches.append((*main_data, extra_dict))

        results = {}
        failed_batches = 0
        successful_batches = 0
        
        # Add loss stage information
        loss_stage = self.get_loss_stage()
        results["loss_stage"] = loss_stage

        for idx, micro_batch in enumerate(micro_batches):
            try:
                # Pass the original num_micro_batches for DDP sync, but track successful ones separately
                micro_results = self.run_micro_batch(micro_batch, idx, num_micro_batches)
                for k, v in micro_results.items():
                    results[k] = results.get(k, 0.0) + v
                successful_batches += 1
            except torch.cuda.OutOfMemoryError:
                print(
                    f"Warning: OOM error in micro-batch {idx+1}/{num_micro_batches} at step {self.curr_step}. Skipping."
                )
                torch.cuda.empty_cache()
                failed_batches += 1
                continue

        failure_ratio = failed_batches / num_micro_batches if num_micro_batches > 0 else 0
        if failure_ratio > 0.1:
            raise RuntimeError(
                f"({failure_ratio:.1%}) of micro-batches failed due to OOM at step {self.curr_step}. "
                f"Please check configuration to reduce memory consumption."
            )

        # Adjust metrics based on successful batches only
        if successful_batches > 0:
            # Average all accumulated values over successful batches (except loss_stage)
            for key in results:
                if key != "loss_stage":
                    results[key] /= successful_batches
            
            if successful_batches != num_micro_batches:
                print(f"[Step {self.curr_step}] Metrics computed from {successful_batches}/{num_micro_batches} successful micro-batches")
        elif num_micro_batches > 0:
            print(f"[Step {self.curr_step}] Warning: All {num_micro_batches} micro-batches failed!")
            # Reset results to prevent invalid metrics, keeping loss_stage
            for key in list(results.keys()):
                if key != "loss_stage":
                    results[key] = 0.0

        if self.config.gradient_clipping > 0 or self.config.grad_stats:
            self.scaler.unscale_(self.optimizer)
        
        # NaN检查5: 梯度累加完成后的梯度检查
        grad_has_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in self.model.parameters())
        if grad_has_nan:
            print(f"[Step {self.curr_step}] NaN detected in GRADIENTS after all micro-batches")
            nan_grads = [(name, torch.isnan(param.grad).sum().item()) for name, param in self.model.named_parameters() if param.grad is not None and torch.isnan(param.grad).any()]
            total_nan_grads = sum(count for _, count in nan_grads)
            print(f"  {len(nan_grads)} layers affected, {total_nan_grads} NaN gradients total")
            print(f"  Skipping optimizer step due to NaN gradients")
            
            # Mimic GradScaler behavior: update scaler first (while gradients still exist)
            self.scaler.update()  # Update scaler state based on current gradients
            # Then clear gradients (same as what GradScaler.step() would do when skipping)
            self.optimizer.zero_grad(set_to_none=True)
            # Update scheduler (consistent with GradScaler's behavior - scheduler still runs)
            self._scheduler_step()
            
            return results
        
        # Clip the gradient
        if self.config.gradient_clipping > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
        
        if self.config.grad_stats:
            grad_info = grad_stats(self.raw_model)
            results.update(grad_info)

        # Adaptive SGD: record loss BEFORE optimizer step
        # (step() will automatically check and reset if needed)
        
        if isinstance(self.optimizer, AdaptiveStageSGDWrapper):
            self.optimizer.record_loss(results['loss'])
        
        param_has_nan_before = any(torch.isnan(p).any() for p in self.model.parameters())
        if param_has_nan_before:
            print(f"[Step {self.curr_step}] NaN detected in model parameters BEFORE optimizer step")
            nan_count = sum(torch.isnan(p).sum().item() for p in self.model.parameters())
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  NaN parameters: {nan_count}/{total_params} ({nan_count/total_params*100:.2f}%)")

        # Update parameters
        # For AdaptiveStageSGD, step() will automatically check and reset before updating
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        param_has_nan_after = any(torch.isnan(p).any() for p in self.model.parameters())
        if param_has_nan_after and not param_has_nan_before:
            print(f"[Step {self.curr_step}] NaN introduced in model parameters DURING optimizer step")
            nan_params_after = [(name, torch.isnan(param).sum().item()) for name, param in self.model.named_parameters() if torch.isnan(param).any()]
            total_nan_after = sum(count for _, count in nan_params_after)
            print(f"  {len(nan_params_after)} layers newly corrupted, {total_nan_after} NaN parameters total")
        elif param_has_nan_after:
            print(f"[Step {self.curr_step}] NaN persists in model parameters AFTER optimizer step")

        # Adaptive SGD: add stats to results (every 10 steps to reduce overhead)
        if isinstance(self.optimizer, AdaptiveStageSGDWrapper):
            if self.curr_step % 10 == 0:
                stats = self.optimizer.get_stats()
                results.update(stats)
        
        # Update the learning rate
        self.optimizer.zero_grad(set_to_none=True)
        self._scheduler_step()

        return results

    def get_loss_stage(self):
        """
        Determine which loss stage we're in based on current training progress.
        Can be overridden by subclasses to implement custom loss scheduling.
        
        Returns
        -------
        str
            Loss stage identifier (e.g., 'warmup', 'main', 'stage1', 'stage2', etc.)
        """
        if hasattr(self.config, 'warmup_loss_proportion') and self.config.warmup_loss_proportion > 0:
            warmup_steps = int(self.config.max_steps * self.config.warmup_loss_proportion)
            if self.curr_step < warmup_steps:
                return 'warmup'
            else:
                return 'main'
        return 'main'
    
