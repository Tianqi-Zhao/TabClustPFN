from __future__ import annotations
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

import torch
import torch.nn.functional as F
from torch.multiprocessing import set_start_method
from functools import partial

from utils.trainer import BaseTrainer
from model.tabcluster import TabClusterIMAB
from datasets.datasets import PriorDataset
from utils.anneal import annealed_params
from configs.train_config import build_parser


def soft_ari(logits, labels, num_classes, eps=1e-9, prob_method='softmax', gumbel_tau=1.0, gumbel_hard=False):
    """
    Compute soft ARI loss for a single dataset.
    
    Parameters
    ----------
    logits : torch.Tensor
        Model predictions of shape (seq_len, max_classes). When num_classes < max_classes,
        the invalid logits (indices >= num_classes) should be -inf.
    labels : torch.Tensor
        Ground truth labels of shape (seq_len,)
    num_classes : int
        Number of valid classes for this dataset (kept for API compatibility, not used in computation
        since invalid logits are already -inf and labels are in [0, num_classes-1])
    eps : float
        Small constant to avoid division by zero
    prob_method : str
        Method for computing probabilities ('softmax' or 'gumbel_softmax')
    gumbel_tau : float
        Temperature for Gumbel-Softmax
    gumbel_hard : bool
        Whether to use hard sampling in Gumbel-Softmax
        
    Returns
    -------
    torch.Tensor
        Soft ARI loss (scalar)
        
    Notes
    -----
    This function uses fixed max_classes dimension instead of dynamic K indexing,
    which is required for vmap compatibility. Since:
    - Invalid logits are -inf, so softmax gives 0 probability
    - Labels are in [0, K-1], so one-hot Y[:, K:] = 0
    The invalid classes contribute 0 to all ARI terms, giving correct results.
    """
    N, max_K = logits.shape
    
    # Compute softmax on full logits (invalid -inf positions become 0 probability)
    if prob_method == 'gumbel_softmax':
        P = F.gumbel_softmax(logits, tau=gumbel_tau, hard=gumbel_hard, dim=-1)  # (T, max_K)
    else:  # default to softmax
        P = F.softmax(logits, dim=-1)  # (T, max_K)
    
    # One-hot with max_K classes (labels in [0, K-1] means Y[:, K:] = 0)
    Y = F.one_hot(labels.long(), num_classes=max_K).float()  # (T, max_K)
    
    # Confusion matrix: invalid rows/cols are 0 since P[:, i>=K] = 0 and Y[:, j>=K] = 0
    M = torch.mm(P.T, Y)  # (max_K, max_K)
    
    def C2(x): return x * (x - 1.0) / 2.0
    
    Index = C2(M).sum()
    a = M.sum(dim=1)  # pred cluster sizes [max_K], invalid entries are 0
    b = Y.sum(dim=0)  # true cluster sizes [max_K], invalid entries are 0
    SumRow = C2(a).sum()
    SumCol = C2(b).sum()
    C2N = C2(N)
    
    expected = (SumRow * SumCol) / (C2N + eps)
    denom = 0.5 * (SumRow + SumCol) - expected
    denom = torch.clamp(denom, min=1e-6)
    
    ari_soft = (Index - expected) / denom
    # Clamp ari_soft to reasonable range [-1, 1] to prevent extreme loss values
    ari_soft = torch.clamp(ari_soft, min=-1.0, max=1.0)
    loss = 1. - ari_soft
    
    return loss


def soft_ari_loss(logits_batch, labels_batch, num_classes_batch, eps=1e-9, prob_method='softmax', gumbel_tau=1.0, gumbel_hard=False):
    """
    Compute soft ARI loss for a batch of datasets using vmap.
    
    Parameters
    ----------
    logits_batch : torch.Tensor
        Model predictions of shape (B, seq_len, max_classes), already gated
    labels_batch : torch.Tensor
        Ground truth labels of shape (B, seq_len)
    num_classes_batch : torch.Tensor
        Number of classes for each dataset, shape (B,) (not used, kept for compatibility)
    eps : float
        Small constant to avoid division by zero
    prob_method : str
        Method for computing probabilities ('softmax' or 'gumbel_softmax')
    gumbel_tau : float
        Temperature for Gumbel-Softmax
    gumbel_hard : bool
        Whether to use hard sampling in Gumbel-Softmax
        
    Returns
    -------
    torch.Tensor
        Soft ARI losses of shape (B,)
    """
    func_with_args = partial(soft_ari, eps=eps, prob_method=prob_method, gumbel_tau=gumbel_tau, gumbel_hard=gumbel_hard)
    loss_fn = torch.vmap(func_with_args, in_dims=(0, 0, 0), randomness='different')
    losses = loss_fn(logits_batch, labels_batch, num_classes_batch)  # (B,)
    return losses


class TrainerProb(BaseTrainer):
    """This class handles the complete training lifecycle for TabCluster, including:

    - Environment setup and distributed training configuration
    - Model building and initialization
    - Optimizer, scheduler, and dataloader configuration
    - Checkpoint management and recovery
    - Training loop execution with gradient accumulation
    - Metrics tracking and logging using wandb

    Parameters
    ----------
    config : argparse.Namespace
        Training configuration parameters containing all settings for model,
        optimizer, distributed training, and data generation.
    """

    MODEL_CLASS = TabClusterIMAB

    def build_model(self):
        """Build and initialize the TabCluster model."""

        self.model_config = {
            "max_classes": self.config.max_classes,
            "embed_dim": self.config.embed_dim,
            "col_num_blocks": self.config.col_num_blocks,
            "col_nhead": self.config.col_nhead,
            "col_num_inds": self.config.col_num_inds,
            "row_num_blocks": self.config.row_num_blocks,
            "row_nhead": self.config.row_nhead,
            "row_num_cls": self.config.row_num_cls,
            "row_rope_base": self.config.row_rope_base,
            "cluster_num_blocks": getattr(self.config, "cluster_num_blocks", 6),
            "cluster_nhead": getattr(self.config, "cluster_nhead", 4),
            "cluster_use_rope_cross_attn": getattr(self.config, "cluster_use_rope_cross_attn", False),
            "cluster_rope_base": getattr(self.config, "cluster_rope_base", 100000),
            "cluster_use_representation_self_att": getattr(self.config, "cluster_use_representation_self_att", False),
            "ff_factor": self.config.ff_factor,
            "dropout": self.config.dropout,
            "activation": self.config.activation,
            "norm_first": self.config.norm_first,
        }

        model = self.MODEL_CLASS(**self.model_config)

        if getattr(self.config, "freeze_col", False):
            model.col_embedder.eval()
            for p in model.col_embedder.parameters():
                p.requires_grad = False
        if getattr(self.config, "freeze_row", False):
            model.row_interactor.eval()
            for p in model.row_interactor.parameters():
                p.requires_grad = False
        if getattr(self.config, "freeze_cluster", False):
            model.cluster_learner.eval()
            for p in model.cluster_learner.parameters():
                p.requires_grad = False

        return model

    def build_dataset(self, generator: torch.Generator):
        """
        Sets up a tabular dataset generator that creates synthetic datasets
        during training with controllable properties and data distributions.
        """
        # Generate prior data on the fly
        dataset = PriorDataset(
            generator=generator,
            batch_size=self.config.batch_size,
            batch_size_per_gp=self.config.batch_size_per_gp,
            min_features=self.config.min_features,
            max_features=self.config.max_features,
            max_classes=self.config.max_classes,
            min_seq_len=self.config.min_seq_len,
            max_seq_len=self.config.max_seq_len,
            log_seq_len=self.config.log_seq_len,
            seq_len_per_gp=self.config.seq_len_per_gp,
            replay_small=self.config.replay_small,
            prior_type=self.config.prior_type,
            fixed_hp=self.get_fixed_hp(),
            sampled_hp=self.get_sampled_hp(),
            device=self.config.prior_device,
            n_jobs=1,  # Set to 1 to avoid nested parallelism during DDP
        )

        return dataset

    def get_extra_log_metrics(self) -> dict:
        """Calculate and return annealed gumbel_tau for logging."""
        metrics = super().get_extra_log_metrics()
        
        if getattr(self.config, "prob_method", 'softmax') == "gumbel_softmax":
            gumbel_tau = annealed_params(
                getattr(self.config, "gumbel_anneal_proportion", 0.0),
                getattr(self.config, "gumbel_tau_initial", 1.0),
                getattr(self.config, "gumbel_tau_final", 0.1),
                self.curr_step,
                self.config.max_steps
            )
            metrics["gumbel_tau"] = gumbel_tau
        
        return metrics

    def compute_loss(self, micro_X, micro_y, micro_d, micro_num_classes, micro_extra_data=None):
        """
        Computes the loss for a micro-batch of data.
        
        When freeze_col, freeze_row, and freeze_cluster are all True, only trains
        k_prediction_mlp and computes k_decision_loss, skipping soft_ari_loss.
        
        Parameters
        ----------
        micro_X: (B, T, D) tensor of input features
        micro_y: (B, T) int labels in [0, num_classes - 1]
        micro_d: (B,) vaild dim of input data 
        micro_num_classes: (B,) int tensor with number of classes for each sample in the batch
        """
        
        # Model returns (logits, k_logits, all_k_logits, query_norm, key_norm, all_query_norms, all_key_norms) when predict_k=True and return_hidden=True
        logits, k_logits, all_k_logits = self.model(
            micro_X, micro_d, num_classes=micro_num_classes, 
            predict_k=True
        )

        # Initialize loss dictionary
        loss_dict = {}
        total_loss = None

        # Compute predicted_num_classes from logits (for logging) - works in both modes
        # Since invalid logits are -inf, softmax will give 0 probability for them
        prob_method = getattr(self.config, "prob_method", 'softmax')
        gumbel_hard = getattr(self.config, "gumbel_hard", False)
        
        # Compute annealed gumbel tau (needed for predicted_k_mean calculation)
        if prob_method == "gumbel_softmax":
            gumbel_tau = annealed_params(
                getattr(self.config, "gumbel_anneal_proportion", 0.0),
                getattr(self.config, "gumbel_tau_initial", 1.0),
                getattr(self.config, "gumbel_tau_final", 0.1),
                self.curr_step,
                self.config.max_steps
            )
        else:
            gumbel_tau = 1.0
        
        # Compute predicted_num_classes from P (for logging) - always computed
        if prob_method == "gumbel_softmax":
            P = F.gumbel_softmax(logits, tau=gumbel_tau, hard=gumbel_hard, dim=-1)  # (B, T, max_classes)
        else:
            P = F.softmax(logits, dim=-1)  # (B, T, max_classes)
        
        # Get predicted labels from P: argmax over classes
        pred_labels = P.argmax(dim=-1)  # (B, T)
        
        # Compute predicted_num_classes: number of unique predicted labels for each sample (batch computation)
        max_classes = logits.shape[-1]  # Get from logits shape: (B, T, max_classes)
        # Use one_hot encoding to count which classes appear in each sample
        one_hot = F.one_hot(pred_labels, num_classes=max_classes)  # (B, T, max_classes)
        # Sum over sequence dimension to get count per class per sample
        class_counts = one_hot.sum(dim=1)  # (B, max_classes)
        # Count how many classes have count > 0 (i.e., appear at least once)
        predicted_num_classes = (class_counts > 0).sum(dim=1).float()  # (B,)

        # ARI is undefined for single-class samples, check if we have valid samples
        valid_mask = micro_num_classes >= 2
        
        if valid_mask.any():
            # Compute ARI loss only for valid samples (num_classes >= 2)
            # Note: logits have -inf for invalid classes, soft_ari only uses valid logits[:, :num_classes]
            eps = 1e-6
            cluster_loss = soft_ari_loss(logits, micro_y, micro_num_classes, eps=eps, prob_method=prob_method, gumbel_tau=gumbel_tau, gumbel_hard=gumbel_hard)  # (B,)
            cluster_loss_mean = (cluster_loss * valid_mask.float()).sum() / valid_mask.float().sum()
            total_loss = cluster_loss_mean
        else:
            # All samples are single-class - cannot compute meaningful ARI
            # Use detached zero tensor to avoid gradient flow issues with DDP
            # This explicitly tells PyTorch that col_embedder/row_interactor should not receive gradients
            total_loss = torch.tensor(0.0, device=micro_X.device, requires_grad=True)
            # Note: cluster_loss is not computed to avoid unnecessary forward passes through soft_ari_loss

        # Track ratio of valid samples (num_classes >= 2) for monitoring data quality
        num_invalid_samples = (micro_num_classes < 2).sum().item()
        if valid_mask.any():
            loss_dict.update({
                "cluster_loss": cluster_loss_mean.item(),
                "ari_loss": cluster_loss_mean.item(),
                "predicted_k_mean": predicted_num_classes.mean().item(),
                "true_k_mean": micro_num_classes.float().mean().item(),
                "invalid_single_class_samples": num_invalid_samples,
            })
        else:
            # All samples invalid - report 0 for cluster loss
            loss_dict.update({
                "cluster_loss": 0.0,
                "ari_loss": 0.0,
                "predicted_k_mean": predicted_num_classes.mean().item(),
                "true_k_mean": micro_num_classes.float().mean().item(),
                "invalid_single_class_samples": num_invalid_samples,
            })


        k_decision_weight = getattr(self.config, "k_decision_weight", 1.0)
        if k_decision_weight > 0:
            # k_logits: (B, max_classes - 1) - logits for k=2,...,max_classes
            # Target: micro_num_classes - 2 (since k=2 corresponds to index 0)
            k_target = (micro_num_classes - 2).long()  # (B,) values in [0, max_classes-2]
            
            # Create mask for valid samples (num_classes >= 2)
            # Samples with only 1 class are invalid for clustering and should be skipped
            valid_mask = micro_num_classes >= 2
            
            if valid_mask.any():
                # Clamp k_target to valid range to prevent CUDA assertion errors
                max_k_idx = k_logits.shape[-1] - 1
                k_target_clamped = k_target.clamp(0, max_k_idx)
                
                # Compute loss only for valid samples using reduction='none' then manual mean
                k_decision_loss_per_sample = F.cross_entropy(k_logits, k_target_clamped, reduction='none')
                k_decision_loss = (k_decision_loss_per_sample * valid_mask.float()).sum() / valid_mask.float().sum()
                
                total_loss = total_loss + k_decision_weight * k_decision_loss
                loss_dict["k_decision_loss"] = k_decision_loss.item()
            else:
                # All samples are invalid (single class) - skip k_decision loss
                loss_dict["k_decision_loss"] = 0.0
                total_loss = torch.tensor(0.0, device=micro_X.device, requires_grad=True)
        
        # Compute K prediction accuracy
        k_pred = k_logits.argmax(dim=-1)  # (B,)
        # Compute predicted K mean from MLP
        k_pred_values = k_pred + 2  # Convert index back to k value
        loss_dict["predicted_k_from_logits_mean"] = k_pred_values.float().mean().item()

        # Return model_output as tuple: (logits, k_logits, all_k_logits)
        # This format is compatible with trainer.py's NaN checking for tuples
        model_output = (logits, k_logits, all_k_logits)
        return total_loss, model_output, loss_dict


    @staticmethod
    @torch.no_grad()
    def predict_labels(model_output, micro_num_classes, micro_extra_data=None, config=None):
        """
        Convert model output to predicted labels using P computed from logits.
        
        Parameters
        ----------
        model_output : tuple
            When use_true_k=True (predict_k=False):
                - (logits,) where logits is (B, seq_len, num_classes) with valid num_classes
            When use_true_k=False (predict_k=True):
                - (logits, k_logits, all_k_logits) where:
                  * logits: (B, seq_len, max_classes) logits using true num_classes
                  * k_logits: (B, max_classes - 1) K prediction MLP output
                  * all_k_logits: list of logits for k=2,...,max_classes
        micro_num_classes : torch.Tensor
            Number of classes for each sample in the batch, shape (B,).
            Used when use_true_k=True.
        micro_extra_data : dict, optional
            Additional data (not used in this implementation)
        config : argparse.Namespace or dict
            Configuration parameters. May contain 'use_true_k' to determine
            whether to use true K or MLP-predicted K.
            
        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            - pred: Predicted labels of shape (B, seq_len) with class indices
            - predicted_num_classes: Predicted number of classes of shape (B,)
              When use_true_k=True: number of unique predicted labels
              When use_true_k=False: MLP-predicted K value
        """
        # Determine which k to use (compatible with both dict and Namespace)
        # Default to False (use MLP-predicted K) if not specified
        if config is None:
            use_true_k = False
        elif isinstance(config, dict):
            use_true_k = config.get("use_true_k", False)
        else:
            use_true_k = getattr(config, "use_true_k", False)
        
        # Check model_output format based on use_true_k
        if use_true_k:
            # When use_true_k=True, model is called with predict_k=False
            # model_output is (logits,) where logits is (B, T, num_classes)
            # The logits already correspond to the true num_classes
            logits = model_output[0] if isinstance(model_output, tuple) else model_output
            
            # Direct prediction using true num_classes
            pred = logits.argmax(dim=-1) 
            
            B = pred.shape[0]
            predicted_num_classes = torch.zeros(B, dtype=torch.float32, device=pred.device)
            for b in range(B):
                predicted_num_classes[b] = torch.unique(pred[b]).numel()
            
            return pred, predicted_num_classes
        else:
            # When use_true_k=False, model is called with predict_k=True
            # model_output is (logits, k_logits, all_k_logits)
            logits, k_logits, all_k_logits = model_output
            
            # Use MLP-predicted K
            k_idx = k_logits.argmax(dim=-1)  # (B,)
            
            B = logits.shape[0]
            T = all_k_logits[0].shape[1]

            if B == 1:
                k = k_idx[0].item()
                pred = all_k_logits[k][0].argmax(dim=-1).unsqueeze(0)  # (1, T)
            else:
                pred = torch.zeros(B, T, dtype=torch.long, device=logits.device)
                for b in range(B):
                    k = k_idx[b].item()
                    pred[b] = all_k_logits[k][b].argmax(dim=-1)
            
            # Use MLP-predicted K value
            predicted_num_classes = (k_idx + 2).float()  # (B,)
            
            return pred, predicted_num_classes

    
if __name__ == "__main__":
    parser = build_parser()
    config = parser.parse_args()

    try:
        # Set the start method for subprocesses to 'spawn'
        set_start_method("spawn")
    except RuntimeError:
        pass  # Ignore the error if the context has already been set

    # Create trainer and start training
    trainer = TrainerProb(config)
    trainer.train()
