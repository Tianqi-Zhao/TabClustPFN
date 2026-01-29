"""Define argument parser for TabICL training."""

import argparse


def str2bool(value):
    return value.lower() == "true"

def build_parser():
    """Build parser with all TabICL training arguments."""
    parser = argparse.ArgumentParser()

    ###########################################################################
    ###### TensorBoard Config #################################################
    ###########################################################################
    parser.add_argument(
        "--log_to_tensorboard", 
        default=True, 
        type=str2bool, 
        help="If True, log results to TensorBoard."
    )

    ###########################################################################
    ###### Training Config ####################################################
    ###########################################################################
    parser.add_argument("--device", default="cuda", type=str, help="Device for training: cpu, cuda, cuda:0")
    parser.add_argument(
        "--dtype", default="float32", type=str, help="Data type (supported for float16, float32) used for training"
    )
    parser.add_argument("--np_seed", type=int, default=42, help="Random seed for numpy")
    parser.add_argument("--torch_seed", type=int, default=42, help="Random seed for torch")
    parser.add_argument("--max_steps", type=int, default=2000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--micro_batch_size", type=int, default=8, help="Size of micro-batches for gradient accumulation"
    )

    # Optimization Config
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--optimizer", 
        type=str, 
        default="sgd", 
        help="Optimizer type: adamw, adam, sgd, stagesgd (adaptive stage SGD), rmsprop, adagrad"
    )
    parser.add_argument("--momentum", type=float, default=0.0, help="SGD momentum factor")
    parser.add_argument("--dampening", type=float, default=0.0, help="SGD dampening factor")
    parser.add_argument("--nesterov", default=False, type=str2bool, help="SGD Nesterov momentum")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam/AdamW beta1 parameter")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam/AdamW beta2 parameter")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam/AdamW epsilon parameter")
    parser.add_argument("--rmsprop_alpha", type=float, default=0.99, help="RMSprop smoothing constant")
    parser.add_argument(
        "--scheduler", type=str, default="cosine_warmup", 
        help="Learning rate scheduler: constant, linear_warmup, cosine_warmup, cosine_with_restarts, polynomial_decay_warmup."
    )
    parser.add_argument(
        "--warmup_proportion",
        type=float,
        default=0.2,
        help="The proportion of total steps over which we warmup. Only used when scheduler is not constant."
        "If this value is set to -1, we warmup for a fixed number of steps.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="The number of steps over which we warm up. Only used when warmup_proportion is set to -1",
    )
    parser.add_argument(
        "--cosine_num_cycles",
        type=int,
        default=3,
        help="Number of hard restarts for cosine schedule. Only used when scheduler is cosine_with_restarts",
    )
    parser.add_argument(
        "--cosine_amplitude_decay",
        type=float,
        default=0.9,
        help="Amplitude scaling factor per cycle. Only used when scheduler is cosine_with_restarts",
    )
    parser.add_argument("--cosine_lr_end", type=float, default=1e-7, help="Final learning rate for cosine_with_restarts")
    parser.add_argument(
        "--poly_decay_lr_end", type=float, default=1e-7, help="Final learning rate for polynomial decay scheduler"
    )
    parser.add_argument(
        "--poly_decay_power", type=float, default=1.0, help="Power factor for polynomial decay scheduler"
    )

    # Gradient Config
    parser.add_argument("--gradient_clipping", type=float, default=1.0, help="If > 0, clip gradients.")
    parser.add_argument("--grad_stats", default=True, type=str2bool, help="If log grad stat.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay / L2 regularization penalty")
    parser.add_argument("--verify_gradients", default=False, type=str2bool, help="If True, verify manual gradient computation against autograd.")
    
    # Adaptive Stage SGD Config
    # Note: To use adaptive stage SGD, simply set --optimizer stagesgd
    # The parameters below are used to configure the adaptive behavior
    parser.add_argument(
        "--adaptive_sgd_use_ema",
        default=True,
        type=str2bool,
        help="Use exponential moving average (EMA) for oscillation detection. Recommended for generated data. If False, use sliding window."
    )
    parser.add_argument(
        "--adaptive_sgd_ema_beta",
        type=float,
        default=0.99,
        help="EMA decay coefficient (0.98-0.995). Higher value means smoother and more dependent on history. Only used when adaptive_sgd_use_ema=True."
    )
    parser.add_argument(
        "--adaptive_sgd_loss_window",
        type=int,
        default=1000,
        help="Loss monitoring window size in steps. Only used when adaptive_sgd_use_ema=False."
    )
    parser.add_argument(
        "--adaptive_sgd_oscillation_threshold",
        type=float,
        default=0.8,
        help="Oscillation threshold (coefficient of variation = std/mean). Trigger reset when exceeded. Range: 0.3 (sensitive) to 0.7 (insensitive)."
    )
    parser.add_argument(
        "--adaptive_sgd_lr_decay_factor",
        type=float,
        default=0.7,
        help="Learning rate decay factor on reset. new_lr = old_lr * decay_factor. Range: 0.5 (aggressive) to 0.7 (mild)."
    )
    parser.add_argument(
        "--adaptive_sgd_min_lr",
        type=float,
        default=1e-5,
        help="Minimum learning rate. LR will not decay below this value."
    )
    parser.add_argument(
        "--adaptive_sgd_check_interval",
        type=int,
        default=1000,
        help="Check interval in steps. How often to check if reset is needed."
    )
    parser.add_argument(
        "--adaptive_sgd_cooldown_steps",
        type=int,
        default=2000,
        help="Cooldown period in steps after each reset. Prevents frequent resets."
    )
    parser.add_argument(
        "--adaptive_sgd_verbose",
        default=True,
        type=str2bool,
        help="Print detailed information when reset occurs."
    )

    # Prior Dataset Config
    parser.add_argument(
        "--prior_dir",
        type=str,
        default=None,
        help="If set, load pre-generated prior datasets directly from this directory on disk instead of generating them on the fly.",
    )
    parser.add_argument(
        "--load_prior_start",
        type=int,
        default=0,
        help="Batch index to start loading from pre-generated prior data. Only used when prior_dir is set.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=10,
        help="Max batches from pre-generated prior data. Only used when prior_dir is set.",
    )
    parser.add_argument(
        "--delete_after_load",
        default=False,
        type=str2bool,
        help="Delete prior data after loading. Only used when prior_dir is set.",
    )
    parser.add_argument("--batch_size_per_gp", type=int, default=4, help="Batch size per group")
    parser.add_argument("--min_features", type=int, default=5, help="The minimum number of features")
    parser.add_argument("--max_features", type=int, default=64, help="The maximum number of features")
    parser.add_argument("--max_classes", type=int, default=10, help="The maximum number of classes")
    parser.add_argument("--min_seq_len", type=int, default=None, help="Minimum samples per dataset")
    parser.add_argument("--max_seq_len", type=int, default=1000, help="Maximum samples per dataset")
    parser.add_argument(
        "--log_seq_len",
        default=False,
        type=str2bool,
        help="If True, sample sequence length from log-uniform distribution between min_seq_len and max_seq_len",
    )
    parser.add_argument(
        "--seq_len_per_gp",
        default=False,
        type=str2bool,
        help="If True, sample sequence length independently for each group",
    )
    parser.add_argument(
        "--replay_small",
        default=False,
        type=str2bool,
        help="If True, occasionally sample smaller sequence lengths to ensure model robustness on smaller datasets"
    )
    parser.add_argument(
        "--prior_type", default="mix_scm", type=str, help="Prior type: dummy, mlp_scm, tree_scm, mix_scm, gm, mix_prior"
    )
    parser.add_argument("--prior_device", default="cpu", type=str, help="Device for prior data generation")
    parser.add_argument(
        "--prior_config",
        default=None,
        type=str,
        help="prior_config path",
    )
    parser.add_argument(
        "--prior_anneal_proportion",
        type=float,
        default=0.0,
        help=(
            "Proportion of total training steps during which prior-type probabilities are annealed "
            "(e.g., 0.3 means annealing over the first 30%% of training steps). "
            "If set to 0, no annealing is applied and the fixed mix_prior_probas from prior_config are used. "
            "If > 0, probabilities are linearly annealed from mix_prior_probas_initial to mix_prior_probas_final."
        ),
    )

    ###########################################################################
    ##### Model Architecture Config ###########################################
    ###########################################################################
    parser.add_argument(
        "--amp",
        default=True,
        type=str2bool,
        help="If True, use automatic mixed precision (AMP) which can provide significant speedups on compatible GPU",
    )
    parser.add_argument(
        "--model_compile",
        default=False,
        type=str2bool,
        help="If True, compile the model using torch.compile for speedup",
    )

    # Column Embedding Config
    parser.add_argument("--embed_dim", type=int, default=128, help="Base embedding dimension")
    parser.add_argument("--col_num_blocks", type=int, default=3, help="Number of blocks in column embedder")
    parser.add_argument("--col_nhead", type=int, default=4, help="Number of attention heads in column embedder")
    parser.add_argument("--col_num_inds", type=int, default=128, help="Number of inducing points in column embedder")
    parser.add_argument("--freeze_col", default=False, type=str2bool, help="Whether to freeze the column embedder")

    # Row Interaction Config
    parser.add_argument("--row_num_blocks", type=int, default=3, help="Number of blocks in row interactor")
    parser.add_argument("--row_nhead", type=int, default=8, help="Number of attention heads in row interactor")
    parser.add_argument("--row_num_cls", type=int, default=4, help="Number of CLS tokens in row interactor")
    parser.add_argument("--row_rope_base", type=float, default=100000, help="RoPE base value for row interactor")
    parser.add_argument("--freeze_row", default=False, type=str2bool, help="Whether to freeze the row interactor")

    # Cluster Learning Config (for TabClusterIMAB)
    parser.add_argument("--cluster_num_blocks", type=int, default=6, help="Number of induced decoder blocks in cluster learning module (TabClusterIMAB)")
    parser.add_argument("--cluster_nhead", type=int, default=4, help="Number of attention heads in cluster learning module")
    parser.add_argument("--cluster_use_rope_cross_attn", default=False, type=str2bool, help="If True, use rotary positional encoding in cross-attention when ind_vectors_hidden queries src")
    parser.add_argument("--cluster_rope_base", type=float, default=100000, help="Base scaling factor for rotary position encoding in cluster learning module (only used if cluster_use_rope_cross_attn=True)")
    parser.add_argument("--cluster_use_representation_self_att", default=False, type=str2bool, help="If True, Stage 2 uses decoder block (self-attention + cross-attention); If False, Stage 2 uses only cross-attention (more efficient)")
    parser.add_argument("--freeze_cluster", default=False, type=str2bool, help="Whether to freeze the cluster learner module")

    parser.add_argument("--use_true_k", default=False, type=str2bool, help="If True, use true k")

    # Shared Architecture Config
    parser.add_argument("--ff_factor", type=int, default=2, help="Expansion factor for feedforward dimensions")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function type")
    parser.add_argument(
        "--norm_first", default=True, type=str2bool, help="If True, use pre-norm transformer architecture"
    )

    ###########################################################################
    ###### Checkpointing ######################################################
    ###########################################################################
    parser.add_argument("--checkpoint_dir", default=None, type=str, help="Directory for checkpoint saving and loading")
    parser.add_argument("--version", default=None, type=int, help="Experiment version number. If not specified, auto-increment from existing versions")
    parser.add_argument("--save_temp_every", default=50, type=int, help="Steps between temporary checkpoints")
    parser.add_argument("--save_perm_every", default=5000, type=int, help="Steps between permanent checkpoints")
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=5,
        help="Maximum number of temporary checkpoints to keep. Permanent checkpoints are not counted.",
    )
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to specific checkpoint file to load")
    parser.add_argument("--only_load_model", default=False, type=str2bool, help="Whether to only load model weights")
    parser.add_argument("--phase_change", default=False, type=str2bool, help="If True, preserve optimizer momentum but reset scheduler (for phase change/re-warmup)")

    ###########################################################################
    ###### Loss Config ########################################################
    ###########################################################################
    # Progressive Loss Configuration
    parser.add_argument(
        "--warmup_loss_proportion", 
        type=float, 
        default=0.0, 
        help="Proportion of total steps to use warmup loss (e.g., 0.1 for first 10%% of training). Set to 0 to disable."
    )
    parser.add_argument("--tau", type=float, default=0.1, help="Temperature for loss computation")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for row loss component")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for column loss component")

    parser.add_argument("--k_decision_weight", type=float, default=1.0, help="Weight for k_decision_loss")
    # ACC Loss
    parser.add_argument("--sinkhorn_iters", type=int, default=20, help="Number of iterations for Sinkhorn-Knopp algorithm in loss calculation")
    parser.add_argument(
        "--prob_method", 
        type=str, 
        default='softmax', 
        choices=['softmax', 'gumbel_softmax'], 
        help="Method to get probabilities from logits ('softmax' or 'gumbel_softmax')"
    )
    parser.add_argument(
        "--gumbel_tau_initial", 
        type=float, 
        default=1.0, 
        help="Initial temperature tau for Gumbel-Softmax"
    )
    parser.add_argument(
        "--gumbel_tau_final", 
        type=float, 
        default=0.1, 
        help="Final temperature tau for Gumbel-Softmax after annealing"
    )
    parser.add_argument(
        "--gumbel_anneal_proportion", 
        type=float, 
        default=0.3, 
        help="Proportion of total steps to anneal Gumbel-Softmax tau. 0 to disable."
    )
    parser.add_argument(
        "--gumbel_hard", 
        type=str2bool, 
        default=False, 
        help="Whether to use hard sampling in Gumbel-Softmax"
    )
    parser.add_argument("--detach_cost_matrix", default=False, type=str2bool, help="Whether to detach cost matrix from gradient computation in OT loss")

    return parser


HPARAM_KEYS = [
    # Training
    "max_steps", "batch_size", "micro_batch_size", "phase_change",
    # Optimization
    "lr", "optimizer", "momentum", "dampening", "nesterov", "beta1", "beta2", "eps", "rmsprop_alpha",
    "scheduler", "warmup_proportion", "warmup_steps", "cosine_num_cycles", "cosine_amplitude_decay",
    "cosine_lr_end", "poly_decay_lr_end", "poly_decay_power",
    # Gradient
    "gradient_clipping", "weight_decay",
    # Adaptive SGD (when optimizer='stagesgd')
    "adaptive_sgd_use_ema", "adaptive_sgd_ema_beta", "adaptive_sgd_loss_window",
    "adaptive_sgd_oscillation_threshold", "adaptive_sgd_lr_decay_factor", "adaptive_sgd_min_lr",
    "adaptive_sgd_check_interval", "adaptive_sgd_cooldown_steps",
    # Prior Dataset
    "prior_dir", "load_prior_start", "max_batches", "batch_size_per_gp", "min_features", "max_features",
    "max_classes", "min_seq_len", "max_seq_len", "log_seq_len", "seq_len_per_gp", "replay_small", "prior_type",
    "prior_anneal_proportion",
    # Model Architecture
    "embed_dim", "col_num_blocks", "col_nhead", "col_num_inds", "freeze_col",
    "row_num_blocks", "row_nhead", "row_num_cls", "row_rope_base", "freeze_row",
    "cluster_num_blocks", "cluster_nhead", "cluster_use_rope_cross_attn", "cluster_rope_base", "cluster_use_representation_self_att", "freeze_cluster",
    "use_true_k",
    "ff_factor", "dropout", "activation", "norm_first",
    # Loss
    "warmup_loss_proportion", "tau", "alpha", "beta",
    "k_decision_weight",
    "sinkhorn_iters", "prob_method", "gumbel_tau_initial", "gumbel_tau_final", "gumbel_anneal_proportion", "gumbel_hard", 
    "detach_cost_matrix",
]

METRIC_KEYS_FOR_HPARAMS = [
    "loss",
    "accuracy",
    "ari",
]
