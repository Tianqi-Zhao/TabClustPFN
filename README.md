# TabClustPFN

This repository contains the implementation for TabClustPFN, a transformer-based model for in-context learning on tabular clustering tasks.

## Quick Start: Clustering with Pretrained Model

### 1. Download Checkpoint

Download the pretrained model checkpoint from Google Drive:

📥 **[Download Checkpoint](https://drive.google.com/file/d/1GzZBgVXwOB8y9JM89xa7ln-la6v2sUO3/view?usp=sharing)**

### 2. Run Clustering

Use `cluster.py` to perform clustering on your tabular data:

```bash
# Basic usage - model automatically predicts number of clusters
python cluster.py --checkpoint /path/to/checkpoint.ckpt --data your_data.csv --device cuda

# Specify number of clusters
python cluster.py --checkpoint /path/to/checkpoint.ckpt --data your_data.csv --num_clusters 5

# Save predictions to file
python cluster.py --checkpoint /path/to/checkpoint.ckpt --data your_data.csv --output predictions.csv

# With true labels for evaluation (computes ARI and NMI)
python cluster.py --checkpoint /path/to/checkpoint.ckpt --data your_data.csv --label_column true_label
```

### Available Options

| Option | Description |
|--------|-------------|
| `--checkpoint, -c` | Path to model checkpoint file (required) |
| `--data, -d` | Path to input CSV file (required) |
| `--output, -o` | Path to save predictions |
| `--num_clusters, -k` | Number of clusters (auto-predicted if not specified) |
| `--device` | `cpu` or `cuda` (default: cpu) |
| `--amp` | Enable mixed precision (CUDA only) |
| `--nan_strategy` | Handle NaN: `zero`, `mean`, `median`, `mode` |
| `--normalization` | `z-norm`, `minmax`, `unit_variance`, `none` |
| `--label_column` | Column name with true labels (for evaluation) |
| `--verbose, -v` | Print detailed output |

For full options: `python cluster.py --help`

---

## Training

To train the model from scratch, run the training script:

```bash
bash scripts/train.sh
```

### Prerequisites

Before training, please ensure that you have:

1. **Downloaded the TabICL checkpoint**: The training script requires a pre-trained TabICL checkpoint file. Please download the checkpoint and update the `--checkpoint_path` parameter in `scripts/train.sh` to point to the checkpoint file location.

   ```bash
   # Example: Update this line in scripts/train.sh
   --checkpoint_path "path/to/tabicl/tabicl.ckpt"
   ```

2. **Configured the experiment directory**: Update the `EXPERIMENTS_BASE` variable in `scripts/train.sh` to specify where checkpoints and logs should be saved.

   ```bash
   # Example: Update this line in scripts/train.sh
   EXPERIMENTS_BASE="/path/to/checkpoints/"
   ```

### Training Configuration

The training script supports various hyperparameters that can be modified directly in `scripts/train.sh`. Key parameters include:

- **Model architecture**: `--embed_dim`, `--col_num_blocks`, `--row_num_blocks`, `--cluster_num_blocks`, etc.
- **Training settings**: `--max_steps`, `--batch_size`, `--lr`, `--optimizer`, etc.
- **Prior dataset configuration**: `--prior_type`, `--max_features`, `--max_classes`, `--min_seq_len`, `--max_seq_len`, etc.

### Distributed Training

The script uses `torchrun` for distributed training. By default, it runs on 4 GPUs (`--nproc_per_node=4`). Adjust this parameter based on your available hardware.

### Outputs

- **Checkpoints**: Saved to `$EXPERIMENTS_BASE/$NAME/` directory
- **Logs**: Saved to `$EXPERIMENTS_BASE/logs/` directory
- **TensorBoard logs**: Enabled by default via `--log_to_tensorboard True`

---

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial** — You may not use the material for commercial purposes.