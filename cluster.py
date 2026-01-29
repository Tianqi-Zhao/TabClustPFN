#!/usr/bin/env python3
"""
TabCluster - Lightweight Clustering Inference Script

A standalone script for clustering tabular data using pretrained TabCluster models.
This script provides a simple interface to load a model checkpoint and perform 
clustering on CSV data without requiring the full evaluation framework.

Usage:
    python cluster.py --checkpoint /path/to/checkpoint.ckpt --data /path/to/data.csv
    python cluster.py --checkpoint /path/to/checkpoint.ckpt --data /path/to/data.csv --output predictions.csv

Example:
    python cluster.py --checkpoint checkpoints/tabcluster.ckpt --data my_data.csv --device cuda

For more options:
    python cluster.py --help
"""

from __future__ import annotations

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional
import numpy as np
import pandas as pd
import torch

from utils.preprocess import DataPreprocessor



# ============================================================================
# Model Loading and Inference
# ============================================================================

def load_model(checkpoint_path: str, device: str = "cpu"):
    """
    Load a TabClusterIMAB model from checkpoint.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the model checkpoint file (.ckpt)
    device : str
        Device to load the model on ('cpu' or 'cuda')
        
    Returns
    -------
    model : nn.Module
        Loaded TabClusterIMAB model in eval mode
    """
    from model.tabcluster import TabClusterIMAB
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    if "config" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'config' key.")
    
    model_config = checkpoint["config"]
    model = TabClusterIMAB(**model_config)
    
    if "state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'state_dict' key.")
    
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    
    print(f"Model (TabClusterIMAB) loaded successfully.")
    return model


def cluster(
    model,
    X: np.ndarray,
    num_clusters: Optional[int] = None,
    device: str = "cpu",
    use_amp: bool = False,
) -> tuple[np.ndarray, int]:
    """
    Perform clustering on preprocessed data.
    
    Parameters
    ----------
    model : nn.Module
        Loaded TabClusterIMAB model
    X : np.ndarray
        Preprocessed data of shape (num_samples, num_features)
    num_clusters : int, optional
        Number of clusters. If None, the model will predict it automatically.
    device : str
        Device to run inference on
    use_amp : bool
        Whether to use automatic mixed precision
        
    Returns
    -------
    labels : np.ndarray
        Cluster labels for each sample
    predicted_k : int
        Number of clusters used/predicted
    """
    from configs.inference_config import InferenceConfig, MgrConfig
    from exp.run_Softari import TrainerProb
    
    # Ensure model is on the correct device
    model = model.to(device)
    
    # Prepare input tensor
    X_tensor = torch.from_numpy(X.astype(np.float64)).float().to(device).unsqueeze(0)  # (1, T, H)
    
    # Configure inference with device setting
    mgr_config = MgrConfig(use_amp=use_amp, device=device)
    inference_config = InferenceConfig(
        COL_CONFIG=mgr_config,
        ROW_CONFIG=mgr_config,
        CLUSTER_CONFIG=mgr_config,
    )
    
    model.eval()
    device_type = 'cuda' if device != 'cpu' else 'cpu'
    amp_context = torch.amp.autocast(device_type, enabled=use_amp and device != 'cpu')
    
    with torch.no_grad(), amp_context:
        if num_clusters is not None:
            # Use specified number of clusters
            num_classes_tensor = torch.tensor([num_clusters], device=device)
            model_output = model(
                X_tensor,
                num_classes=num_classes_tensor,
                inference_config=inference_config,
                predict_k=False
            )
            # predict_labels expects tuple format
            if not isinstance(model_output, tuple):
                model_output = (model_output,)
            
            y_pred, predicted_k = TrainerProb.predict_labels(
                model_output, 
                num_classes_tensor,
                config={'use_true_k': True}
            )
        else:
            # Let model predict k automatically
            # Note: model needs a dummy num_classes for internal extraction,
            # but predict_labels will use k_logits to select from all_k_logits
            max_classes = model.max_classes
            dummy_num_classes = torch.tensor([max_classes], device=device)
            model_output = model(
                X_tensor,
                num_classes=dummy_num_classes,
                predict_k=True,
                inference_config=inference_config
            )
            
            y_pred, predicted_k = TrainerProb.predict_labels(
                model_output, 
                None,
                config={'use_true_k': False}
            )
    
    labels = y_pred.squeeze(0).cpu().numpy()
    k = int(predicted_k.squeeze(0).cpu().item()) if predicted_k.dim() > 0 else int(predicted_k.cpu().item())
    
    return labels, k


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TabCluster - Clustering tabular data with pretrained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Basic usage with automatic cluster number prediction
    python cluster.py --checkpoint model.ckpt --data data.csv

    # Specify number of clusters
    python cluster.py --checkpoint model.ckpt --data data.csv --num_clusters 5

    # Save predictions to file
    python cluster.py --checkpoint model.ckpt --data data.csv --output predictions.csv

    # Use GPU with mixed precision
    python cluster.py --checkpoint model.ckpt --data data.csv --device cuda --amp
            """
        )
    
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint file (.ckpt)"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to input data file (CSV format)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save predictions (CSV format). If not specified, prints to stdout."
    )
    parser.add_argument(
        "--num_clusters", "-k",
        type=int,
        default=None,
        help="Number of clusters. If not specified, the model will predict it automatically."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on (default: cpu)"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision (only effective on CUDA)"
    )
    parser.add_argument(
        "--nan_strategy",
        type=str,
        default="zero",
        choices=["zero", "mean", "median", "mode"],
        help="Strategy for handling NaN values (default: zero)"
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="z-norm",
        choices=["z-norm", "minmax", "unit_variance", "none"],
        help="Normalization method (default: z-norm)"
    )
    parser.add_argument(
        "--no_header",
        action="store_true",
        help="Treat first row as data (no header row)"
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default=None,
        help="Name of column containing true labels (will be excluded from clustering)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Load data
    print(f"Loading data from: {args.data}")
    header = None if args.no_header else 'infer'
    df = pd.read_csv(args.data, header=header)
    
    if args.verbose:
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
    
    # Extract true labels if specified
    true_labels = None
    if args.label_column is not None:
        if args.label_column not in df.columns:
            print(f"Warning: Label column '{args.label_column}' not found in data")
        else:
            true_labels = df[args.label_column].values
            df = df.drop(columns=[args.label_column])
            if args.verbose:
                print(f"  Extracted label column: {args.label_column}")
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor(
        categorical_encoding="label",
        nan_strategy=args.nan_strategy,
        normalization=args.normalization,
        remove_zero_variance=True,
    )
    X = preprocessor.fit_transform(df)
    
    if args.verbose:
        print(f"  Preprocessed shape: {X.shape}")
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Perform clustering
    print("Performing clustering...")
    labels, predicted_k = cluster(
        model=model,
        X=X,
        num_clusters=args.num_clusters,
        device=args.device,
        use_amp=args.amp,
    )
    
    print(f"Clustering complete!")
    print(f"  Predicted/Used number of clusters: {predicted_k}")
    print(f"  Unique labels in predictions: {len(np.unique(labels))}")
    
    # Compute ARI if true labels are available
    if true_labels is not None:
        try:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            ari = adjusted_rand_score(true_labels, labels)
            nmi = normalized_mutual_info_score(true_labels, labels)
            print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
            print(f"  Normalized Mutual Information (NMI): {nmi:.4f}")
        except Exception as e:
            print(f"  Could not compute metrics: {e}")
    
    # Save or print results
    if args.output:
        result_df = pd.DataFrame({
            'cluster_label': labels
        })
        if true_labels is not None:
            result_df['true_label'] = true_labels
        result_df.to_csv(args.output, index=False)
        print(f"Predictions saved to: {args.output}")
    else:
        print("\nCluster assignments:")
        print(labels)


if __name__ == "__main__":
    main()
