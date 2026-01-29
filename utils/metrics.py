import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
from sklearn.metrics import confusion_matrix
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def accuracy(true_row_labels, predicted_row_labels):
    """Get the best accuracy.

    Parameters
    ----------
    true_row_labels: array-like
        The true row labels, given as external information
    predicted_row_labels: array-like
        The row labels predicted by the model

    Returns
    -------
    float
        Best value of accuracy
    """

    cm = confusion_matrix(true_row_labels, predicted_row_labels)
    row_ind, col_ind = linear_sum_assignment(_make_cost_m(cm))
    total = cm[row_ind, col_ind].sum()

    return (total * 1. / np.sum(cm))

def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the accuracy between true and predicted labels.
    Handles empty inputs gracefully.
    """
    if y_true.shape != y_pred.shape:
        # For safety, though in this project they should always match.
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    if y_true.size == 0:
        # Convention: Accuracy is 100% for empty sets (no mistakes made).
        return 1.0
    
    return accuracy(y_true, y_pred)

def calculate_entropy_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate entropy-based splitting and merging metrics.
    
    Returns:
        dict: {
            "splitting_entropy": float, # Lower is better, 0 is perfect
            "merging_entropy": float,   # Lower is better, 0 is perfect
            "splitting_perplexity": float, # Effective number of splits (e^entropy), 1.0 is perfect
            "merging_perplexity": float    # Effective number of merges (e^entropy), 1.0 is perfect
        }
    """
    # 1. Build contingency matrix
    # shape: (n_true_classes, n_pred_clusters)
    cm = contingency_matrix(y_true, y_pred)
    
    if hasattr(cm, "toarray"):
        cm = cm.toarray()
    cm = cm.astype(np.float64)

    # ---------------------------
    # Calculate Splitting (Over-segmentation) - based on rows (True Classes)
    # ---------------------------
    row_sums = cm.sum(axis=1)
    valid_rows_mask = row_sums > 0
    
    if valid_rows_mask.sum() == 0:
        split_ent = 0.0
    else:
        # Row normalization: P(pred | true)
        # Each row of probs sums to 1
        row_probs = cm[valid_rows_mask] / (row_sums[valid_rows_mask][:, np.newaxis] + 1e-10)
        
        # Calculate entropy for each row: -sum(p * log(p))
        # Note: log(0) causes issues, so add eps, or use xlogx property (0*log0 = 0)
        row_probs_safe = np.maximum(row_probs, 1e-10)
        row_entropies = -np.sum(row_probs * np.log(row_probs_safe), axis=1)
        
        # Take mean (Macro-average): average splitting degree per class
        # Could also use Weighted Average (weighted by row_sums), but Macro is more sensitive for diagnosing class splitting
        split_ent = float(np.mean(row_entropies))

    # ---------------------------
    # Calculate Merging (Under-segmentation) - based on columns (Pred Clusters)
    # ---------------------------
    col_sums = cm.sum(axis=0)
    
    # This step is crucial: must ignore completely unused predicted clusters (Empty Clusters)
    # Otherwise there will be many 0 entropies (due to 0/0), or meaningless calculations that lower the average
    valid_cols_mask = col_sums > 0
    
    if valid_cols_mask.sum() == 0:
        merge_ent = 0.0
    else:
        # Column normalization: P(true | pred)
        # Each column of probs sums to 1
        col_probs = cm[:, valid_cols_mask] / (col_sums[valid_cols_mask][np.newaxis, :] + 1e-10)
        
        col_probs_safe = np.maximum(col_probs, 1e-10)
        col_entropies = -np.sum(col_probs * np.log(col_probs_safe), axis=0)
        
        merge_ent = float(np.mean(col_entropies))

    return {
        "splitting_entropy": split_ent,
        "merging_entropy": merge_ent,
        "splitting_perplexity": float(np.exp(split_ent)),
        "merging_perplexity": float(np.exp(merge_ent))
    }

def calculate_internal_metrics(X: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate internal clustering metrics (no ground truth labels required).
    
    Args:
        X: Feature matrix, shape (n_samples, n_features)
        y_pred: Predicted cluster labels, shape (n_samples,)
    
    Returns:
        dict: Contains the following internal metrics:
            - "silhouette": Silhouette Coefficient (higher is better, range [-1, 1])
            - "calinski_harabasz": Calinski-Harabasz Index (higher is better)
            - "davies_bouldin": Davies-Bouldin Index (lower is better)
    
    Note:
        - If there is only one cluster or all samples are in the same cluster, some metrics may not be computable
        - If the number of samples is less than 2, returns default values
    """
    
    X = np.asarray(X)
    y_pred = np.asarray(y_pred).flatten()
    
    # Check input validity
    if X.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Shape mismatch: X has {X.shape[0]} samples, y_pred has {y_pred.shape[0]} samples")
    
    if X.shape[0] < 2:
        # Too few samples, return default values
        return {
            "silhouette": 0.0,
            "calinski_harabasz": 0.0,
            "davies_bouldin": float('inf')
        }
    
    # Check number of clusters
    unique_labels = np.unique(y_pred)
    n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise label -1
    
    if n_clusters < 2:
        # Only one cluster, cannot compute some metrics
        return {
            "silhouette": 0.0,
            "calinski_harabasz": 0.0,
            "davies_bouldin": float('inf')
        }
    
    # Calculate Silhouette Coefficient
    if n_clusters == 1:
        silhouette = 0.0
    else:
        silhouette = silhouette_score(X, y_pred)
    
    # Calculate Calinski-Harabasz Index
    if n_clusters == 1:
        calinski_harabasz = 0.0
    else:
        calinski_harabasz = calinski_harabasz_score(X, y_pred)
    
    # Calculate Davies-Bouldin Index
    if n_clusters == 1:
        davies_bouldin = float('inf')
    else:
        davies_bouldin = davies_bouldin_score(X, y_pred)
    
    return {
        "silhouette": float(silhouette),
        "calinski_harabasz": float(calinski_harabasz),
        "davies_bouldin": float(davies_bouldin)
    }

def calculate_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, seq_lens: torch.Tensor = None, 
                     pred_num_classes: torch.Tensor = None, true_num_classes: torch.Tensor = None,
                     return_individual: bool = False, X: np.ndarray = None) -> dict:
    """
    Calculates evaluation metrics for a batch of predictions.

    This function can handle both a single dataset and a batch of datasets.
    If `seq_lens` is provided, it calculates metrics for each item in the batch
    up to its specified sequence length and returns the averaged metrics.
    If `seq_lens` is None, it assumes a single dataset and calculates metrics directly.

    Parameters
    ----------
    y_pred : torch.Tensor or np.ndarray
        Predicted labels. Shape can be (T,) for a single dataset or (B, T) for a batch.
    y_true : torch.Tensor or np.ndarray
        Ground truth labels. Shape should match y_pred.
    seq_lens : torch.Tensor or np.ndarray, optional
        Tensor of sequence lengths for each item in the batch. Shape (B,).
        If None, the entire length of the tensors is used.
    pred_num_classes : torch.Tensor or np.ndarray or int, optional
        Predicted number of classes for each item in the batch. Shape (B,) or scalar.
        If provided, num_classes accuracy will be calculated.
    true_num_classes : torch.Tensor or np.ndarray or int, optional
        True number of classes for each item in the batch. Shape (B,) or scalar.
        If provided, num_classes accuracy will be calculated.
    return_individual : bool, optional
        If True, returns lists of individual scores for each item in the batch,
        in addition to the averaged metrics. Defaults to False.
    X : np.ndarray, optional
        Feature matrix for calculating internal clustering metrics (e.g., Silhouette Coefficient).
        Shape should be (n_samples, n_features). If provided, internal metrics will be calculated.

    Returns
    -------
    dict
        A dictionary containing the calculated metrics.
        If X is provided, also includes internal metrics: silhouette, calinski_harabasz, davies_bouldin.
        If return_individual is True, it also includes lists of individual scores.
    """
    # Helper function: convert to numpy array uniformly
    def _to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        elif isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        else:
            raise TypeError(f"Expected torch.Tensor or numpy.ndarray, got {type(x)}")
    
    # Convert inputs to numpy uniformly
    y_pred_np = _to_numpy(y_pred)
    y_true_np = _to_numpy(y_true)
    if seq_lens is not None:
        seq_lens_np = _to_numpy(seq_lens)
    else:
        seq_lens_np = None

    # Process X parameter
    if X is not None:
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)
    else:
        X_np = None
    
    if y_pred_np.ndim > 1:
        # Batch mode: iterate through each item in the batch
        batch_size = y_pred_np.shape[0]
        if batch_size == 0:
            base_metrics = {"ari": 0.0, "accuracy": 0.0, "nmi": 0.0, "splitting_entropy": 0.0, "merging_entropy": 0.0, "splitting_perplexity": 0.0, "merging_perplexity": 0.0}
            if X_np is not None:
                base_metrics.update({"silhouette": 0.0, "calinski_harabasz": 0.0, "davies_bouldin": float('inf')})
            return base_metrics

        individual_results = {
            "ari": [], "accuracy": [], "nmi": [], "splitting_entropy": [], "merging_entropy": [],
            "splitting_perplexity": [], "merging_perplexity": []
        }
        
        # Add internal metrics if X is provided
        if X_np is not None:
            individual_results.update({"silhouette": [], "calinski_harabasz": [], "davies_bouldin": []})

        if seq_lens_np is None:
            seq_lens_np = np.array([y_pred_np.shape[1]] * batch_size)

        for i in range(batch_size):
            seq_len_i = int(seq_lens_np[i])
            y_pred_i = y_pred_np[i, :seq_len_i]
            y_true_i = y_true_np[i, :seq_len_i]

            # Metrics
            individual_results["ari"].append(adjusted_rand_score(y_true_i, y_pred_i))
            individual_results["accuracy"].append(calculate_accuracy(y_true_i, y_pred_i))
            individual_results["nmi"].append(normalized_mutual_info_score(y_true_i, y_pred_i))
            
            # Entropy Metrics
            ent_metrics = calculate_entropy_metrics(y_true_i, y_pred_i)
            individual_results["splitting_entropy"].append(ent_metrics["splitting_entropy"])
            individual_results["merging_entropy"].append(ent_metrics["merging_entropy"])
            individual_results["splitting_perplexity"].append(ent_metrics["splitting_perplexity"])
            individual_results["merging_perplexity"].append(ent_metrics["merging_perplexity"])
            
            # Internal Metrics (if X is provided)
            if X_np is not None:
                X_i = X_np[i, :seq_len_i]
                internal_metrics = calculate_internal_metrics(X_i, y_pred_i)
                individual_results["silhouette"].append(internal_metrics["silhouette"])
                individual_results["calinski_harabasz"].append(internal_metrics["calinski_harabasz"])
                individual_results["davies_bouldin"].append(internal_metrics["davies_bouldin"])

        agg_results = {
            "ari": float(sum(individual_results["ari"]) / batch_size),
            "accuracy": float(sum(individual_results["accuracy"]) / batch_size),
            "nmi": float(sum(individual_results["nmi"]) / batch_size),
            "splitting_entropy": float(sum(individual_results["splitting_entropy"]) / batch_size),
            "merging_entropy": float(sum(individual_results["merging_entropy"]) / batch_size),
            "splitting_perplexity": float(sum(individual_results["splitting_perplexity"]) / batch_size), # Average effective number of split clusters
            "merging_perplexity": float(sum(individual_results["merging_perplexity"]) / batch_size),     # Average effective number of merge sources
        }

        # Calculate num_classes accuracy if both pred_num_classes and true_num_classes are provided
        # Batch mode: assume pred_num_classes and true_num_classes are arrays of shape (B,)
        if pred_num_classes is not None and true_num_classes is not None:
            # Convert to numpy arrays (assume shape (B,))
            pred_num_classes_np = _to_numpy(pred_num_classes)
            true_num_classes_np = _to_numpy(true_num_classes)
            
            # Exact match accuracy: percentage of samples where predicted num_classes equals true num_classes
            num_classes_matches = (pred_num_classes_np == true_num_classes_np).astype(float)
            num_classes_accuracy = float(num_classes_matches.mean())
            agg_results["num_classes_accuracy"] = num_classes_accuracy
            
            # Mean absolute error for num_classes
            num_classes_mae = float(np.abs(pred_num_classes_np - true_num_classes_np).mean())
            agg_results["num_classes_mae"] = num_classes_mae
            
            if return_individual:
                agg_results["individual_num_classes_matches"] = num_classes_matches.tolist()

        
        # Add internal metrics if X was provided
        if X_np is not None:
            agg_results.update({
                "silhouette": float(sum(individual_results["silhouette"]) / batch_size),
                "calinski_harabasz": float(sum(individual_results["calinski_harabasz"]) / batch_size),
                "davies_bouldin": float(sum(individual_results["davies_bouldin"]) / batch_size)
            })
        if return_individual:
            agg_results.update({f"individual_{k}": v for k, v in individual_results.items()})
        
        return agg_results
    else:
        # Single instance mode (or batch treated as a flat array)
        y_pred_flat = y_pred_np.flatten()
        y_true_flat = y_true_np.flatten()
        
        # ARI
        ari = adjusted_rand_score(y_true_flat, y_pred_flat)

        # Accuracy
        acc = calculate_accuracy(y_true_flat, y_pred_flat)

        # NMI
        nmi = normalized_mutual_info_score(y_true_flat, y_pred_flat)

        ent_metrics = calculate_entropy_metrics(y_true_flat, y_pred_flat)

        results = {"ari": ari, 
                "accuracy": acc, 
                "nmi": nmi,
                "splitting_entropy": ent_metrics["splitting_entropy"], 
                "merging_entropy": ent_metrics["merging_entropy"], 
                "splitting_perplexity": ent_metrics["splitting_perplexity"], 
                "merging_perplexity": ent_metrics["merging_perplexity"]}
        
        # Calculate num_classes accuracy if both pred_num_classes and true_num_classes are provided
        # Single instance mode: assume pred_num_classes and true_num_classes are int
        if pred_num_classes is not None and true_num_classes is not None:
            # Directly use as int (no conversion needed)
            pred_num_val = int(pred_num_classes)
            true_num_val = int(true_num_classes)
            
            # Exact match accuracy
            num_classes_accuracy = 1.0 if pred_num_val == true_num_val else 0.0
            results["num_classes_accuracy"] = num_classes_accuracy
            
            # Mean absolute error for num_classes
            num_classes_mae = abs(pred_num_val - true_num_val)
            results["num_classes_mae"] = num_classes_mae
        
        if X_np is not None:
            X_flat = X_np.reshape(-1, X_np.shape[-1])
            internal_metrics = calculate_internal_metrics(X_flat, y_pred_flat)
            results.update(internal_metrics)
        
        return results
