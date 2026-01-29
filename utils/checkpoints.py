import os
import torch
from typing import Optional, Type

def get_latest_checkpoint(ckpt_dir: str, version: Optional[int] = None) -> Optional[str]:
    """
    Get the path to the latest checkpoint file in the specified directory.
    Args:
        ckpt_dir (str): Checkpoint directory.
        version (Optional[int]): Version number. If provided, search within that version directory.
    Returns:
        Optional[str]: Path to the latest checkpoint file, or None if not found.
    """
    if version is not None:
        ckpt_dir = os.path.join(ckpt_dir, f"version_{version}")
    if not os.path.isdir(ckpt_dir):
        return None
    checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]
    if not checkpoints:
        return None
    try:
        latest_ckpt = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]), reverse=True)[0]
        return os.path.join(ckpt_dir, latest_ckpt)
    except Exception as e:
        print(f"Error parsing checkpoint filenames: {e}")
        return None

def load_model(model_class: Type, checkpoint_path: Optional[str], checkpoint_dir: Optional[str] = None, version: Optional[int] = None, device: str = "cpu", strict: bool = True, config: Optional[dict] = None):
    """
    Generic model loading function.
    Args:
        checkpoint_path (Optional[str]): Path to the model checkpoint file. If None, uses checkpoint_dir and version to find it.
        device (str): Target device for loading the model ('cpu' or 'cuda').
        model_class (Type): Model class, e.g., TabClusterMAB.
        checkpoint_dir (Optional[str]): Checkpoint directory. If provided and checkpoint_path is None, finds the latest checkpoint.
        version (Optional[int]): Version number.
        strict (bool): Whether to strictly match weights, default True. Set to False to allow partial weight loading.
        config (Optional[dict]): Model configuration. Used if checkpoint does not contain config.
    Returns:
        model_class: Model instance with loaded weights.
    """
    if checkpoint_path is None and checkpoint_dir is not None:
        checkpoint_path = get_latest_checkpoint(checkpoint_dir, version)
    if checkpoint_path is None:
        raise ValueError("Either checkpoint_path or checkpoint_dir must be provided.")
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "config" not in checkpoint:
        if config is None:
            raise ValueError("Checkpoint does not contain 'config' key and no config provided.")
        model_config = config
    else:
        model_config = checkpoint["config"]
    model = model_class(**model_config)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
    else:
        raise ValueError("Checkpoint does not contain 'state_dict' key.")
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model
