"""Utilities for loading prior configuration modules at runtime."""

from __future__ import annotations

import copy
import importlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Tuple


DEFAULT_MODULE = "configs.prior_config"


class PriorConfigNotFoundError(ImportError):
    """Raised when the specified prior configuration cannot be loaded."""


def _load_module_from_path(path: Path) -> ModuleType:
    path = path.expanduser().resolve()
    if not path.exists():
        raise PriorConfigNotFoundError(f"prior_config file not found: {path}")

    module_name = f"_prior_config_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise PriorConfigNotFoundError(f"can not load prior_config: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_module_name(name: str) -> str:
    module_name = name.strip().replace("/", ".")
    if module_name.endswith(".py"):
        module_name = module_name[:-3]
    if module_name.startswith("."):
        raise PriorConfigNotFoundError(f"prior_config module name should not start with '.': {name}")
    if module_name.startswith("configs."):
        return module_name
    return f"configs.{module_name}"


def _load_module(spec: str | None) -> Tuple[ModuleType, str]:
    if spec is None:
        module = importlib.import_module(DEFAULT_MODULE)
        return module, DEFAULT_MODULE

    if os.path.sep in spec or spec.endswith(".py"):
        module = _load_module_from_path(Path(spec))
        return module, str(Path(spec).expanduser().resolve())

    module_name = _normalize_module_name(spec)
    module = importlib.import_module(module_name)
    return module, module_name


def load_prior_config(spec: str | None) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Load prior configuration and return deep copies of fixed/sampled hyperparameters.

    Parameters
    ----------
    spec: str or None
        Optional way to specify the `prior_config`. It can be a module name
        (e.g., "prior_config_mix", "configs.custom_prior") or a `.py` file path.
        If None, it falls back to the default `configs.prior_config`.
    
    Returns
    -------
    fixed_hp: Dict[str, Any]
        Deep copy of fixed hyperparameters.
    sampled_hp: Dict[str, Any]
        Deep copy of sampled hyperparameters.
    source: str
        The actual loaded module path or name, for logging and debugging.
    """

    module, source = _load_module(spec)

    if not hasattr(module, "DEFAULT_FIXED_HP"):
        raise AttributeError(f"prior_config module is missing DEFAULT_FIXED_HP: {source}")
    if not hasattr(module, "DEFAULT_SAMPLED_HP"):
        raise AttributeError(f"prior_config module is missing DEFAULT_SAMPLED_HP: {source}")

    fixed_hp = copy.deepcopy(getattr(module, "DEFAULT_FIXED_HP"))
    sampled_hp = copy.deepcopy(getattr(module, "DEFAULT_SAMPLED_HP"))

    return fixed_hp, sampled_hp, source

