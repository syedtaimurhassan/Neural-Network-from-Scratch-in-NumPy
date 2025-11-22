from typing import Tuple

import numpy as np


def normalize_images(X: np.ndarray) -> np.ndarray:
    """Scale image pixels to [0,1]."""
    return X.astype(np.float32) / 255.0


def flatten_images(X: np.ndarray) -> np.ndarray:
    """Flatten images to vectors."""
    return X.reshape(X.shape[0], -1)


def train_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split into train/val sets."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    val_size = int(len(X) * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def standardize_per_channel(X: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None):
    """Standardize per channel; returns standardized data and stats."""
    if mean is None or std is None:
        mean = X.mean(axis=(0, 2, 3), keepdims=True)
        std = X.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    X_std = (X - mean) / std
    return X_std, mean, std


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot encode integer labels."""
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out
