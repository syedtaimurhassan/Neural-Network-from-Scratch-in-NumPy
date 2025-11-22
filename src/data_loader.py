import gzip
import pickle
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np


def _load_idx_images(path: Path) -> np.ndarray:
    """Load IDX image file (Fashion-MNIST)."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic, num_images, rows, cols = np.frombuffer(data[:16], dtype=">i4")
    if magic != 2051:
        raise ValueError(f"Unexpected magic number for images: {magic}")
    images = np.frombuffer(data[16:], dtype=np.uint8).reshape(num_images, rows, cols)
    return images


def _load_idx_labels(path: Path) -> np.ndarray:
    """Load IDX label file (Fashion-MNIST)."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic, num_items = np.frombuffer(data[:8], dtype=">i4")
    if magic != 2049:
        raise ValueError(f"Unexpected magic number for labels: {magic}")
    labels = np.frombuffer(data[8:], dtype=np.uint8)
    return labels[:num_items]


def load_fashion_mnist(data_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return Fashion-MNIST images and labels for train or test split."""
    data_dir = Path(data_dir) / "fashion-mnist"
    if split == "train":
        images_path = data_dir / "train-images-idx3-ubyte.gz"
        labels_path = data_dir / "train-labels-idx1-ubyte.gz"
    elif split == "test":
        images_path = data_dir / "t10k-images-idx3-ubyte.gz"
        labels_path = data_dir / "t10k-labels-idx1-ubyte.gz"
    else:
        raise ValueError("split must be 'train' or 'test'")

    images = _load_idx_images(images_path)
    labels = _load_idx_labels(labels_path)
    return images, labels


def _load_cifar_batch(path: Path):
    """Load one CIFAR-10 batch file."""
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="latin1")
    data = batch["data"].reshape(-1, 3, 32, 32)
    labels = np.array(batch["labels"], dtype=np.int64)
    return data, labels


def load_cifar10(data_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return CIFAR-10 data and labels for train or test split."""
    data_root = Path(data_dir) / "cifar-10" / "cifar-10-batches-py"
    if split == "train":
        batches = [data_root / f"data_batch_{i}" for i in range(1, 6)]
    elif split == "test":
        batches = [data_root / "test_batch"]
    else:
        raise ValueError("split must be 'train' or 'test'")

    data_list = []
    label_list = []
    for batch_path in batches:
        data, labels = _load_cifar_batch(batch_path)
        data_list.append(data)
        label_list.append(labels)
    return np.concatenate(data_list, axis=0), np.concatenate(label_list, axis=0)


def make_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield mini-batches from arrays."""
    indices = np.arange(len(X))
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]
