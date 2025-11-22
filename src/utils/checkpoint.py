import numpy as np
from pathlib import Path
from typing import Any


def save_model(model, path: Path):
    """Save model parameters to npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = model.parameters()
    data = {f"p{i}": p for i, (p, _g, _l2) in enumerate(params)}
    np.savez(path, **data)


def load_model(model, path: Path):
    """Load model parameters from npz."""
    path = Path(path)
    loaded = np.load(path, allow_pickle=False)
    params = model.parameters()
    for i, (p, _g, _l2) in enumerate(params):
        key = f"p{i}"
        if key not in loaded:
            raise KeyError(f"Missing parameter {key} in checkpoint.")
        p[:] = loaded[key]
