import numpy as np


class CrossEntropyLoss:
    """Cross-entropy loss with logits support."""

    def __init__(self) -> None:
        self.probs = None
        self.y_true = None

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.probs = exp / exp.sum(axis=1, keepdims=True)

        if y_true.ndim == 1:
            self.y_true = y_true
            log_likelihood = -np.log(self.probs[np.arange(len(y_true)), y_true] + 1e-12)
        else:
            self.y_true = y_true
            log_likelihood = -np.sum(y_true * np.log(self.probs + 1e-12), axis=1)
        return float(log_likelihood.mean())

    def backward(self) -> np.ndarray:
        if self.probs is None or self.y_true is None:
            raise RuntimeError("Must call forward before backward.")
        batch_size = self.probs.shape[0]
        grad = self.probs.copy()
        if self.y_true.ndim == 1:
            grad[np.arange(batch_size), self.y_true] -= 1.0
        else:
            grad -= self.y_true
        return grad / batch_size


class MSELoss:
    """Mean squared error loss."""

    def __init__(self) -> None:
        self.pred = None
        self.target = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self.pred = pred
        self.target = target
        return float(np.mean((pred - target) ** 2))

    def backward(self) -> np.ndarray:
        if self.pred is None or self.target is None:
            raise RuntimeError("Must call forward before backward.")
        return 2.0 * (self.pred - self.target) / self.pred.shape[0]
