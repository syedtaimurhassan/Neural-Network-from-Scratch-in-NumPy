import numpy as np

from .activations import get_activation, Activation
from .layers import Dense, Dropout, BatchNorm1d


class Sequential:
    """Lightweight container executing modules in order."""

    def __init__(self, modules):
        self.modules = modules

    def train(self):
        for module in self.modules:
            if hasattr(module, "training"):
                module.training = True

    def eval(self):
        for module in self.modules:
            if hasattr(module, "training"):
                module.training = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        for module in reversed(self.modules):
            grad_output = module.backward(grad_output)
        return grad_output

    def parameters(self):
        params = []
        for module in self.modules:
            if hasattr(module, "parameters"):
                params.extend(module.parameters())
        return params


def build_mlp(
    input_dim: int,
    hidden_layers,
    output_dim: int,
    activation: str = "relu",
    init: str = "xavier",
    l2_coeff: float = 0.0,
    dropout: float = 0.0,
    batch_norm: bool = False,
) -> Sequential:
    """Build an MLP stack with optional batch norm and dropout."""
    modules = []
    prev_dim = input_dim
    for hidden_dim in hidden_layers:
        modules.append(Dense(prev_dim, hidden_dim, init=init, l2_coeff=l2_coeff))
        if batch_norm:
            modules.append(BatchNorm1d(hidden_dim))
        modules.append(get_activation(activation))
        if dropout > 0.0:
            modules.append(Dropout(dropout))
        prev_dim = hidden_dim
    modules.append(Dense(prev_dim, output_dim, init=init, l2_coeff=l2_coeff))
    return Sequential(modules)


class Trainer:
    """Handles training and evaluation loops."""

    def __init__(self, model: Sequential, loss_fn, optimizer, metrics_fn=None, grad_clip_norm: float = None) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics_fn = metrics_fn
        self.grad_clip_norm = grad_clip_norm

    def _step_batch(self, X: np.ndarray, y: np.ndarray):
        self.model.train()
        logits_or_pred = self.model.forward(X)
        loss = self.loss_fn.forward(logits_or_pred, y)
        grad_loss = self.loss_fn.backward()
        self.model.backward(grad_loss)
        if self.grad_clip_norm is not None:
            self._clip_gradients(self.grad_clip_norm)
        self.optimizer.step(self.model.parameters())
        metric = self.metrics_fn(logits_or_pred, y) if self.metrics_fn else None
        grad_norm = self._grad_norm()
        return loss, metric, grad_norm

    def train_epoch(self, dataloader):
        total_loss = 0.0
        total_metric = 0.0
        total_grad_norm = 0.0
        count = 0
        for X, y in dataloader:
            loss, metric, grad_norm = self._step_batch(X, y)
            total_loss += loss
            if metric is not None:
                total_metric += metric
            total_grad_norm += grad_norm
            count += 1
        avg_loss = total_loss / max(count, 1)
        avg_metric = total_metric / max(count, 1) if self.metrics_fn else None
        avg_grad_norm = total_grad_norm / max(count, 1)
        return avg_loss, avg_metric, avg_grad_norm

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        count = 0
        for X, y in dataloader:
            logits_or_pred = self.model.forward(X)
            loss = self.loss_fn.forward(logits_or_pred, y)
            if self.metrics_fn:
                total_metric += self.metrics_fn(logits_or_pred, y)
            total_loss += loss
            count += 1
        avg_loss = total_loss / max(count, 1)
        avg_metric = total_metric / max(count, 1) if self.metrics_fn else None
        return avg_loss, avg_metric

    def _grad_norm(self) -> float:
        sq_sum = 0.0
        for _, g, _ in self.model.parameters():
            if g is None:
                continue
            sq_sum += float(np.sum(g ** 2))
        return float(np.sqrt(sq_sum))

    def _clip_gradients(self, max_norm: float):
        norm = self._grad_norm()
        if norm == 0 or norm <= max_norm:
            return
        scale = max_norm / (norm + 1e-8)
        for _, g, _ in self.model.parameters():
            if g is None:
                continue
            g *= scale


def accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Compute accuracy for classification logits and integer targets."""
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == targets))


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error metric."""
    return float(np.mean((pred - target) ** 2))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error metric."""
    return float(np.mean(np.abs(pred - target)))
