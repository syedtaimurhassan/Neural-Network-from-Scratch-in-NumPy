import numpy as np


def _xavier_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Xavier uniform initializer."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))


def _he_init(fan_in: int, fan_out: int) -> np.ndarray:
    """He normal initializer."""
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std


class Dense:
    """Fully connected affine layer: y = xW + b."""

    def __init__(self, in_features: int, out_features: int, init: str = "xavier", l2_coeff: float = 0.0) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.l2_coeff = l2_coeff

        if init == "xavier":
            self.W = _xavier_init(in_features, out_features)
        elif init == "he":
            self.W = _he_init(in_features, out_features)
        else:
            raise ValueError(f"Unsupported init: {init}")
        self.b = np.zeros(out_features)

        self.x = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # grad_output shape: (batch, out_features)
        self.dW = self.x.T @ grad_output / self.x.shape[0]
        if self.l2_coeff > 0:
            self.dW += self.l2_coeff * self.W
        self.db = grad_output.mean(axis=0)
        grad_input = grad_output @ self.W.T
        return grad_input

    def parameters(self):
        return [(self.W, self.dW, self.l2_coeff), (self.b, self.db, 0.0)]


class Dropout:
    """Inverted dropout layer."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p <= 0.0:
            return x
        # Inverted dropout
        self.mask = (np.random.rand(*x.shape) >= self.p).astype(x.dtype) / (1.0 - self.p)
        return x * self.mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if not self.training or self.p <= 0.0:
            return grad_output
        return grad_output * self.mask

    def parameters(self):
        return []


class BatchNorm1d:
    """Batch normalization for 2D inputs (N, C)."""

    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-5) -> None:
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

        self.cache = None
        self.training = True
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            x_hat = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            self.cache = (x, x_hat, batch_mean, batch_var)
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * x_hat + self.beta

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        x, x_hat, mean, var = self.cache
        N, _ = x.shape
        self.dbeta = grad_output.sum(axis=0)
        self.dgamma = np.sum(grad_output * x_hat, axis=0)

        dxhat = grad_output * self.gamma
        dvar = np.sum(dxhat * (x - mean) * -0.5 * (var + self.eps) ** (-1.5), axis=0)
        dmean = np.sum(dxhat * -1 / np.sqrt(var + self.eps), axis=0) + dvar * np.mean(-2 * (x - mean), axis=0)
        dx = dxhat / np.sqrt(var + self.eps) + dvar * 2 * (x - mean) / N + dmean / N
        return dx

    def parameters(self):
        return [(self.gamma, self.dgamma, 0.0), (self.beta, self.dbeta, 0.0)]
