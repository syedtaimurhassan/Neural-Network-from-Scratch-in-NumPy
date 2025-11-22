import numpy as np


class SGD:
    """Stochastic gradient descent with optional weight decay."""

    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0) -> None:
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, params):
        for p, g, _l2 in params:
            grad = g + self.weight_decay * p if self.weight_decay else g
            p -= self.lr * grad


class Momentum:
    """SGD with momentum."""

    def __init__(self, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0) -> None:
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}

    def step(self, params):
        for p, g, _l2 in params:
            grad = g + self.weight_decay * p if self.weight_decay else g
            key = id(p)
            v = self.velocity.get(key, np.zeros_like(p))
            v = self.momentum * v - self.lr * grad
            p += v
            self.velocity[key] = v


class Adam:
    """Adam optimizer."""

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params):
        self.t += 1
        for p, g, _l2 in params:
            grad = g + self.weight_decay * p if self.weight_decay else g
            key = id(p)
            m = self.m.get(key, np.zeros_like(p))
            v = self.v.get(key, np.zeros_like(p))

            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad * grad)

            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            self.m[key] = m
            self.v[key] = v


class NAdam:
    """Nesterov-accelerated Adam."""

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params):
        self.t += 1
        for p, g, _l2 in params:
            grad = g + self.weight_decay * p if self.weight_decay else g
            key = id(p)
            m = self.m.get(key, np.zeros_like(p))
            v = self.v.get(key, np.zeros_like(p))

            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad * grad)

            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)

            nesterov_m = self.beta1 * m_hat + (1.0 - self.beta1) * grad / (1.0 - self.beta1 ** self.t)
            p -= self.lr * nesterov_m / (np.sqrt(v_hat) + self.eps)

            self.m[key] = m
            self.v[key] = v


def get_optimizer(name: str, **kwargs):
    """Return optimizer instance by name."""
    name = name.lower()
    if name == "sgd":
        return SGD(**kwargs)
    if name == "momentum":
        return Momentum(**kwargs)
    if name == "adam":
        return Adam(**kwargs)
    if name == "nadam":
        return NAdam(**kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")
