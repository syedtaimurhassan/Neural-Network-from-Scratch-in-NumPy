import numpy as np


class Activation:
    """Base activation with forward/backward API."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLU(Activation):
    """Rectified Linear Unit."""

    def __init__(self) -> None:
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.where(self.mask, x, 0.0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.mask


class Sigmoid(Activation):
    """Sigmoid activation."""

    def __init__(self) -> None:
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_clipped = np.clip(x, -30, 30)
        self.out = 1.0 / (1.0 + np.exp(-x_clipped))
        return self.out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.out * (1.0 - self.out)


class Tanh(Activation):
    """Hyperbolic tangent activation."""

    def __init__(self) -> None:
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (1.0 - self.out ** 2)


class GeLU(Activation):
    """Gaussian Error Linear Unit using tanh approximation."""

    def __init__(self) -> None:
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        x = self.x
        tanh_arg = np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
        tanh_val = np.tanh(tanh_arg)
        left = 0.5 * (1.0 + tanh_val)
        right = 0.5 * x * (1.0 - tanh_val ** 2) * (np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x ** 2))
        return grad_output * (left + right)


class SELU(Activation):
    """Scaled Exponential Linear Unit."""

    def __init__(self) -> None:
        self.x = None
        self.scale = 1.0507009873554804934193349852946
        self.alpha = 1.6732632423543772848170429916717

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return self.scale * np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        dx = np.where(self.x > 0, 1.0, self.alpha * np.exp(self.x))
        return grad_output * self.scale * dx


def get_activation(name: str) -> Activation:
    """Factory for supported activations."""
    name = name.lower()
    if name == "relu":
        return ReLU()
    if name == "sigmoid":
        return Sigmoid()
    if name == "tanh":
        return Tanh()
    if name == "gelu":
        return GeLU()
    if name == "selu":
        return SELU()
    raise ValueError(f"Unsupported activation: {name}")
