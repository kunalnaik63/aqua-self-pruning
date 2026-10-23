"""Small NumPy-only reverse-mode autodiff engine for the AQUA challenge."""
from __future__ import annotations
import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Undo NumPy broadcasting by summing axes that were broadcast."""
    grad = np.asarray(grad, dtype=float)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    def __init__(self, data, parents=(), op=""):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data, dtype=float)
        self.parents = tuple(parents)
        self.op = op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data!r}, grad={self.grad!r}, op={self.op!r})"

    def _coerce(self, other):
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other):
        other = self._coerce(other)
        out = Tensor(self.data + other.data, (self, other), "add")
        def backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = backward
        return out
    __radd__ = __add__

    def __sub__(self, other):
        other = self._coerce(other)
        out = Tensor(self.data - other.data, (self, other), "sub")
        def backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad -= _unbroadcast(out.grad, other.data.shape)
        out._backward = backward
        return out

    def __rsub__(self, other):
        return self._coerce(other).__sub__(self)

    def __mul__(self, other):
        other = self._coerce(other)
        out = Tensor(self.data * other.data, (self, other), "mul")
        def backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = backward
        return out
    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._coerce(other)
        if np.any(other.data == 0):
            raise ZeroDivisionError("Tensor division by zero")
        out = Tensor(self.data / other.data, (self, other), "div")
        def backward():
            self.grad += _unbroadcast(out.grad / other.data, self.data.shape)
            other.grad += _unbroadcast(-self.data * out.grad / (other.data ** 2), other.data.shape)
        out._backward = backward
        return out

    def __rtruediv__(self, other):
        return self._coerce(other).__truediv__(self)

    def __neg__(self):
        return self * -1.0

    def __matmul__(self, other):
        other = self._coerce(other)
        out = Tensor(self.data @ other.data, (self, other), "matmul")
        def backward():
            # Handles vectors and matrices; unbroadcast handles batch dimensions.
            ga = out.grad @ np.swapaxes(other.data, -1, -2)
            gb = np.swapaxes(self.data, -1, -2) @ out.grad
            self.grad += _unbroadcast(ga, self.data.shape)
            other.grad += _unbroadcast(gb, other.data.shape)
        out._backward = backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")
        def backward():
            g = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                axes = tuple(a if a >= 0 else self.data.ndim + a for a in axes)
                for a in sorted(axes):
                    g = np.expand_dims(g, axis=a)
            self.grad += np.ones_like(self.data) * g
        out._backward = backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            count = int(np.prod([self.data.shape[a] for a in axes]))
        out = self.sum(axis=axis, keepdims=keepdims) / count
        out.op = "mean"
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,), "relu")
        def backward():
            self.grad += out.grad * (self.data > 0)
        out._backward = backward
        return out

    def tanh(self):
        value = np.tanh(self.data)
        out = Tensor(value, (self,), "tanh")
        def backward():
            self.grad += out.grad * (1.0 - value ** 2)
        out._backward = backward
        return out

    def sigmoid(self):
        # Stable sigmoid: separate positive/negative branches avoid overflow.
        x = self.data
        value = np.empty_like(x)
        pos = x >= 0
        value[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[~pos])
        value[~pos] = ex / (1.0 + ex)
        out = Tensor(value, (self,), "sigmoid")
        def backward():
            self.grad += out.grad * value * (1.0 - value)
        out._backward = backward
        return out

    def gelu(self):
        # Common tanh approximation to GELU and its matching derivative.
        x = self.data
        k = np.sqrt(2.0 / np.pi)
        u = k * (x + 0.044715 * x**3)
        t = np.tanh(u)
        value = 0.5 * x * (1.0 + t)
        out = Tensor(value, (self,), "gelu")
        def backward():
            du = k * (1.0 + 3.0 * 0.044715 * x**2)
            derivative = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * du
            self.grad += out.grad * derivative
        out._backward = backward
        return out

    def softmax(self, axis=-1):
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / exps.sum(axis=axis, keepdims=True)
        out = Tensor(probs, (self,), "softmax")
        def backward():
            # Jacobian-vector product without materializing the Jacobian.
            dot = np.sum(out.grad * probs, axis=axis, keepdims=True)
            self.grad += probs * (out.grad - dot)
        out._backward = backward
        return out

    def backward(self):
        ordered, visited = [], set()
        def visit(node):
            if id(node) not in visited:
                visited.add(id(node))
                for parent in node.parents:
                    visit(parent)
                ordered.append(node)
        visit(self)
        # A fresh backward pass should not silently add gradients from a prior pass.
        for node in ordered:
            node.grad = np.zeros_like(node.data, dtype=float)
        self.grad = np.ones_like(self.data, dtype=float)
        for node in reversed(ordered):
            node._backward()


def softmax_cross_entropy(logits: Tensor, labels) -> Tensor:
    """Stable fused softmax + mean cross-entropy for integer class labels."""
    y = np.asarray(labels, dtype=int)
    z = logits.data
    if z.ndim != 2 or y.shape != (z.shape[0],):
        raise ValueError("Expected logits (batch, classes) and labels (batch,)")
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    probs = exp_z / exp_z.sum(axis=1, keepdims=True)
    n = z.shape[0]
    loss_value = -np.log(np.clip(probs[np.arange(n), y], 1e-15, 1.0)).mean()
    out = Tensor(loss_value, (logits,), "softmax_cross_entropy")
    def backward():
        grad = probs.copy()
        grad[np.arange(n), y] -= 1.0
        logits.grad += grad * (out.grad / n)
    out._backward = backward
    return out
