"""MLP layers. Linear uses a sparse-aware forward/backward path."""
import numpy as np
from engine.tensor import Tensor


class Linear:
    def __init__(self, in_features, out_features, rng, init="he"):
        self.in_features = in_features
        self.out_features = out_features
        if init == "he":
            scale = np.sqrt(2.0 / in_features)
        elif init == "xavier":
            scale = np.sqrt(2.0 / (in_features + out_features))
        else:
            raise ValueError("init must be 'he' or 'xavier'")
        self.weight = Tensor(rng.normal(0.0, scale, (in_features, out_features)))
        self.bias = Tensor(np.zeros(out_features))
        self.mask = np.ones((in_features, out_features), dtype=bool)
        self.active_multiply_accumulates = 0

    def __call__(self, x: Tensor) -> Tensor:
        if x.data.ndim != 2 or x.data.shape[1] != self.in_features:
            raise ValueError(f"Expected input (batch, {self.in_features})")
        batch = x.data.shape[0]
        y_data = np.zeros((batch, self.out_features), dtype=float)
        active = int(self.mask.sum())
        # Sparse-aware: only active weight coordinates participate in arithmetic.
        for j in range(self.out_features):
            idx = np.flatnonzero(self.mask[:, j])
            if idx.size:
                y_data[:, j] = x.data[:, idx] @ self.weight.data[idx, j]
        y_data += self.bias.data
        self.active_multiply_accumulates = batch * active
        out = Tensor(y_data, (x, self.weight, self.bias), "sparse_linear")

        def backward():
            gy = out.grad
            # Bias participates for every example.
            self.bias.grad += gy.sum(axis=0)
            # Compute only active connections; masked weight gradients remain zero.
            for j in range(self.out_features):
                idx = np.flatnonzero(self.mask[:, j])
                if idx.size:
                    x.grad[:, idx] += gy[:, j:j+1] * self.weight.data[idx, j][None, :]
                    self.weight.grad[idx, j] += x.data[:, idx].T @ gy[:, j]
        out._backward = backward
        return out

    def parameters(self):
        return [self.weight, self.bias]


class MLP:
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=3, seed=0):
        rng = np.random.default_rng(seed)
        self.layers = [Linear(input_dim, hidden_dim, rng, "he"),
                       Linear(hidden_dim, output_dim, rng, "xavier")]

    def __call__(self, x):
        x = self.layers[0](x).relu()
        return self.layers[1](x)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def prunable_layers(self):
        return self.layers
