"""Adam with explicit mask transitions and stale-state clearing."""
import numpy as np


class Adam:
    def __init__(self, params, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m = {id(p): np.zeros_like(p.data) for p in self.params}
        self.v = {id(p): np.zeros_like(p.data) for p in self.params}
        # Per-coordinate age lets revived weights restart Adam bias correction.
        self.steps = {id(p): np.zeros_like(p.data, dtype=np.int64) for p in self.params}
        self.masks = {}

    def set_mask(self, param, new_mask):
        """Register/update a parameter mask; clear Adam state on any transition."""
        new_mask = np.asarray(new_mask, dtype=bool)
        if new_mask.shape != param.data.shape:
            raise ValueError("mask shape must match parameter shape")
        key = id(param)
        old = self.masks.get(key, np.ones_like(new_mask, dtype=bool))
        changed = old != new_mask
        self.m[key][changed] = 0.0
        self.v[key][changed] = 0.0
        self.steps[key][changed] = 0
        self.masks[key] = new_mask.copy()
        # Enforce exact zero for currently masked entries.
        param.data[~new_mask] = 0.0

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self):
        self.t += 1
        for p in self.params:
            key = id(p)
            g = p.grad.copy()
            mask = self.masks.get(key)
            if mask is not None:
                g[~mask] = 0.0
                self.m[key][~mask] = 0.0
                self.v[key][~mask] = 0.0
                self.steps[key][~mask] = 0
                self.steps[key][mask] += 1
            else:
                self.steps[key] += 1
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g * g)
            m_denom = np.where(self.steps[key] > 0, 1 - np.power(self.beta1, self.steps[key]), 1.0)
            v_denom = np.where(self.steps[key] > 0, 1 - np.power(self.beta2, self.steps[key]), 1.0)
            mhat = self.m[key] / m_denom
            vhat = self.v[key] / v_denom
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
            if mask is not None:
                p.data[~mask] = 0.0
