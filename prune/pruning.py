"""Gradual pruning using first-order Taylor saliency and magnitude baseline."""
import numpy as np


def target_sparsity(step, total_steps, final_sparsity, initial_sparsity=0.0):
    """Cubic ramp: s(t)=s_f + (s_i-s_f)*(1-t/T)^3."""
    progress = min(max(step / total_steps, 0.0), 1.0)
    return initial_sparsity + (final_sparsity - initial_sparsity) * progress ** 3


def importance(weight, grad, criterion="saliency"):
    if criterion == "saliency":
        # |w * dL/dw| approximates first-order loss change from setting w to zero.
        return np.abs(weight * grad)
    if criterion == "magnitude":
        return np.abs(weight)
    raise ValueError("criterion must be 'saliency' or 'magnitude'")


def prune_to_sparsity(layer, optimizer, desired_sparsity, criterion="saliency"):
    """Monotonically prune active weights; bias is not pruned."""
    w = layer.weight.data
    current = layer.mask.copy()
    total = current.size
    desired_pruned = int(round(desired_sparsity * total))
    already_pruned = int((~current).sum())
    number_to_prune = max(0, desired_pruned - already_pruned)
    active_indices = np.flatnonzero(current)
    if number_to_prune == 0 or active_indices.size == 0:
        return
    scores = importance(w, layer.weight.grad, criterion).ravel()[active_indices]
    number_to_prune = min(number_to_prune, active_indices.size)
    remove_flat = active_indices[np.argsort(scores)[:number_to_prune]]
    new_mask = current.ravel().copy()
    new_mask[remove_flat] = False
    layer.mask = new_mask.reshape(current.shape)
    w[~layer.mask] = 0.0
    optimizer.set_mask(layer.weight, layer.mask)


def set_mask(layer, optimizer, mask):
    """Utility for explicit prune/revive tests and experiments."""
    layer.mask = np.asarray(mask, dtype=bool).copy()
    layer.weight.data[~layer.mask] = 0.0
    optimizer.set_mask(layer.weight, layer.mask)


def sparsity(layers):
    total = sum(layer.mask.size for layer in layers)
    active = sum(int(layer.mask.sum()) for layer in layers)
    return 1.0 - active / total if total else 0.0


def active_parameters(layers):
    return sum(int(layer.mask.sum()) for layer in layers)


def active_mac_estimate(layers, batch_size):
    return int(batch_size * sum(int(layer.mask.sum()) for layer in layers))
