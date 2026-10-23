import argparse, csv, json, os
import numpy as np
from engine.tensor import Tensor, softmax_cross_entropy
from nn.layers import MLP
from nn.optim import Adam
from prune.pruning import prune_to_sparsity, sparsity, active_parameters, active_mac_estimate, target_sparsity
from train.data import make_spirals, train_test_split


def evaluate(model, X, y):
    logits = model(Tensor(X)).data
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted); probs /= probs.sum(axis=1, keepdims=True)
    loss = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1)).mean()
    return float(loss), float((probs.argmax(axis=1) == y).mean())


def run_one(seed=0, final_sparsity=0.0, criterion="saliency", epochs=80, hidden=32, prune=True):
    X, y = make_spirals(seed=seed)
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=seed + 100)
    # Standardize using training split only.
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    model = MLP(2, hidden, 3, seed=seed + 200)
    optimizer = Adam(model.parameters(), lr=0.01)
    for layer in model.prunable_layers():
        optimizer.set_mask(layer.weight, layer.mask)
    batch_size = 48
    history = []
    for epoch in range(1, epochs + 1):
        order = np.random.default_rng(seed * 1000 + epoch).permutation(len(ytr))
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(Tensor(Xtr[ids]))
            loss = softmax_cross_entropy(logits, ytr[ids])
            loss.backward()
            optimizer.step()
        # Gradual pruning after warmup. Gradients from the final minibatch supply saliency.
        if prune and final_sparsity > 0 and epoch >= max(5, epochs // 10):
            # Estimate saliency on the full training split, not only the final minibatch.
            optimizer.zero_grad()
            full_loss = softmax_cross_entropy(model(Tensor(Xtr)), ytr)
            full_loss.backward()
            warmup = max(5, epochs // 10)
            ramp_step = epoch - warmup + 1
            ramp_total = epochs - warmup + 1
            scheduled = target_sparsity(ramp_step, ramp_total, final_sparsity)
            for layer in model.prunable_layers():
                prune_to_sparsity(layer, optimizer, scheduled, criterion)
        train_loss, train_acc = evaluate(model, Xtr, ytr)
        test_loss, test_acc = evaluate(model, Xte, yte)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                        "test_loss": test_loss, "test_accuracy": test_acc,
                        "sparsity": sparsity(model.prunable_layers())})
        if not np.isfinite(test_loss):
            raise FloatingPointError("Non-finite loss detected")
    layers = model.prunable_layers()
    result = {"seed": seed, "criterion": criterion, "target_sparsity": final_sparsity,
              "achieved_sparsity": sparsity(layers), "active_parameters": active_parameters(layers),
              "total_prunable_parameters": sum(l.mask.size for l in layers),
              "estimated_active_macs_per_example": active_mac_estimate(layers, 1),
              "test_loss": history[-1]["test_loss"], "test_accuracy": history[-1]["test_accuracy"],
              "train_accuracy": history[-1]["train_accuracy"], "epochs": epochs, "hidden_dim": hidden}
    return result, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparsity", type=float, default=0.0)
    parser.add_argument("--criterion", choices=["saliency", "magnitude"], default="saliency")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--no-prune", action="store_true")
    parser.add_argument("--out", default="results/single_run")
    args = parser.parse_args()
    result, history = run_one(args.seed, args.sparsity, args.criterion, args.epochs, args.hidden, not args.no_prune)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics.json"), "w") as f: json.dump(result, f, indent=2)
    with open(os.path.join(args.out, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    # Save learning curves alongside raw per-epoch values.
    try:
        import matplotlib.pyplot as plt
        epochs_axis = [r["epoch"] for r in history]
        fig, ax = plt.subplots()
        ax.plot(epochs_axis, [r["train_loss"] for r in history], label="train loss")
        ax.plot(epochs_axis, [r["test_loss"] for r in history], label="test loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
        ax.set_title("Learning curve"); ax.grid(True, alpha=0.25); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "learning_curve.png"), dpi=160); plt.close(fig)
        fig, ax = plt.subplots()
        ax.plot(epochs_axis, [r["train_accuracy"] for r in history], label="train accuracy")
        ax.plot(epochs_axis, [r["test_accuracy"] for r in history], label="test accuracy")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy curve"); ax.grid(True, alpha=0.25); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "accuracy_curve.png"), dpi=160); plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
