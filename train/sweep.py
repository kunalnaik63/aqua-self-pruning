"""Run multi-seed sparsity/accuracy experiments and save raw CSV plus Pareto plot."""
import argparse, csv, os
import numpy as np
from train.experiment import run_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default="results/sweep")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for criterion in ("magnitude", "saliency"):
        for target in (0.0, 0.50, 0.75, 0.90, 0.95):
            for seed in range(args.seeds):
                result, _ = run_one(seed=seed, final_sparsity=target, criterion=criterion,
                                    epochs=args.epochs, prune=(target > 0))
                rows.append(result)
                print(criterion, target, seed, round(result['test_accuracy'], 4), round(result['achieved_sparsity'], 4))
    fields = list(rows[0].keys())
    with open(os.path.join(args.out, "pareto_raw.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for criterion, marker in (("magnitude", "o"), ("saliency", "s")):
            subset = [r for r in rows if r["criterion"] == criterion]
            grouped = sorted(set(r["target_sparsity"] for r in subset))
            means = [np.mean([r["test_accuracy"] for r in subset if r["target_sparsity"] == s]) for s in grouped]
            stds = [np.std([r["test_accuracy"] for r in subset if r["target_sparsity"] == s]) for s in grouped]
            achieved = [np.mean([r["achieved_sparsity"] for r in subset if r["target_sparsity"] == s]) for s in grouped]
            ax.errorbar(achieved, means, yerr=stds, marker=marker, capsize=3, label=criterion)
        ax.set_xlabel("Achieved sparsity")
        ax.set_ylabel("Test accuracy")
        ax.set_title("Sparsity–accuracy Pareto curve (mean ± seed std)")
        ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(args.out, "pareto.png"), dpi=160)
        plt.close(fig)
    except ImportError:
        print("matplotlib not installed; raw CSV was still written")

if __name__ == "__main__": main()
