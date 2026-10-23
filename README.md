# AQUA Platform — Self-Pruning Network

A small NumPy-only reverse-mode autodiff engine, MLP, Adam optimizer, and gradual pruning experiment. This repository is a reproducible starting implementation for the AQUA take-home; the author should understand, run, inspect, and be prepared to modify every part before submitting.

## What is implemented

- `engine/tensor.py`: Tensor wrapper around `numpy.ndarray`; reverse-mode graph; addition, subtraction, multiplication, division, matrix multiplication, sum, mean, ReLU, tanh, sigmoid, tanh-approximate GELU, stable softmax, and fused stable softmax-cross-entropy.
- `nn/layers.py`: MLP and a sparse-aware `Linear`. Its forward and backward loops visit only active weight coordinates; it does not run a dense matrix multiplication and then multiply by a mask.
- `nn/optim.py`: Adam with first/second moments, binary masks, and per-coordinate Adam age. Moment estimates and age are cleared on mask transitions so a revived connection does not reuse stale optimizer history.
- `prune/pruning.py`: gradual cubic schedule helper; magnitude baseline; first-order Taylor saliency `abs(weight * gradient)`; exact binary masks.
- `train/`: deterministic three-class spiral dataset, minibatched training, evaluation, single-run outputs, multi-seed sweep, CSV data, and plots.
- `tests/`: finite-difference gradient checks and a masked/pruned/revived connection test.

Only NumPy is used for the core engine, model, optimizer, pruning, and training. Matplotlib is used for plotting. No PyTorch, TensorFlow, JAX, Keras, Hugging Face, or scikit-learn is used. There is no external dataset loader.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Run commands from the repository root so Python imports resolve.

## Reproduction commands

```bash
# Part 1: numerical gradient checks and masked-weight test
python -m unittest discover -s tests -v

# Part 2: dense baseline training (writes metrics, per-epoch CSV, and learning curves)
python -m train.experiment --seed 0 --sparsity 0 --no-prune --epochs 80 --out results/baseline

# Part 3: gradual 90% pruning with Taylor saliency
python -m train.experiment --seed 0 --sparsity 0.90 --criterion saliency --epochs 80 --out results/saliency90

# Part 4: compare magnitude vs Taylor saliency at 0/50/75/90/95%, across 3 seeds
python -m train.sweep --epochs 60 --seeds 3 --out results/sweep
```

The sweep writes `results/sweep/pareto_raw.csv` and `results/sweep/pareto.png`. Each single run writes `metrics.json`, `history.csv`, `learning_curve.png`, and `accuracy_curve.png`. Seeds are fixed and passed explicitly.

## Results included in this repository

`results/sweep/pareto_raw.csv` and `results/sweep/pareto.png` contain a run with 60 epochs and 3 seeds per criterion/target. The small spiral dataset is synthetic and results are not claims about production workloads. At 50% achieved sparsity, the saved run produced mean test accuracy 95.80% (population standard deviation 1.85 percentage points) for saliency versus 88.15% (2.77 pp) for magnitude, a 7.65 pp difference in this experiment. This is a falsifiable, dataset-specific result, not proof that saliency always wins. At higher sparsity the variance and accuracy cost are material; inspect the raw numbers.

The reported active MAC estimate is `batch_size * active_weight_count` for one forward pass through the prunable linear layers, treating each active weight as one multiply-accumulate per example. This is an operation-count proxy, not a wall-clock speedup claim. The sparse-aware implementation skips masked coordinates, but Python-loop overhead means it may not be faster than NumPy's optimized dense BLAS for small layers. No latency improvement is claimed.

## Important limitations to disclose

- This is a compact educational engine, not a production framework. It prioritizes correctness and inspectability over speed.
- The engine's `matmul` gradient is intended for ordinary vector/matrix use in this project; the automated test covers 2-D matrix multiplication, not every possible NumPy batched/broadcasted matmul signature.
- Pruning is monotonic in the experiment. The optimizer and explicit mask utility support reactivating a connection for the correctness test, but the training sweep does not implement a learned regrowth policy.
- The Taylor score `|w * dL/dw|` is a first-order approximation; interactions between weights and nonlinear effects are not captured exactly.
- Active MAC counts are not benchmarked latency. Do not describe the count as a measured speedup.

## Architecture

```text
engine/tensor.py       Tensor + operation-specific backward rules
nn/layers.py           sparse-aware Linear + MLP
nn/optim.py            Adam state and mask-transition handling
prune/pruning.py       importance criteria + gradual mask updates
train/experiment.py    minibatches, loss, evaluation, curves
train/sweep.py         seeds × sparsity × criterion, raw CSV + Pareto plot
tests/test_engine.py   finite differences + masked/revived-weight test
DESIGN.md              derivations and engineering trade-offs
```
