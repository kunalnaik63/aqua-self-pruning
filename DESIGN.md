# Design notes

## 1. Importance criterion and loss-change approximation

For a scalar connection weight `w`, removing the connection means changing it to zero, so `Δw = -w`. A first-order Taylor expansion around the current parameter gives

`L(w + Δw) ≈ L(w) + (∂L/∂w) Δw`.

Substituting `Δw = -w`, the estimated loss change is `ΔL ≈ -w (∂L/∂w)`. Its magnitude is `|w * ∂L/∂w|`. The implementation uses `abs(weight * gradient)` as a saliency score and removes the lowest-scoring currently active weights. This approximates the magnitude of the first-order loss change if a connection is removed. It is not exact: second-order curvature, interactions among simultaneously removed weights, and the fact that scores change during training can affect the outcome.

The baseline is magnitude pruning, `abs(weight)`, which does not account for the current loss gradient. Both criteria are compared at the same target sparsities and seeds in `results/sweep/pareto_raw.csv`.

## 2. Gradient of a masked weight

The forward operation is conceptually `W_eff = W * M`, where `M` is a binary mask. For a masked coordinate, `M_ij = 0`; therefore the effective weight contributes zero to the forward computation and the derivative of the effective weight with respect to the underlying parameter is also multiplied by zero. In this implementation, `Linear` explicitly visits only active coordinates in both forward and backward passes. Its backward rule never writes a gradient for a masked coordinate, so that gradient remains zero after a fresh backward pass.

This is the chosen convention while a connection is masked: it receives zero gradient and cannot update. If a connection is explicitly revived, the mask becomes one and it participates in forward/backward again. The test checks both states.

## 3. Adam state for pruned and revived connections

Adam keeps first and second moments (`m` and `v`). If a coordinate is pruned, retaining its old moments would allow stale history to influence it if it were later revived. `Adam.set_mask` detects every mask transition and clears `m`, `v`, and that coordinate's bias-correction age. The optimizer also zeros masked gradients and parameters. Per-coordinate step counts are used for bias correction so a revived connection starts with a fresh first-step correction instead of inheriting the global training age. `test_masked_gradient_and_adam_revival_state` seeds nonzero moments, prunes a coordinate, verifies zero gradient/value/state, revives it, verifies clean state, and verifies it can update again.

There is a design choice here: clearing state discards useful history, but is predictable and prevents stale momentum from corrupting a revived connection. An alternative is to preserve state and explicitly justify that behavior; this implementation chooses reset-on-transition.

## 4. Pruning schedule and hard sparsity target

The helper `target_sparsity` implements a cubic ramp:

`s(t) = s_i + (s_f - s_i) (t/T)^3`, where `t/T` is clamped to `[0, 1]`. With `s_i = 0`, the target rises smoothly from zero to the final target.

The training loop uses a warmup period, then increases the target sparsity cubically. Gradual pruning gives the network time to adapt compared with removing the final fraction in one step. The experiment applies the target separately to each linear layer's weight matrix; achieved overall sparsity is reported from actual masks, not inferred from the requested target. Biases are not pruned.

## 5. Bottlenecks and possible optimizations

The autodiff engine uses Python objects and a Python backward closure per operation. The sparse-aware `Linear` loops over output columns and active input indices; this is easy to inspect but incurs Python overhead. Improvements include vectorized sparse kernels, CSR/CSC storage, fused linear+activation operations, graph-node memory reuse, compiled kernels, and batching many small operations. For a production service, sparse format selection should be benchmarked against dense BLAS because high sparsity alone does not guarantee lower latency.

## 6. Serving in a multi-tenant inference service

I would export a versioned inference artifact containing architecture, weights, mask/sparse representation, label/config metadata, and a checksum. A model registry would track validation metrics and compatibility. A scheduler would batch requests by model/version and latency budget; tenant isolation would include per-tenant quotas, admission control, and separate resource accounting. The inference worker would load immutable model versions, expose latency/throughput/error/sparsity metrics, and support canary rollout and rollback. Model execution would run with bounded CPU/memory, timeouts, request validation, and audit logs. I would measure p50/p95/p99 latency and throughput under realistic concurrent load before claiming a cost reduction.

## 7. Evidence and falsifiable claim

The committed sweep uses a synthetic three-class spiral dataset, fixed seeds 0–2, 60 epochs, target sparsities 0%, 50%, 75%, 90%, and 95%, and both magnitude and saliency criteria. The raw run-level results are in `results/sweep/pareto_raw.csv`; `pareto.png` shows mean test accuracy with one population-standard-deviation error bars.

Falsifiable claim for this exact experiment: at 50% achieved sparsity, Taylor saliency achieved mean test accuracy 95.80% across three seeds versus 88.15% for magnitude pruning, a 7.65 percentage-point advantage. The small sample size and synthetic dataset limit generalization; a different seed set, dataset, or training budget may change the conclusion. This is not evidence that saliency always outperforms magnitude pruning.

The active MAC estimate is a count of active weight multiply-accumulates per example. It is not wall-clock latency and is not presented as a measured speedup. The sparse-aware linear implementation skips masked coordinates; its real runtime advantage still needs hardware-specific benchmarking.
