# Reproducibility Protocol

## Experimental contract

The repository treats the classification task as **binary disease screening**. The original PlantVillage class is retained as metadata and is never discarded.

Primary target:

- `0`: Healthy
- `1`: Diseased

The test split is a locked evaluation set. It must not be used for:

- hyperparameter selection;
- GOA fitness;
- AAO fitness;
- early stopping;
- temperature fitting;
- decision-threshold selection;
- optimizer comparison selection.

## Split policy

The default split is 70/15/15.

Splitting is performed with stratification on `original_class`, not on the merged binary label alone. This ensures every available original class is represented proportionally.

The preparation script computes SHA-256 hashes. A file with the same hash may not appear in more than one split.

## Seeds

Default seeds:

```text
17, 23, 41, 59, 83
```

The seed is applied to Python, NumPy, and PyTorch. CUDA deterministic behavior is enabled where supported.

## Primary result

The authoritative test result is the metric recomputed from `test_predictions.csv`. Aggregate tables are derived from this file.

A valid test-prediction file contains:

```text
path
original_class
binary_label
prob_diseased
predicted_label
decision_threshold
seed
model_variant
```

## Confidence intervals

The default confidence interval is a stratified bootstrap over test observations, 2000 resamples, 95% interval.

## Statistical testing

- McNemar: paired test-set predictions from two classifiers on the same examples.
- Wilcoxon signed-rank: paired repeated-run metrics.
- Friedman + Nemenyi-ready ranks: multi-model repeated-run comparisons.
- Exact p-values/statistics are stored; statements such as only “p < 0.05” are insufficient as the sole record.

## Calibration

Temperature scaling is fitted using the validation split only.

Metrics:

- ECE (15 bins by default);
- Brier score;
- NLL;
- reliability-curve points.

## GOA fairness

GOA candidate masks are evaluated on the same frozen validation feature matrix. The test split is not loaded inside the GOA objective.

The default feature-selection objective is:

```text
alpha * balanced_accuracy + beta * reduction_ratio
```

with `alpha=0.95`, `beta=0.05`.

## AAO fairness

AAO, PSO, GA, GWO, and Bayesian optimization use the same configured hyperparameter bounds. Population-based comparisons should use comparable objective-evaluation budgets.

## Complexity

The code reports both theoretical scaling and empirical measurements.

Theoretical components:

- CLAHE: O(N)
- SLIC: approximately O(N) per local assignment iteration
- K-means: O(N K I)
- VGG16: convolution-dominated
- GOA: O(P G D)
- MHA: O(L^2 d)
- RNN: O(L d h + L h^2)
- AAO: O(P_A I_A C_obj)

Empirical measurements:

- trainable parameters;
- latency;
- peak CUDA memory when available;
- AAO/GOA search wall time;
- convergence history.

## External validation

External evaluation must use the frozen model and frozen threshold. Any new tuning constitutes adaptation and must be reported separately.

## Hardware metadata

Each experiment writes:

- platform;
- Python version;
- PyTorch version;
- CUDA version;
- GPU name when available;
- configuration SHA-256;
- Git commit if available.

## Expected deviation policy

GPU kernels, library versions, and nondeterministic low-level operations may produce small numerical differences. The repository therefore reports repeated runs and confidence intervals instead of requiring byte-identical floating-point outputs.
