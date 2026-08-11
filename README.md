# AAO-MARN Reproducibility

Reference implementation for the experimental pipeline described in **“Multi-Attentive Deep Learning and Bio-Inspired Optimization for Enhanced Plant Disease Identification.”**

The repository is designed to make the experimental protocol auditable and to address the principal reproducibility concerns raised during peer review:

- one explicit **binary primary endpoint**: Healthy vs Diseased;
- preservation of the original 15 PlantVillage labels for subgroup, separability, and error analysis;
- fixed train/validation/test manifests with leakage checks and reproducible seeds;
- CLAHE, SLIC, K-means, VGG16, GOA feature selection, Multi-Head Attention, recurrent spatial-context modelling, and AAO hyperparameter search;
- component-wise ablation;
- AAO comparison against Adam, PSO, GA, GWO, and Bayesian optimization;
- calibration analysis (ECE, Brier score, NLL, reliability curves);
- repeated-run statistics, bootstrap confidence intervals, McNemar and Wilcoxon tests;
- t-SNE/UMAP feature-space analysis and class-overlap statistics;
- attention-map and difficult-case analysis;
- complexity, runtime, parameter-count, and convergence reporting;
- external-dataset evaluation using a frozen trained model.

No result in this repository is hard-coded to reproduce a manuscript number. Final reported values must come from a stored prediction file generated on the locked test split.

## Repository structure

```text
AAO-MARN-Reproducibility/
├── README.md
├── CODE_AVAILABILITY.md
├── REPRODUCIBILITY.md
├── CITATION.cff
├── requirements.txt
├── pyproject.toml
├── config.yaml
├── class_mapping.csv
├── run_all.py
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── optimizers.py
│   ├── baselines.py
│   ├── evaluation.py
│   ├── analysis.py
│   └── utils.py
└── experiments/
    ├── prepare_data.py
    ├── primary_binary.py
    ├── separability_15class.py
    ├── ablation.py
    ├── optimizer_benchmark.py
    ├── modern_baselines.py
    ├── calibration_stats.py
    ├── attention_errors.py
    ├── complexity_runtime.py
    ├── external_validation.py
    └── verify_all.py
```

Only two source folders are committed. Runtime outputs are written to the directory configured by `output_dir` and are intentionally excluded from version control.

## 1. Dataset preparation

The expected PlantVillage layout is one folder per original class:

```text
PlantVillage/
├── Pepper__bell___Bacterial_spot/
├── Pepper__bell___healthy/
├── Potato___Early_blight/
...
└── Tomato_healthy/
```

Set the local dataset path in `config.yaml`:

```yaml
data:
  plantvillage_root: "/path/to/PlantVillage"
```

Then run:

```bash
python experiments/prepare_data.py --config config.yaml
```

The script creates a deterministic manifest with:

- `original_class`
- `binary_label`
- `crop`
- SHA-256 file hash
- width and height
- brightness/contrast/saturation/entropy/blur statistics
- split assignment

The split is stratified at the **original-class level** before binary aggregation, so the binary task does not destroy class-coverage balance. Duplicate hashes crossing splits cause an error.

## 2. Primary endpoint

The primary task is binary disease screening:

```text
Healthy  -> 0
Diseased -> 1
```

The original 15 labels are retained only for secondary analysis.

Run:

```bash
python experiments/primary_binary.py --config config.yaml
```

The script trains the proposed model, freezes the chosen threshold from the validation set, and evaluates the locked test split once. It writes:

- predictions;
- confusion matrix;
- ROC and PR data;
- point estimates and 95% bootstrap CIs;
- calibration metrics;
- training history;
- selected GOA feature channels;
- AAO search history;
- model checkpoint metadata.

## 3. Spatial-sequence formulation

The model does **not** interpret a static leaf photograph as a temporal signal.

VGG16 produces a `7 x 7 x 512` spatial feature map. The 49 spatial cells are ordered in raster order, producing a 49-token spatial sequence. The recurrent layer therefore models contextual dependencies among neighboring leaf regions.

```text
image
  -> CLAHE
  -> SLIC
  -> K-means lesion refinement
  -> VGG16 feature map (7 x 7 x 512)
  -> 49 spatial tokens
  -> GOA-selected feature channels
  -> linear embedding
  -> Multi-Head Self-Attention
  -> recurrent spatial-context layer
  -> binary classifier
```

## 4. GOA objective

GOA selects VGG feature channels using a validation-only fitness:

```text
fitness = alpha * validation_balanced_accuracy
        + beta  * feature_reduction_ratio
```

where `alpha + beta = 1`.

To prevent optimistic bias, the held-out test split is never used by GOA.

## 5. AAO search

AAO searches the hyperparameter space configured in `config.yaml`. Each algae colony is a concrete model configuration. The implementation records:

- candidate;
- validation objective;
- iteration;
- starvation state;
- best-so-far candidate;
- convergence reason;
- search runtime.

AAO may tune learning rate, weight decay, dropout, attention dimension, recurrent hidden dimension, and number of heads.

## 6. Reviewer-requested experiments

```bash
python experiments/separability_15class.py --config config.yaml
python experiments/ablation.py --config config.yaml
python experiments/optimizer_benchmark.py --config config.yaml
python experiments/modern_baselines.py --config config.yaml
python experiments/calibration_stats.py --config config.yaml
python experiments/attention_errors.py --config config.yaml
python experiments/complexity_runtime.py --config config.yaml
```

### Separability analysis

Produces:

- t-SNE and UMAP coordinates;
- silhouette score;
- Davies-Bouldin score;
- original-class centroid distance matrix;
- nearest-centroid confusion;
- crop-conditioned binary performance.

### Ablation variants

1. VGG16 classifier
2. VGG16 + GOA
3. VGG16 + GOA + recurrent context
4. VGG16 + GOA + recurrent context + MHA
5. complete AAO-MARN

Every variant uses the same split and seed schedule.

### Optimizer comparison

AAO is compared under an explicit search budget with:

- Adam/AdamW baseline training;
- PSO;
- Genetic Algorithm;
- Grey Wolf Optimizer;
- Bayesian optimization.

Population methods receive comparable objective-call budgets where possible.

### Calibration

The validation split is used for temperature fitting and threshold selection. The test set remains untouched until final evaluation.

Reported quantities:

- ECE;
- Brier score;
- NLL;
- calibrated and uncalibrated reliability data;
- confidence distributions.

## 7. External validation

Configure a second manifest/root in `config.yaml` and run:

```bash
python experiments/external_validation.py --config config.yaml \
  --checkpoint /path/to/best_model.pt \
  --external-root /path/to/external_dataset
```

The external dataset is evaluated with:

- frozen model weights;
- frozen preprocessing;
- frozen decision threshold;
- no tuning against external labels.

Classes that cannot be mapped unambiguously to Healthy/Diseased are rejected.

## 8. Full reproduction

After setting the dataset path:

```bash
python run_all.py --config config.yaml
```

For a reviewer-oriented smoke test:

```bash
python run_all.py --config config.yaml --quick
```

`--quick` reduces epochs and search budgets. It is for installation verification only and must not be used to report manuscript results.

## 9. Verification

```bash
python experiments/verify_all.py --config config.yaml
```

The verifier checks:

- required files;
- manifest integrity;
- split disjointness by path and SHA-256;
- result-to-prediction consistency;
- test-support consistency;
- metric recomputation;
- seed metadata;
- configuration hash;
- absence of hard-coded manuscript result claims.

## Environment

Recommended:

- Python 3.10 or 3.11
- PyTorch 2.4+
- CUDA-capable GPU recommended for full experiments

Install:

```bash
pip install -r requirements.txt
```

## Reproducibility policy

All manuscript tables and figures should be regenerated from machine-readable experiment artifacts. If a rerun produces values that differ from the current manuscript, the manuscript must be updated to the reproduced values rather than modifying the code to force a predetermined number.

## Code archival

The manuscript is under review, so no software DOI is claimed in this repository. After the GitHub release is finalized, archive the exact tagged release in a DOI-assigning repository such as Zenodo and replace the placeholder in `CODE_AVAILABILITY.md` and `CITATION.cff` with the assigned software DOI.
