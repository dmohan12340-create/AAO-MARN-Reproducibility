from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import compute_metrics
from src.utils import load_config, sha256_jsonable


REQUIRED_REPO_FILES = [
    "README.md",
    "CODE_AVAILABILITY.md",
    "REPRODUCIBILITY.md",
    "config.yaml",
    "class_mapping.csv",
    "run_all.py",
    "src/model.py",
    "src/optimizers.py",
    "experiments/primary_binary.py",
]


def fail(msg):
    raise AssertionError(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    for f in REQUIRED_REPO_FILES:
        if not Path(f).exists():
            fail(f"Missing required repository file: {f}")

    manifest_path = Path(cfg["data"]["manifest_csv"])
    if manifest_path.exists():
        m = pd.read_csv(manifest_path)
        if m["path"].duplicated().any():
            fail("Manifest contains duplicate paths.")
        if "sha256" in m and m["sha256"].astype(str).str.len().gt(0).any():
            leaks = m.groupby("sha256")["split"].nunique()
            if (leaks > 1).any():
                fail("Duplicate hashes cross split boundaries.")
        if set(m["split"].unique()) != {"train", "val", "test"}:
            fail("Manifest must contain train/val/test splits.")
        if set(m["binary_label"].unique()) - {0, 1}:
            fail("Primary target is not binary.")

    out = Path(cfg["project"]["output_dir"])
    pred_path = out / "test_predictions.csv"
    metrics_path = out / "primary_metrics.json"
    if pred_path.exists() and metrics_path.exists():
        p = pd.read_csv(pred_path)
        saved = json.loads(metrics_path.read_text())
        threshold = float(p["decision_threshold"].iloc[0])
        recalculated = compute_metrics(
            p["binary_label"].to_numpy(),
            p["prob_diseased"].to_numpy(),
            threshold,
        )
        for k in ["accuracy", "balanced_accuracy", "precision", "recall_sensitivity", "f1", "roc_auc", "pr_auc"]:
            if not np.isclose(saved[k], recalculated[k], atol=1e-10):
                fail(f"Saved metric {k} does not match prediction-file recomputation.")
        if int(saved["support"]) != len(p):
            fail("Metric support differs from test-prediction row count.")

    # Guard against accidentally hard-coding the manuscript's headline result in source.
    source_files = list(Path("src").glob("*.py")) + list(Path("experiments").glob("*.py"))
    forbidden = ["97.92", "0.9792", "0.9982"]
    hits = []
    for path in source_files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                hits.append((str(path), token))
    if hits:
        fail(f"Potential hard-coded manuscript metrics found in source: {hits}")

    print("Repository verification passed.")
    print("Config SHA-256:", sha256_jsonable(cfg))


if __name__ == "__main__":
    main()
