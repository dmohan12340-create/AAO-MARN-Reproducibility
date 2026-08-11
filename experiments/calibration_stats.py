from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import (
    apply_temperature,
    compute_metrics,
    expected_calibration_error,
    temperature_scale_binary,
)
from src.utils import ensure_dir, load_config, save_json


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg["project"]["output_dir"])

    val = pd.read_csv(out / "validation_predictions.csv")
    test = pd.read_csv(out / "test_predictions.csv")

    temperature = temperature_scale_binary(
        logit(val["prob_diseased"]), val["binary_label"].to_numpy()
    )
    calibrated = apply_temperature(logit(test["prob_diseased"]), temperature)
    y = test["binary_label"].to_numpy()
    threshold = float(test["decision_threshold"].iloc[0])

    before = compute_metrics(y, test["prob_diseased"].to_numpy(), threshold)
    after = compute_metrics(y, calibrated, threshold)
    ece_before, rel_before = expected_calibration_error(
        y, test["prob_diseased"].to_numpy(), int(cfg["evaluation"].get("ece_bins", 15))
    )
    ece_after, rel_after = expected_calibration_error(
        y, calibrated, int(cfg["evaluation"].get("ece_bins", 15))
    )
    before["ece"] = ece_before
    after["ece"] = ece_after

    rel_before.to_csv(out / "reliability_before_temperature.csv", index=False)
    rel_after.to_csv(out / "reliability_after_temperature.csv", index=False)
    test.assign(prob_diseased_calibrated=calibrated).to_csv(
        out / "test_predictions_calibrated.csv", index=False
    )
    save_json(
        {"temperature": temperature, "before": before, "after": after},
        out / "calibration_metrics.json",
    )
    print(f"Temperature: {temperature:.4f}")
    print("Before ECE:", ece_before, "After ECE:", ece_after)


if __name__ == "__main__":
    main()
