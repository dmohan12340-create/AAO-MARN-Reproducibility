from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data_pipeline import PlantVillageDataset, load_class_mapping, scan_plantvillage
from src.evaluation import bootstrap_confidence_intervals, compute_metrics, expected_calibration_error
from src.model import model_from_config
from src.utils import ensure_dir, load_config, resolve_device, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--external-root", required=True)
    ap.add_argument("--mapping", default="class_mapping.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = ensure_dir(cfg["project"]["output_dir"])
    device = resolve_device(cfg["runtime"].get("device", "auto"))

    # This scanner intentionally requires an explicit class mapping. If external
    # class names differ, create a separate mapping CSV instead of guessing.
    ext = scan_plantvillage(
        args.external_root,
        args.mapping,
        cfg["data"].get("allowed_extensions", [".jpg", ".jpeg", ".png"]),
        compute_hash=True,
    )
    ext["split"] = "external"
    ds = PlantVillageDataset(
        ext, "external", cfg["preprocessing"], int(cfg["data"]["image_size"]), training=False
    )
    loader = DataLoader(ds, batch_size=int(cfg["data"]["batch_size"]), shuffle=False, num_workers=0)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = model_from_config(ckpt["config"], ckpt["selected_channels"])
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    threshold = float(ckpt["threshold"])

    rows = []
    with torch.no_grad():
        for b in loader:
            logits = model(b["image"].to(device))
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for i, p in enumerate(prob):
                rows.append(
                    {
                        "path": b["path"][i],
                        "original_class": b["original_class"][i],
                        "crop": b["crop"][i],
                        "binary_label": int(b["label"][i]),
                        "prob_diseased": float(p),
                        "predicted_label": int(p >= threshold),
                        "decision_threshold": threshold,
                    }
                )

    pred = pd.DataFrame(rows)
    pred.to_csv(out / "external_predictions.csv", index=False)
    y, p = pred["binary_label"].to_numpy(), pred["prob_diseased"].to_numpy()
    metrics = compute_metrics(y, p, threshold)
    ece, rel = expected_calibration_error(y, p, int(cfg["evaluation"].get("ece_bins", 15)))
    metrics["ece"] = ece
    ci = bootstrap_confidence_intervals(
        y, p, threshold,
        int(cfg["evaluation"].get("bootstrap_samples", 2000)),
        float(cfg["evaluation"].get("confidence_level", 0.95)),
        int(cfg["project"]["seed"]),
    )
    rel.to_csv(out / "external_reliability.csv", index=False)
    save_json(metrics, out / "external_metrics.json")
    save_json(ci, out / "external_metrics_ci.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
