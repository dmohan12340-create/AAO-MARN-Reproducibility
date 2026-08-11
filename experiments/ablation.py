from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from experiments.primary_binary import train_once
from src.data_pipeline import load_manifest
from src.utils import ensure_dir, load_config, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = ensure_dir(cfg["project"]["output_dir"])
    manifest = load_manifest(cfg["data"]["manifest_csv"])

    selected = list(range(512))
    goa_path = out / "goa_selected_channels.json"
    if goa_path.exists():
        import json
        selected = json.loads(goa_path.read_text())["selected_channels"]

    variants = [
        ("VGG16_full", list(range(512)), {"attention_heads": 1, "attention_dim": 512, "recurrent_hidden": 64, "dropout": 0.3}),
        ("VGG16_GOA", selected, {"attention_heads": 1, "attention_dim": 256, "recurrent_hidden": 64, "dropout": 0.3}),
        ("VGG16_GOA_RNN", selected, {"attention_heads": 1, "attention_dim": 256, "recurrent_hidden": cfg["model"]["recurrent_hidden"], "dropout": cfg["model"]["dropout"]}),
        ("VGG16_GOA_RNN_MHA", selected, {"attention_heads": cfg["model"]["attention_heads"], "attention_dim": cfg["model"]["attention_dim"], "recurrent_hidden": cfg["model"]["recurrent_hidden"], "dropout": cfg["model"]["dropout"]}),
        ("AAO_MARN_complete", selected, {}),
    ]

    rows = []
    for seed in cfg["project"]["repeated_seeds"]:
        for name, channels, override in variants:
            _, _, _, test_df, threshold, _ = train_once(
                cfg, manifest, channels, int(seed), quick=args.quick, param_override=override
            )
            from src.evaluation import compute_metrics
            m = compute_metrics(test_df["binary_label"], test_df["prob_diseased"], threshold)
            rows.append({"seed": seed, "variant": name, **m})

    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation_runs.csv", index=False)
    summary = df.groupby("variant")[["accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc"]].agg(["mean", "std"])
    summary.to_csv(out / "ablation_summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
