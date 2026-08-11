from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_pipeline import build_manifest, summarize_manifest
from src.utils import ensure_dir, load_config, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mapping", default="class_mapping.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    manifest = build_manifest(cfg, args.mapping)

    out = Path(cfg["data"]["manifest_csv"])
    ensure_dir(out.parent)
    manifest.to_csv(out, index=False)
    save_json(summarize_manifest(manifest), out.with_suffix(".summary.json"))

    print(f"Saved manifest: {out}")
    print(manifest.groupby(["split", "binary_class"]).size().unstack(fill_value=0))
    print("\nOriginal-class distribution:")
    print(manifest.groupby(["split", "original_class"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
