from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.analysis import attention_to_grid, difficult_cases
from src.model import model_from_config
from src.preprocessing import preprocess_leaf
from src.utils import ensure_dir, load_config, resolve_device


def save_overlay(image_path, grid, out_path, image_size=224):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return
    bgr = cv2.resize(bgr, (image_size, image_size))
    heat = cv2.resize(grid.astype(np.float32), (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    heat_u8 = np.uint8(np.clip(heat, 0, 1) * 255)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr, 0.55, heat_color, 0.45, 0)
    cv2.imwrite(str(out_path), overlay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg["project"]["output_dir"])
    attn_dir = ensure_dir(out / "attention_overlays_runtime")

    pred = pd.read_csv(out / "test_predictions.csv")
    difficult_cases(pred, int(cfg["analysis"].get("error_samples", 40))).to_csv(
        out / "difficult_cases.csv", index=False
    )

    checkpoint = torch.load(out / "best_model.pt", map_location="cpu")
    model = model_from_config(checkpoint["config"], checkpoint["selected_channels"])
    model.load_state_dict(checkpoint["state_dict"])
    device = resolve_device(cfg["runtime"].get("device", "auto"))
    model = model.to(device).eval()

    sample = pred.sort_values(
        "prob_diseased",
        key=lambda s: np.abs(s - 0.5)
    ).head(int(cfg["analysis"].get("attention_samples", 24)))

    records = []
    for i, row in sample.reset_index(drop=True).iterrows():
        arr = preprocess_leaf(
            row["path"], cfg["preprocessing"], int(cfg["data"]["image_size"]), augment=False
        )
        x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits, attn = model(x, return_attention=True)
        grid = attention_to_grid(attn.detach().cpu().numpy(), (7, 7))
        out_path = attn_dir / f"attention_{i:03d}.png"
        save_overlay(row["path"], grid, out_path, int(cfg["data"]["image_size"]))
        records.append(
            {
                "path": row["path"],
                "binary_label": int(row["binary_label"]),
                "prob_diseased": float(row["prob_diseased"]),
                "overlay": str(out_path),
            }
        )
    pd.DataFrame(records).to_csv(out / "attention_overlay_index.csv", index=False)
    print(f"Saved {len(records)} attention overlays.")


if __name__ == "__main__":
    main()
