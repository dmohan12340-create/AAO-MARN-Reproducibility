from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.baselines import build_baseline
from src.data_pipeline import PlantVillageDataset, load_manifest
from src.evaluation import compute_metrics, optimal_threshold
from src.model import normalize_imagenet
from src.utils import ensure_dir, load_config, resolve_device, seed_everything


def make_loader(cfg, manifest, split, training):
    ds = PlantVillageDataset(
        manifest, split, cfg["preprocessing"], int(cfg["data"]["image_size"]), training=training
    )
    return DataLoader(ds, batch_size=int(cfg["data"]["batch_size"]), shuffle=training, num_workers=0)


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            x = normalize_imagenet(b["image"].to(device))
            logits = model(x)
            ps.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            ys.extend(b["label"].numpy())
    return np.asarray(ys), np.asarray(ps)


def train_model(name, cfg, manifest, seed, quick):
    seed_everything(seed)
    device = resolve_device(cfg["runtime"].get("device", "auto"))
    model = build_baseline(name, pretrained=True).to(device)
    tr = make_loader(cfg, manifest, "train", True)
    va = make_loader(cfg, manifest, "val", False)
    te = make_loader(cfg, manifest, "test", False)
    epochs = int(cfg["training"]["quick_epochs"] if quick else cfg["training"]["epochs"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    loss_fn = nn.CrossEntropyLoss()

    best, best_state = -1, None
    for _ in range(epochs):
        model.train()
        for b in tqdm(tr, desc=f"{name} train", leave=False):
            x = normalize_imagenet(b["image"].to(device))
            y = b["label"].to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        yv, pv = evaluate(model, va, device)
        t = optimal_threshold(yv, pv, cfg["training"].get("threshold_metric", "youden"))
        score = compute_metrics(yv, pv, t)["balanced_accuracy"]
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    yv, pv = evaluate(model, va, device)
    t = optimal_threshold(yv, pv, cfg["training"].get("threshold_metric", "youden"))
    yt, pt = evaluate(model, te, device)
    return compute_metrics(yt, pt, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    manifest = load_manifest(cfg["data"]["manifest_csv"])
    out = ensure_dir(cfg["project"]["output_dir"])

    models = ["mobilenet_v2", "efficientnet_b0", "convnext_tiny", "vit_b_16", "swin_t"]
    rows = []
    for seed in cfg["project"]["repeated_seeds"]:
        for name in models:
            try:
                m = train_model(name, cfg, manifest, int(seed), args.quick)
                rows.append({"seed": seed, "model": name, **m})
            except RuntimeError as e:
                rows.append({"seed": seed, "model": name, "error": str(e)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "modern_baseline_runs.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()
