from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_pipeline import PlantVillageDataset, load_manifest
from src.evaluation import (
    bootstrap_confidence_intervals,
    compute_metrics,
    expected_calibration_error,
    optimal_threshold,
)
from src.model import VGG16SpatialBackbone, model_from_config
from src.optimizers import artificial_algae_search, goa_feature_selection
from src.utils import (
    ensure_dir,
    hardware_metadata,
    load_config,
    save_json,
    seed_everything,
    resolve_device,
)


def make_loaders(cfg, manifest, seed):
    seed_everything(seed)
    bs = int(cfg["data"]["batch_size"])
    nw = int(cfg["data"].get("num_workers", 4))
    size = int(cfg["data"]["image_size"])
    pcfg = cfg["preprocessing"]
    g = torch.Generator().manual_seed(seed)

    def loader(split, training):
        ds = PlantVillageDataset(manifest, split, pcfg, size, training=training)
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=training,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=nw > 0,
            generator=g if training else None,
        )

    return loader("train", True), loader("val", False), loader("test", False)


@torch.no_grad()
def extract_pooled_vgg_features(loader, device, pretrained=True):
    backbone = VGG16SpatialBackbone(pretrained=pretrained).to(device).eval()
    feats, ys = [], []
    for batch in tqdm(loader, desc="Extract VGG features", leave=False):
        x = batch["image"].to(device)
        f = backbone(x).mean(dim=(2, 3))
        feats.append(f.cpu().numpy())
        ys.append(batch["label"].numpy())
    return np.concatenate(feats), np.concatenate(ys)


def choose_goa_channels(cfg, train_loader, val_loader, device, quick=False):
    gcfg = cfg["goa"]
    if not gcfg.get("enabled", True):
        return list(range(512)), []

    xtr, ytr = extract_pooled_vgg_features(
        train_loader, device, cfg["model"].get("pretrained_vgg16", True)
    )
    xva, yva = extract_pooled_vgg_features(
        val_loader, device, cfg["model"].get("pretrained_vgg16", True)
    )
    pop = int(gcfg["quick_population"] if quick else gcfg["population"])
    gen = int(gcfg["quick_generations"] if quick else gcfg["generations"])
    selected, history = goa_feature_selection(
        xtr, ytr, xva, yva,
        population=pop,
        generations=gen,
        alpha=float(gcfg["alpha"]),
        beta=float(gcfg["beta"]),
        min_selected=int(gcfg["min_selected_channels"]),
        max_selected=int(gcfg["max_selected_channels"]),
        c_max=float(gcfg.get("c_max", 1.0)),
        c_min=float(gcfg.get("c_min", 0.00004)),
        seed=int(cfg["project"]["seed"]),
    )
    return selected.tolist(), history


def _optimizer(model, cfg):
    t = cfg["training"]
    lr, wd = float(t["learning_rate"]), float(t["weight_decay"])
    if t["optimizer"].lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def _scheduler(opt, cfg, epochs):
    if cfg["training"].get("scheduler", "cosine").lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    return None


def _binary_prob(logits):
    return torch.softmax(logits, dim=1)[:, 1]


@torch.no_grad()
def predict(model, loader, device, threshold=0.5):
    model.eval()
    rows = []
    for batch in tqdm(loader, desc="Predict", leave=False):
        x = batch["image"].to(device)
        logits = model(x)
        prob = _binary_prob(logits).cpu().numpy()
        y = batch["label"].numpy()
        for i in range(len(y)):
            rows.append(
                {
                    "path": batch["path"][i],
                    "original_class": batch["original_class"][i],
                    "crop": batch["crop"][i],
                    "binary_label": int(y[i]),
                    "prob_diseased": float(prob[i]),
                }
            )
    df = pd.DataFrame(rows)
    df["predicted_label"] = (df["prob_diseased"] >= threshold).astype(int)
    df["decision_threshold"] = float(threshold)
    return df


def train_once(cfg, manifest, selected_channels, seed, quick=False, param_override=None):
    seed_everything(seed)
    local = copy.deepcopy(cfg)
    if param_override:
        for k, v in param_override.items():
            if k in {"learning_rate", "weight_decay"}:
                local["training"][k] = v
            elif k in {"dropout", "attention_dim", "attention_heads", "recurrent_hidden"}:
                local["model"][k] = v

    train_loader, val_loader, test_loader = make_loaders(local, manifest, seed)
    device = resolve_device(local["runtime"].get("device", "auto"))
    model = model_from_config(local, selected_channels).to(device)
    freeze_epochs = int(local["model"].get("freeze_backbone_epochs", 0))
    model.set_backbone_trainable(freeze_epochs <= 0)

    epochs = int(local["training"]["quick_epochs"] if quick else local["training"]["epochs"])
    opt = _optimizer(model, local)
    sch = _scheduler(opt, local, epochs)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(local["training"].get("label_smoothing", 0.0))
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(local["training"].get("amp", True)) and device.type == "cuda"
    )

    best_state = None
    best_val = -np.inf
    patience = int(local["training"].get("early_stopping_patience", 5))
    wait = 0
    history = []

    for epoch in range(epochs):
        if epoch == freeze_epochs and freeze_epochs > 0:
            model.set_backbone_trainable(True)

        model.train()
        losses, y_train, p_train = [], [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=bool(local["training"].get("amp", True)) and device.type == "cuda",
            ):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(local["training"].get("gradient_clip_norm", 2.0))
            )
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
            y_train.extend(y.detach().cpu().numpy().tolist())
            p_train.extend(_binary_prob(logits.detach()).cpu().numpy().tolist())

        val_df = predict(model, val_loader, device, threshold=0.5)
        val_thr = optimal_threshold(
            val_df["binary_label"].to_numpy(),
            val_df["prob_diseased"].to_numpy(),
            local["training"].get("threshold_metric", "youden"),
        )
        vm = compute_metrics(
            val_df["binary_label"].to_numpy(),
            val_df["prob_diseased"].to_numpy(),
            val_thr,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_balanced_accuracy": vm["balanced_accuracy"],
                "val_roc_auc": vm["roc_auc"],
                "val_pr_auc": vm["pr_auc"],
                "val_threshold": val_thr,
            }
        )

        score = vm["balanced_accuracy"]
        if score > best_val + 1e-6:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if sch is not None:
            sch.step()
        if wait >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint.")
    model.load_state_dict(best_state)

    val_df = predict(model, val_loader, device, threshold=0.5)
    threshold = optimal_threshold(
        val_df["binary_label"].to_numpy(),
        val_df["prob_diseased"].to_numpy(),
        local["training"].get("threshold_metric", "youden"),
    )
    test_df = predict(model, test_loader, device, threshold=threshold)
    return model, history, val_df, test_df, threshold, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-aao", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    outdir = ensure_dir(cfg["project"]["output_dir"])
    manifest = load_manifest(cfg["data"]["manifest_csv"])
    seed = int(cfg["project"]["seed"])
    device = resolve_device(cfg["runtime"].get("device", "auto"))

    # GOA is fitted using training and validation data only.
    train_loader, val_loader, _ = make_loaders(cfg, manifest, seed)
    selected_channels, goa_history = choose_goa_channels(
        cfg, train_loader, val_loader, device, quick=args.quick
    )
    save_json(
        {"selected_channels": selected_channels, "n": len(selected_channels)},
        outdir / "goa_selected_channels.json",
    )
    pd.DataFrame(goa_history).to_csv(outdir / "goa_convergence.csv", index=False)

    # AAO uses validation performance only. The objective never evaluates the test split.
    chosen_params = {}
    aao_result = None
    if cfg["aao"].get("enabled", True) and not args.skip_aao:
        acfg = cfg["aao"]
        pop = int(acfg["quick_population"] if args.quick else acfg["population"])
        iters = int(acfg["quick_iterations"] if args.quick else acfg["iterations"])

        def objective(params):
            _, _, _, _, _, best_val = train_once(
                cfg, manifest, selected_channels, seed, quick=True, param_override=params
            )
            return best_val

        aao_result = artificial_algae_search(
            objective,
            acfg["search_space"],
            population=pop,
            iterations=iters,
            patience=int(acfg.get("patience", 5)),
            min_delta=float(acfg.get("min_delta", 0.0005)),
            adaptation_probability=float(acfg.get("adaptation_probability", 0.3)),
            seed=seed,
        )
        chosen_params = aao_result.best_params
        save_json(
            {
                "best_params": aao_result.best_params,
                "best_score": aao_result.best_score,
                "runtime_seconds": aao_result.runtime_seconds,
                "history": aao_result.history,
            },
            outdir / "aao_search.json",
        )

    model, history, val_df, test_df, threshold, _ = train_once(
        cfg, manifest, selected_channels, seed, quick=args.quick, param_override=chosen_params
    )

    # Save predictions before aggregate results.
    test_df["seed"] = seed
    test_df["model_variant"] = "AAO-MARN"
    pred_path = outdir / "test_predictions.csv"
    test_df.to_csv(pred_path, index=False)
    val_df.to_csv(outdir / "validation_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)

    y = test_df["binary_label"].to_numpy()
    p = test_df["prob_diseased"].to_numpy()
    metrics = compute_metrics(y, p, threshold)
    ece, reliability = expected_calibration_error(
        y, p, int(cfg["evaluation"].get("ece_bins", 15))
    )
    metrics["ece"] = ece
    ci = bootstrap_confidence_intervals(
        y, p, threshold,
        samples=int(cfg["evaluation"].get("bootstrap_samples", 2000)),
        confidence=float(cfg["evaluation"].get("confidence_level", 0.95)),
        seed=seed,
    )

    reliability.to_csv(outdir / "reliability_uncalibrated.csv", index=False)
    save_json(metrics, outdir / "primary_metrics.json")
    save_json(ci, outdir / "primary_metrics_ci.json")
    save_json(hardware_metadata(cfg), outdir / "environment.json")

    checkpoint = {
        "state_dict": model.state_dict(),
        "selected_channels": selected_channels,
        "threshold": threshold,
        "config": cfg,
        "chosen_params": chosen_params,
    }
    torch.save(checkpoint, outdir / "best_model.pt")

    print(json.dumps(metrics, indent=2))
    print(f"Authoritative predictions: {pred_path}")


if __name__ == "__main__":
    main()
