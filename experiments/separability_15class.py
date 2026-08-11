from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.analysis import centroid_distance_matrix, separability_embeddings
from src.data_pipeline import PlantVillageDataset, load_manifest
from src.model import VGG16SpatialBackbone
from src.utils import ensure_dir, load_config, resolve_device, save_json, seed_everything


@torch.no_grad()
def extract(loader, device, pretrained=True):
    m = VGG16SpatialBackbone(pretrained).to(device).eval()
    xs, labels, crops, paths = [], [], [], []
    for b in tqdm(loader, desc="Extract features"):
        f = m(b["image"].to(device)).mean(dim=(2, 3))
        xs.append(f.cpu().numpy())
        labels += list(b["original_class"])
        crops += list(b["crop"])
        paths += list(b["path"])
    return np.concatenate(xs), labels, crops, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["project"]["seed"])
    seed_everything(seed)
    out = ensure_dir(cfg["project"]["output_dir"])

    manifest = load_manifest(cfg["data"]["manifest_csv"])
    # Separability is descriptive and may use all images; it does not fit the predictive test pipeline.
    max_n = int(cfg["analysis"].get("separability_max_samples", 5000))
    if len(manifest) > max_n:
        manifest = (
            manifest.groupby("original_class", group_keys=False)
            .apply(lambda g: g.sample(max(1, int(max_n * len(g) / len(manifest))), random_state=seed))
            .reset_index(drop=True)
        )

    manifest = manifest.assign(split="analysis")
    ds = PlantVillageDataset(
        manifest, "analysis", cfg["preprocessing"], int(cfg["data"]["image_size"]), training=False
    )
    loader = DataLoader(ds, batch_size=int(cfg["data"]["batch_size"]), shuffle=False, num_workers=0)
    device = resolve_device(cfg["runtime"].get("device", "auto"))
    x, labels, crops, paths = extract(loader, device, cfg["model"].get("pretrained_vgg16", True))

    emb = separability_embeddings(
        x, labels, seed,
        int(cfg["analysis"].get("tsne_perplexity", 30)),
        int(cfg["analysis"].get("umap_neighbors", 15)),
    )
    coords = pd.DataFrame(
        {
            "path": paths,
            "original_class": labels,
            "crop": crops,
            "tsne_x": emb["tsne"][:, 0],
            "tsne_y": emb["tsne"][:, 1],
            "umap_x": emb["umap"][:, 0],
            "umap_y": emb["umap"][:, 1],
        }
    )
    coords.to_csv(out / "feature_space_coordinates.csv", index=False)
    centroid_distance_matrix(x, labels).to_csv(out / "class_centroid_distances.csv")
    save_json(
        {"silhouette": emb["silhouette"], "davies_bouldin": emb["davies_bouldin"]},
        out / "separability_metrics.json",
    )
    print("Saved feature-space coordinates and separability metrics.")


if __name__ == "__main__":
    main()
