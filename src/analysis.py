from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler

from .utils import ensure_dir


def separability_embeddings(
    features: np.ndarray,
    labels: Iterable[str],
    seed: int = 17,
    tsne_perplexity: int = 30,
    umap_neighbors: int = 15,
) -> Dict[str, object]:
    labels = np.asarray(list(labels))
    x = StandardScaler().fit_transform(features)

    pca_dim = min(50, x.shape[1], max(2, x.shape[0] - 1))
    x_small = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)

    perplexity = min(tsne_perplexity, max(5, (len(x_small) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(x_small)

    try:
        import umap
        um = umap.UMAP(
            n_components=2,
            n_neighbors=min(umap_neighbors, max(2, len(x_small) - 1)),
            random_state=seed,
        ).fit_transform(x_small)
    except Exception:
        um = np.full((len(x_small), 2), np.nan)

    sil = float(silhouette_score(x_small, labels)) if len(np.unique(labels)) > 1 else float("nan")
    db = float(davies_bouldin_score(x_small, labels)) if len(np.unique(labels)) > 1 else float("nan")

    return {"tsne": tsne, "umap": um, "silhouette": sil, "davies_bouldin": db}


def centroid_distance_matrix(features: np.ndarray, labels: Iterable[str]) -> pd.DataFrame:
    labels = np.asarray(list(labels))
    classes = sorted(np.unique(labels))
    centroids = np.vstack([features[labels == c].mean(axis=0) for c in classes])
    d = pairwise_distances(centroids, metric="euclidean")
    return pd.DataFrame(d, index=classes, columns=classes)


def subgroup_metrics(pred_df: pd.DataFrame, group_col: str, metric_fn) -> pd.DataFrame:
    rows = []
    for group, g in pred_df.groupby(group_col):
        if g["binary_label"].nunique() < 2:
            # ROC-AUC is undefined but confusion metrics remain valid.
            y, p = g["binary_label"].to_numpy(), g["prob_diseased"].to_numpy()
            pred = (p >= g["decision_threshold"].iloc[0]).astype(int)
            rows.append(
                {
                    group_col: group,
                    "n": len(g),
                    "accuracy": float((pred == y).mean()),
                    "roc_auc": np.nan,
                }
            )
        else:
            m = metric_fn(
                g["binary_label"].to_numpy(),
                g["prob_diseased"].to_numpy(),
                float(g["decision_threshold"].iloc[0]),
            )
            rows.append({group_col: group, "n": len(g), **m})
    return pd.DataFrame(rows)


def attention_to_grid(attn: np.ndarray, grid_size: Tuple[int, int] = (7, 7)) -> np.ndarray:
    """
    Convert multi-head attention BxHxLxL or HxLxL to a token importance map.
    Importance is mean received attention across queries and heads.
    """
    a = np.asarray(attn)
    if a.ndim == 4:
        a = a[0]
    if a.ndim != 3:
        raise ValueError(f"Expected HxLxL attention, got shape {a.shape}")
    importance = a.mean(axis=0).mean(axis=0)
    if importance.size != grid_size[0] * grid_size[1]:
        raise ValueError("Attention token count does not match requested grid size.")
    grid = importance.reshape(grid_size)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    return grid


def difficult_cases(pred_df: pd.DataFrame, n: int = 40) -> pd.DataFrame:
    df = pred_df.copy()
    df["error"] = (df["predicted_label"] != df["binary_label"]).astype(int)
    df["uncertainty"] = np.abs(df["prob_diseased"] - 0.5)
    errors = df[df["error"] == 1].sort_values("uncertainty")
    correct_hard = df[df["error"] == 0].sort_values("uncertainty")
    return pd.concat([errors.head(n), correct_hard.head(max(0, n - len(errors.head(n))))])
