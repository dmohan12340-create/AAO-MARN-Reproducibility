from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from skimage.segmentation import slic


def apply_clahe_rgb(
    rgb: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(map(int, tile_grid)))
    l2 = clahe.apply(l)
    out = cv2.merge([l2, a, b])
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def slic_labels(rgb: np.ndarray, n_segments: int, compactness: float, sigma: float) -> np.ndarray:
    labels = slic(
        rgb,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=0,
        channel_axis=-1,
        convert2lab=True,
    )
    return labels.astype(np.int32)


def kmeans_region_mask(
    rgb: np.ndarray,
    segments: np.ndarray,
    n_clusters: int = 3,
    attempts: int = 3,
    mode: str = "adaptive",
) -> np.ndarray:
    """
    K-means is applied to superpixel mean Lab+HSV descriptors rather than to
    independent pixels. This preserves SLIC spatial coherence and reduces noise.

    The default adaptive lesion-cluster heuristic favors clusters with stronger
    chromatic deviation and local contrast. It is intentionally unsupervised.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    ids = np.unique(segments)
    feats = []
    for sid in ids:
        m = segments == sid
        if not np.any(m):
            continue
        vals_lab = lab[m].astype(np.float32)
        vals_hsv = hsv[m].astype(np.float32)
        vals_gray = gray[m].astype(np.float32)
        feats.append(
            [
                vals_lab[:, 0].mean(),
                vals_lab[:, 1].mean(),
                vals_lab[:, 2].mean(),
                vals_hsv[:, 1].mean(),
                vals_hsv[:, 0].mean(),
                vals_gray.std(),
            ]
        )
    feats = np.asarray(feats, np.float32)
    if len(feats) < n_clusters:
        return np.ones(rgb.shape[:2], dtype=np.uint8)

    # Standardize descriptors before k-means.
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True) + 1e-6
    z = (feats - mu) / sd

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    _, labels, centers = cv2.kmeans(
        z,
        int(n_clusters),
        None,
        criteria,
        int(attempts),
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.ravel()

    # Heuristic lesion score: saturation/chromatic deviation + texture.
    # z columns: L,a,b,S,H,gray_std
    scores = (
        0.65 * np.abs(centers[:, 1])
        + 0.65 * np.abs(centers[:, 2])
        + 0.35 * centers[:, 3]
        + 0.40 * centers[:, 5]
    )
    lesion_cluster = int(np.argmax(scores))

    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for sid, c in zip(ids, labels):
        if int(c) == lesion_cluster:
            mask[segments == sid] = 1

    # Avoid pathological all-black or all-white masks.
    fraction = mask.mean()
    if fraction < 0.03 or fraction > 0.90:
        return np.ones_like(mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def apply_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.max() == 0:
        return rgb
    bg = np.full_like(rgb, int(np.median(rgb)))
    return np.where(mask[..., None].astype(bool), rgb, bg)


def _augment(rgb: np.ndarray) -> np.ndarray:
    if np.random.rand() < 0.5:
        rgb = np.ascontiguousarray(rgb[:, ::-1])
    if np.random.rand() < 0.25:
        rgb = np.ascontiguousarray(rgb[::-1])
    k = np.random.randint(0, 4)
    if k:
        rgb = np.ascontiguousarray(np.rot90(rgb, k))
    return rgb


def preprocess_leaf(
    path: str | Path,
    cfg: Dict,
    image_size: int = 224,
    augment: bool = False,
) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    rgb = apply_clahe_rgb(
        rgb,
        cfg.get("clahe_clip_limit", 2.0),
        tuple(cfg.get("clahe_tile_grid", [8, 8])),
    )

    if bool(cfg.get("use_segmentation_mask", True)):
        seg = slic_labels(
            rgb,
            cfg.get("slic_segments", 120),
            cfg.get("slic_compactness", 10.0),
            cfg.get("slic_sigma", 1.0),
        )
        mask = kmeans_region_mask(
            rgb,
            seg,
            cfg.get("kmeans_clusters", 3),
            cfg.get("kmeans_attempts", 3),
            cfg.get("lesion_mask_mode", "adaptive"),
        )
        rgb = apply_mask(rgb, mask)

    if augment:
        rgb = _augment(rgb)

    rgb = cv2.resize(rgb, (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.float32) / 255.0
