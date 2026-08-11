from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import entropy as scipy_entropy
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .preprocessing import preprocess_leaf
from .utils import sha256_file


def load_class_mapping(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"original_class", "crop", "binary_class", "binary_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"class_mapping.csv missing columns: {sorted(missing)}")
    if set(df["binary_label"].unique()) - {0, 1}:
        raise ValueError("binary_label must contain only 0/1")
    return df


def _image_quality(path: Path) -> Dict[str, float]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unreadable image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1.0)

    return {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "brightness_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "saturation_mean": float(hsv[..., 1].mean()),
        "entropy": float(scipy_entropy(p + 1e-12, base=2)),
        "blur_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "r_mean": float(rgb[..., 0].mean()),
        "g_mean": float(rgb[..., 1].mean()),
        "b_mean": float(rgb[..., 2].mean()),
    }


def scan_plantvillage(
    root: str | Path,
    mapping_csv: str | Path,
    allowed_extensions: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp"),
    compute_hash: bool = True,
) -> pd.DataFrame:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    mapping = load_class_mapping(mapping_csv).set_index("original_class")
    extensions = {e.lower() for e in allowed_extensions}
    rows: List[Dict] = []

    for class_name in mapping.index:
        class_dir = root / class_name
        if not class_dir.exists():
            # Some PlantVillage mirrors rename underscores. Fail later if class absent.
            continue
        meta = mapping.loc[class_name]
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            quality = _image_quality(path)
            rows.append(
                {
                    "path": str(path.resolve()),
                    "original_class": class_name,
                    "crop": str(meta["crop"]),
                    "binary_class": str(meta["binary_class"]),
                    "binary_label": int(meta["binary_label"]),
                    "sha256": sha256_file(path) if compute_hash else "",
                    **quality,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "No mapped images were found. Verify data.plantvillage_root and folder names."
        )

    present = set(df["original_class"].unique())
    expected = set(mapping.index)
    missing = sorted(expected - present)
    if missing:
        print(f"WARNING: mapped classes absent from dataset: {missing}")

    return df


def stratified_split(
    df: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    if df["original_class"].value_counts().min() < 3:
        raise ValueError("Every original class needs at least 3 samples for a 3-way split.")

    train, temp = train_test_split(
        df,
        test_size=1.0 - train_fraction,
        random_state=seed,
        stratify=df["original_class"],
    )
    relative_test = test_fraction / (val_fraction + test_fraction)
    val, test = train_test_split(
        temp,
        test_size=relative_test,
        random_state=seed,
        stratify=temp["original_class"],
    )

    out = pd.concat(
        [
            train.assign(split="train"),
            val.assign(split="val"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )

    # Check exact membership disjointness.
    if out["path"].duplicated().any():
        raise RuntimeError("Duplicate path assigned to more than one split.")

    hashes = out[out["sha256"].astype(str).str.len() > 0]
    if not hashes.empty:
        collision = (
            hashes.groupby("sha256")["split"]
            .nunique()
            .reset_index(name="n_splits")
            .query("n_splits > 1")
        )
        if not collision.empty:
            raise RuntimeError(
                f"Data leakage: {len(collision)} duplicate file hashes cross split boundaries."
            )

    return out.sort_values(["split", "original_class", "path"]).reset_index(drop=True)


def build_manifest(config: Dict, mapping_csv: str | Path = "class_mapping.csv") -> pd.DataFrame:
    dcfg = config["data"]
    raw = scan_plantvillage(
        dcfg["plantvillage_root"],
        mapping_csv,
        allowed_extensions=dcfg.get("allowed_extensions", [".jpg", ".jpeg", ".png"]),
        compute_hash=bool(dcfg.get("verify_hashes", True)),
    )
    return stratified_split(
        raw,
        dcfg["train_fraction"],
        dcfg["val_fraction"],
        dcfg["test_fraction"],
        int(config["project"]["seed"]),
    )


def summarize_manifest(df: pd.DataFrame) -> Dict:
    return {
        "n": int(len(df)),
        "by_split": df["split"].value_counts().to_dict(),
        "by_binary": df["binary_class"].value_counts().to_dict(),
        "by_original_class": df["original_class"].value_counts().sort_index().to_dict(),
        "by_crop": df["crop"].value_counts().to_dict(),
    }


class PlantVillageDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        preprocessing_cfg: Dict,
        image_size: int = 224,
        training: bool = False,
        return_path: bool = True,
    ):
        self.df = manifest[manifest["split"] == split].reset_index(drop=True).copy()
        self.preprocessing_cfg = preprocessing_cfg
        self.image_size = image_size
        self.training = training
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        arr = preprocess_leaf(
            row["path"],
            cfg=self.preprocessing_cfg,
            image_size=self.image_size,
            augment=self.training,
        )
        # HWC float32 [0,1] -> CHW
        import torch

        x = torch.from_numpy(arr.transpose(2, 0, 1)).float()
        y = torch.tensor(int(row["binary_label"]), dtype=torch.long)
        item = {
            "image": x,
            "label": y,
            "original_class": row["original_class"],
            "crop": row["crop"],
        }
        if self.return_path:
            item["path"] = row["path"]
        return item


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "path",
        "original_class",
        "crop",
        "binary_class",
        "binary_label",
        "split",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return df
