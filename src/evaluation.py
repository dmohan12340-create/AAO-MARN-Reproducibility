from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
)


def optimal_threshold(y_true: np.ndarray, prob: np.ndarray, method: str = "youden") -> float:
    if method == "youden":
        fpr, tpr, thr = roc_curve(y_true, prob)
        idx = int(np.nanargmax(tpr - fpr))
        t = float(thr[idx])
        return float(np.clip(t, 0.0, 1.0))
    if method == "f1":
        candidates = np.linspace(0.01, 0.99, 99)
        vals = [f1_score(y_true, prob >= t, zero_division=0) for t in candidates]
        return float(candidates[int(np.argmax(vals))])
    return 0.5


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = tn / max(tn + fp, 1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "nll": float(log_loss(y_true, np.c_[1 - prob, prob], labels=[0, 1])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "support": int(len(y_true)),
        "threshold": float(threshold),
    }


def expected_calibration_error(
    y_true: np.ndarray, prob: np.ndarray, n_bins: int = 15
) -> Tuple[float, pd.DataFrame]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    confidence = np.maximum(prob, 1 - prob)
    pred = (prob >= 0.5).astype(int)
    correct = (pred == y_true).astype(float)

    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (confidence > lo) & (confidence <= hi) if i else (confidence >= lo) & (confidence <= hi)
        if not np.any(m):
            continue
        conf = float(confidence[m].mean())
        acc = float(correct[m].mean())
        frac = float(m.mean())
        ece += frac * abs(acc - conf)
        rows.append(
            {"bin": i, "lower": lo, "upper": hi, "n": int(m.sum()), "confidence": conf, "accuracy": acc}
        )
    return float(ece), pd.DataFrame(rows)


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 17,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    n = len(y_true)
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier",
    ]
    draws = {m: [] for m in metric_names}

    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    for _ in range(samples):
        idx = np.concatenate(
            [rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        )
        m = compute_metrics(y_true[idx], prob[idx], threshold)
        for name in metric_names:
            draws[name].append(m[name])

    alpha = 1.0 - confidence
    out = {}
    point = compute_metrics(y_true, prob, threshold)
    for name in metric_names:
        vals = np.asarray(draws[name])
        out[name] = {
            "estimate": float(point[name]),
            "ci_lower": float(np.quantile(vals, alpha / 2)),
            "ci_upper": float(np.quantile(vals, 1 - alpha / 2)),
        }
    return out


def mcnemar_exact(y_true, pred_a, pred_b) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    n = b + c
    p = 1.0 if n == 0 else float(binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
    stat = 0.0 if n == 0 else float((abs(b - c) - 1) ** 2 / n)
    return {"discordant_a_only": b, "discordant_b_only": c, "chi2_cc": stat, "p_value_exact": p}


def wilcoxon_paired(a: Iterable[float], b: Iterable[float]) -> Dict[str, float]:
    a, b = np.asarray(list(a), float), np.asarray(list(b), float)
    if len(a) != len(b):
        raise ValueError("Paired vectors must have equal length.")
    if np.allclose(a, b):
        return {"statistic": 0.0, "p_value": 1.0}
    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(stat), "p_value": float(p)}


def temperature_scale_binary(logits: np.ndarray, y_true: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    def nll(log_temp):
        t = np.exp(log_temp)
        z = logits / t
        p = 1.0 / (1.0 + np.exp(-z))
        return log_loss(y_true, np.c_[1 - p, p], labels=[0, 1])

    res = minimize_scalar(nll, bounds=(-4, 4), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(logits, float) / float(temperature)
    return 1.0 / (1.0 + np.exp(-z))
