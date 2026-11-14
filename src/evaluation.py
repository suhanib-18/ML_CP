from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score


def f_beta_curve(y, scores, beta: float) -> tuple[np.ndarray, np.ndarray]:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    precision, recall = precision[:-1], recall[:-1]
    weight = beta ** 2
    score = (1 + weight) * precision * recall / np.maximum(weight * precision + recall, 1e-12)
    return thresholds, score


def best_threshold(y, scores, beta: float = 1.0) -> float:
    thresholds, score = f_beta_curve(y, scores, beta)
    return float(thresholds[int(np.argmax(score))])


def at_threshold(y, scores, threshold: float) -> dict:
    predicted = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def summarize(y, scores) -> dict:
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "base_rate": float(np.mean(y)),
    }


def pr_curve(y, scores, points: int = 300) -> dict:
    precision, recall, _ = precision_recall_curve(y, scores)
    order = np.argsort(recall, kind="stable")
    keep = np.unique(np.linspace(0, len(order) - 1, min(points, len(order))).astype(int))
    return {"recall": recall[order][keep].tolist(), "precision": precision[order][keep].tolist()}
