"""Shared helpers: dataset splits, image cache, metrics and result files.

Label convention: def_front = 1 (positive class, the defect we must not miss), ok_front = 0.

Two splits:
  official  The Kaggle 300x300 set (train/test folders). Validation is carved from train.
            Those 300px images are augmented copies of the 1300 originals, so near-identical
            variants of one part can sit in both train and test (leakage).
  clean     Only the 1300 512x512 originals, split stratified into train/val/test.
            No augmented copies, so no part appears in two splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_DIR = ROOT / "casting_data" / "casting_data"
ORIGINAL_DIR = ROOT / "casting_512x512" / "casting_512x512"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
CLASSES = {"ok_front": 0, "def_front": 1}
SEED = 42
TARGET_RECALL = 0.99


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(os.cpu_count() or 1)
    except ImportError:
        pass


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--split", choices=["official", "clean"], default="official")
    p.add_argument("--smoke", action="store_true", help="tiny run (~50 images per class) to check the script works")
    return p


def _list_dir(folder: Path) -> list[tuple[Path, int]]:
    items = []
    for cls, label in CLASSES.items():
        for f in sorted((folder / cls).glob("*.jpeg")):
            items.append((f, label))
    return items


def _subsample(items, n_per_class, seed=SEED):
    rng = random.Random(seed)
    out = []
    for label in (0, 1):
        cls_items = [it for it in items if it[1] == label]
        rng.shuffle(cls_items)
        out += cls_items[:n_per_class]
    return out


def get_split(split: str, smoke: bool = False) -> dict[str, list[tuple[Path, int]]]:
    """Return {'train': [...], 'val': [...], 'test': [...]} with (path, label) pairs."""
    if split == "official":
        train_all = _list_dir(OFFICIAL_DIR / "train")
        test = _list_dir(OFFICIAL_DIR / "test")
        labels = [y for _, y in train_all]
        train, val = train_test_split(train_all, test_size=0.15, stratify=labels, random_state=SEED)
    elif split == "clean":
        items = _list_dir(ORIGINAL_DIR)
        labels = [y for _, y in items]
        train, rest = train_test_split(items, test_size=0.30, stratify=labels, random_state=SEED)
        val, test = train_test_split(rest, test_size=0.50, stratify=[y for _, y in rest], random_state=SEED)
    else:
        raise ValueError(split)
    parts = {"train": list(train), "val": list(val), "test": list(test)}
    if smoke:
        parts = {k: _subsample(v, 50 if k == "train" else 20) for k, v in parts.items()}
    return parts


def load_images(items, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Load grayscale images resized to size x size as uint8 (N, size, size). Cached on disk."""
    CACHE_DIR.mkdir(exist_ok=True)
    key = hashlib.sha1("|".join(str(p) for p, _ in items).encode()).hexdigest()[:12]
    cache = CACHE_DIR / f"img_{size}_{len(items)}_{key}.npz"
    y = np.array([lbl for _, lbl in items], dtype=np.int64)
    if cache.exists():
        return np.load(cache)["x"], y
    x = np.empty((len(items), size, size), dtype=np.uint8)
    for i, (p, _) in enumerate(items):
        with Image.open(p) as im:
            x[i] = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR))
    np.savez(cache, x=x)
    return x, y


def threshold_for_recall(y_true, scores, target: float = TARGET_RECALL) -> float:
    """Highest threshold whose defect recall on (validation) data is >= target."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pos = np.sort(scores[y_true == 1])
    if len(pos) == 0:
        return float(np.median(scores))
    k = int(np.floor((1 - target) * len(pos)))  # how many defects we may miss
    return float(pos[k])


def metrics_at(y_true, scores, thr: float) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(scores) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "accuracy": float((tp + tn) / len(y_true)),
        "recall_def": float(recall_score(y_true, pred, zero_division=0)),
        "precision_def": float(precision_score(y_true, pred, zero_division=0)),
        "f1_def": float(f1_score(y_true, pred, zero_division=0)),
        "false_alarm_rate": float(fp / max(fp + tn, 1)),
        "missed_defects": int(fn),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate(name: str, split: str, y_val, s_val, y_test, s_test, default_thr: float, extra: dict | None = None,
             smoke: bool = False) -> dict:
    """Score the test set at the default threshold and at the validation-chosen high-recall threshold."""
    # Never stricter than the default: if validation is perfectly separable, the 99 % rule alone can pick a
    # threshold right under the top defect scores and miss more defects on test than the default does.
    thr = min(threshold_for_recall(y_val, s_val), default_thr)
    res = {
        "method": name,
        "split": split,
        "n_test": int(len(y_test)),
        "roc_auc": float(roc_auc_score(y_test, s_test)) if len(set(y_test)) > 1 else None,
        "default": metrics_at(y_test, s_test, default_thr),
        f"recall{int(TARGET_RECALL * 100)}": metrics_at(y_test, s_test, thr),
    }
    if extra:
        res.update(extra)
    res["scores"] = {"y_val": [int(v) for v in y_val], "val": [float(v) for v in s_val],
                     "y_test": [int(v) for v in y_test], "test": [float(v) for v in s_test]}
    save_result(res, smoke)
    d, h = res["default"], res[f"recall{int(TARGET_RECALL * 100)}"]
    print(f"[{name} | {split}] AUC={res['roc_auc']:.4f}  "
          f"default: acc={d['accuracy']:.4f} recall={d['recall_def']:.4f} FA={d['false_alarm_rate']:.4f}  |  "
          f"high-recall thr: recall={h['recall_def']:.4f} FA={h['false_alarm_rate']:.4f} missed={h['missed_defects']}")
    return res


def save_result(res: dict, smoke: bool = False) -> Path:
    out_dir = RESULTS_DIR / ("smoke" if smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{res['method']}_{res['split']}.json"
    path.write_text(json.dumps(res, indent=2))
    return path


def results_dir(smoke: bool = False) -> Path:
    d = RESULTS_DIR / ("smoke" if smoke else "")
    d.mkdir(parents=True, exist_ok=True)
    return d
