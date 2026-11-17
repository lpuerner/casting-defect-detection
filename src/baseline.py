"""Method 1: classic baseline. Hand-crafted features (HOG + LBP histogram) + linear / RBF classifiers.

Run:  python src/baseline.py --split official
      python src/baseline.py --split clean
"""
from __future__ import annotations

import time

import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import base_parser, evaluate, get_split, load_images, seed_everything

SIZE = 128


def features(x: np.ndarray) -> np.ndarray:
    out = []
    for img in x:
        h = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
        lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
        out.append(np.concatenate([h, lbp_hist]))
    return np.asarray(out, dtype=np.float32)


def main() -> None:
    args = base_parser(__doc__).parse_args()
    seed_everything()
    parts = get_split(args.split, args.smoke)
    t0 = time.time()
    data = {}
    for k, items in parts.items():
        x, y = load_images(items, SIZE)
        data[k] = (features(x), y)
    print(f"features: {data['train'][0].shape[1]} dims, {time.time() - t0:.1f}s")

    (xtr, ytr), (xva, yva), (xte, yte) = data["train"], data["val"], data["test"]
    models = {
        "hog_logreg": (make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)), 0.5),
        "hog_svm_rbf": (make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")), 0.0),
    }
    for name, (model, default_thr) in models.items():
        t0 = time.time()
        model.fit(xtr, ytr)
        score = (lambda m, x: m.predict_proba(x)[:, 1]) if name.endswith("logreg") else (lambda m, x: m.decision_function(x))
        evaluate(name, args.split, yva, score(model, xva), yte, score(model, xte), default_thr,
                 extra={"train_seconds": round(time.time() - t0, 1)}, smoke=args.smoke)


if __name__ == "__main__":
    main()
