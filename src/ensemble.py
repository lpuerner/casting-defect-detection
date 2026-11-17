"""Method 5: ensembles built from the saved validation/test scores of the other methods (no retraining).

Every method script stores its val and test scores in results/{method}_{split}.json. The splits are
deterministic, so row i is the same image in every file (checked below).

The methods output scores on different scales (probabilities, SVM margins, anomaly distances), so each
member is first calibrated to a probability with Platt scaling (1-D logistic regression) fitted on the
validation set. To keep the validation scores honest for threshold selection, the calibrated validation
scores are out-of-fold (5-fold cross_val_predict); the test scores use a calibrator fitted on all of val.

ensemble_soft_vote  mean of the calibrated probabilities of the strong supervised members
ensemble_stacking   logistic regression on the calibrated logits of ALL members (incl. anomaly detection),
                    i.e. the ensemble learns how much to trust each member

Run after the other scripts:  python src/ensemble.py --split official
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from common import base_parser, evaluate, results_dir, seed_everything, SEED

ALL_MEMBERS = ["hog_logreg", "hog_svm_rbf", "small_cnn", "resnet18_frozen_logreg", "resnet18_finetune",
               "patchcore_anomaly"]
SOFT_VOTE = ["hog_svm_rbf", "small_cnn", "resnet18_frozen_logreg", "resnet18_finetune"]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_members(split: str, smoke: bool) -> tuple[dict, np.ndarray, np.ndarray]:
    members, y_val, y_test = {}, None, None
    for m in ALL_MEMBERS:
        f = results_dir(smoke) / f"{m}_{split}.json"
        if not f.exists():
            print(f"  skipping {m}: {f.name} not found")
            continue
        s = json.loads(f.read_text())["scores"]
        if y_val is None:
            y_val, y_test = np.array(s["y_val"]), np.array(s["y_test"])
        elif not (np.array_equal(y_val, s["y_val"]) and np.array_equal(y_test, s["y_test"])):
            raise ValueError(f"{m}: label order differs, the score files do not belong to the same split")
        members[m] = (np.array(s["val"]), np.array(s["test"]))
    return members, y_val, y_test


def calibrate(s_val: np.ndarray, s_test: np.ndarray, y_val: np.ndarray, cv) -> tuple[np.ndarray, np.ndarray]:
    """Platt scaling: out-of-fold probabilities for val, full-val fit for test."""
    mu, sd = s_val.mean(), s_val.std() + 1e-12
    xv, xt = ((s_val - mu) / sd)[:, None], ((s_test - mu) / sd)[:, None]
    lr = LogisticRegression(C=1.0)
    p_val = cross_val_predict(lr, xv, y_val, cv=cv, method="predict_proba")[:, 1]
    p_test = lr.fit(xv, y_val).predict_proba(xt)[:, 1]
    return p_val, p_test


def main() -> None:
    args = base_parser(__doc__).parse_args()
    seed_everything()
    members, y_val, y_test = load_members(args.split, args.smoke)
    if len(members) < 2:
        raise SystemExit("need at least two member results; run the method scripts first")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cal = {m: calibrate(sv, st, y_val, cv) for m, (sv, st) in members.items()}

    # --- soft voting ---------------------------------------------------------------
    vote = [m for m in SOFT_VOTE if m in cal]
    p_val = np.mean([cal[m][0] for m in vote], axis=0)
    p_test = np.mean([cal[m][1] for m in vote], axis=0)
    evaluate("ensemble_soft_vote", args.split, y_val, p_val, y_test, p_test, 0.5,
             extra={"members": vote, "calibration": "Platt scaling on val (out-of-fold for val scores)"},
             smoke=args.smoke)

    # --- stacking ------------------------------------------------------------------
    names = list(cal)
    z_val = np.column_stack([logit(cal[m][0]) for m in names])
    z_test = np.column_stack([logit(cal[m][1]) for m in names])
    meta = LogisticRegression(C=0.5, max_iter=2000)
    s_val = cross_val_predict(meta, z_val, y_val, cv=cv, method="predict_proba")[:, 1]
    meta.fit(z_val, y_val)
    s_test = meta.predict_proba(z_test)[:, 1]
    weights = {m: round(float(w), 3) for m, w in zip(names, meta.coef_[0])}
    print("  stacking weights:", weights)
    evaluate("ensemble_stacking", args.split, y_val, s_val, y_test, s_test, 0.5,
             extra={"members": names, "weights": weights,
                    "calibration": "Platt scaling on val (out-of-fold for val scores)"},
             smoke=args.smoke)


if __name__ == "__main__":
    main()
