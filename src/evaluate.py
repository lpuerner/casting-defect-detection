"""Collect all results/*.json into one comparison table (results/summary.md) and chart.

Run after the method scripts:  python src/evaluate.py   (add --smoke for the smoke-test results)
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import TARGET_RECALL, results_dir

ORDER = ["hog_logreg", "hog_svm_rbf", "small_cnn", "resnet18_frozen_logreg", "resnet18_finetune", "patchcore_anomaly",
         "ensemble_soft_vote", "ensemble_stacking"]
HR = f"recall{int(TARGET_RECALL * 100)}"


def load(smoke: bool) -> pd.DataFrame:
    rows = []
    for f in sorted(results_dir(smoke).glob("*.json")):
        r = json.loads(f.read_text())
        if "method" not in r:
            continue
        d, h = r["default"], r[HR]
        rows.append({
            "method": r["method"], "split": r["split"], "n_test": r["n_test"], "roc_auc": r["roc_auc"],
            "acc": d["accuracy"], "recall_def": d["recall_def"], "precision_def": d["precision_def"],
            "f1_def": d["f1_def"], "false_alarms": d["false_alarm_rate"],
            "hr_recall_def": h["recall_def"], "hr_false_alarms": h["false_alarm_rate"], "hr_missed": h["missed_defects"],
            "train_s": r.get("train_seconds"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["order"] = df["method"].map({m: i for i, m in enumerate(ORDER)})
    return df.sort_values(["split", "order"]).drop(columns="order")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    smoke = p.parse_args().smoke
    df = load(smoke)
    if df.empty:
        print("no results yet")
        return
    out = results_dir(smoke)
    lines = ["# Results\n",
             "Default = standard threshold (0.5; anomaly: 5 % false alarms on validation ok parts).",
             f"High-recall = threshold chosen on validation so that defect recall >= {TARGET_RECALL:.0%}, "
             "then applied unchanged to test.\n"]
    for split, g in df.groupby("split", sort=False):
        lines.append(f"## Split: {split} (n_test = {g['n_test'].iloc[0]})\n")
        lines.append("| Method | ROC-AUC | Acc | Recall def | Precision def | F1 def | False alarms | "
                     "HR recall def | HR false alarms | HR missed | Train s |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            lines.append(f"| {r.method} | {r.roc_auc:.4f} | {r.acc:.4f} | {r.recall_def:.4f} | {r.precision_def:.4f} | "
                         f"{r.f1_def:.4f} | {r.false_alarms:.4f} | {r.hr_recall_def:.4f} | {r.hr_false_alarms:.4f} | "
                         f"{int(r.hr_missed)} | {r.train_s if pd.notna(r.train_s) else '–'} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    df.to_csv(out / "summary.csv", index=False)

    splits = list(df["split"].unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(6.5 * len(splits), 4), squeeze=False)
    for ax, split in zip(axes[0], splits):
        g = df[df["split"] == split]
        x = range(len(g))
        ax.bar([i - 0.2 for i in x], g["roc_auc"], width=0.4, label="ROC-AUC", color="#4C72B0")
        ax.bar([i + 0.2 for i in x], g["acc"], width=0.4, label="Accuracy", color="#DD8452")
        ax.set_xticks(list(x))
        ax.set_xticklabels(g["method"], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(max(0.5, min(g["acc"].min(), g["roc_auc"].min()) - 0.05), 1.005)
        ax.set_title(f"{split} split")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=110)
    print((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
