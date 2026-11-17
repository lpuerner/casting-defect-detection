"""Method 3: transfer learning with an ImageNet-pretrained ResNet18 (torchvision).

Stage A  frozen backbone as feature extractor + logistic regression  (fast, no backprop)
Stage B  fine-tune layer4 + a new head                               (main model)

Run:  python src/transfer.py --split official
      python src/transfer.py --split clean --epochs 8
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet18_Weights, resnet18

from common import base_parser, evaluate, get_split, load_images, results_dir, seed_everything
from torch_utils import predict, prep_imagenet, save_gradcam_grid, train


@torch.no_grad()
def embed(backbone: nn.Module, x: np.ndarray, bs: int = 64) -> np.ndarray:
    backbone.eval()
    return torch.cat([backbone(prep_imagenet(x[i:i + bs])) for i in range(0, len(x), bs)]).numpy()


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--size", type=int, default=160)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--skip-finetune", action="store_true")
    args = p.parse_args()
    seed_everything()
    epochs = 1 if args.smoke else args.epochs
    parts = get_split(args.split, args.smoke)
    data = {k: load_images(v, args.size) for k, v in parts.items()}
    (xtr, ytr), (xva, yva), (xte, yte) = data["train"], data["val"], data["test"]
    out = results_dir(args.smoke)

    # --- Stage A: frozen features + logistic regression -------------------------------
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    t0 = time.time()
    f_tr, f_va, f_te = embed(model, xtr), embed(model, xva), embed(model, xte)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)).fit(f_tr, ytr)
    evaluate("resnet18_frozen_logreg", args.split, yva, clf.predict_proba(f_va)[:, 1],
             yte, clf.predict_proba(f_te)[:, 1], 0.5,
             extra={"train_seconds": round(time.time() - t0, 1), "size": args.size}, smoke=args.smoke)
    if args.skip_finetune:
        return

    # --- Stage B: fine-tune layer4 + head ----------------------------------------------
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 1))
    for name, prm in model.named_parameters():
        prm.requires_grad = name.startswith(("layer4", "fc"))
    trainable = [q for q in model.parameters() if q.requires_grad]
    print(f"Fine-tuning ResNet18 layer4+fc: {sum(q.numel() for q in trainable):,} trainable params, "
          f"{args.size}px, {epochs} epochs")
    t0 = time.time()
    model, history = train(model, data, prep_imagenet, epochs=epochs, lr=args.lr, params=trainable)
    s_val, s_test = predict(model, xva, prep_imagenet), predict(model, xte, prep_imagenet)
    evaluate("resnet18_finetune", args.split, yva, s_val, yte, s_test, 0.5,
             extra={"train_seconds": round(time.time() - t0, 1), "size": args.size, "history": history},
             smoke=args.smoke)
    torch.save(model.state_dict(), out / f"resnet18_finetune_{args.split}.pt")
    save_gradcam_grid(model, model.layer4, xte, yte, s_test, prep_imagenet,
                      out / f"gradcam_resnet18_{args.split}.png", f"ResNet18 fine-tuned – Grad-CAM ({args.split} split)")


if __name__ == "__main__":
    main()
