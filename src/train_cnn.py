"""Method 2: small CNN trained from scratch on grayscale images.

Run:  python src/train_cnn.py --split official
      python src/train_cnn.py --split clean --epochs 25
"""
from __future__ import annotations

import json
import time

import torch
import torch.nn as nn

from common import base_parser, evaluate, get_split, load_images, results_dir, seed_everything
from torch_utils import predict, prep_gray, save_gradcam_grid, train


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 96))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(96, 1))

    def forward(self, x):
        return self.head(self.features(x))


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-3)
    args = p.parse_args()
    seed_everything()
    epochs = 2 if args.smoke else args.epochs
    parts = get_split(args.split, args.smoke)
    data = {k: load_images(v, args.size) for k, v in parts.items()}

    model = SmallCNN()
    print(f"SmallCNN: {sum(q.numel() for q in model.parameters()):,} parameters, {args.size}px, {epochs} epochs")
    t0 = time.time()
    model, history = train(model, data, prep_gray, epochs=epochs, lr=args.lr)
    train_s = round(time.time() - t0, 1)

    (xva, yva), (xte, yte) = data["val"], data["test"]
    s_val, s_test = predict(model, xva, prep_gray), predict(model, xte, prep_gray)
    evaluate("small_cnn", args.split, yva, s_val, yte, s_test, 0.5,
             extra={"train_seconds": train_s, "size": args.size, "history": history}, smoke=args.smoke)

    out = results_dir(args.smoke)
    torch.save(model.state_dict(), out / f"small_cnn_{args.split}.pt")
    save_gradcam_grid(model, model.features[-1], xte, yte, s_test, prep_gray,
                      out / f"gradcam_small_cnn_{args.split}.png", f"Small CNN – Grad-CAM ({args.split} split)")
    (out / f"small_cnn_{args.split}_history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
