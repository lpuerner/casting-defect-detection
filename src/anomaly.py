"""Method 4 (optional): anomaly detection trained ONLY on ok parts (PatchCore-style, no training loop).

Idea: describe every image patch with mid-level ResNet18 features, keep a memory bank of patches from
ok parts, and score a new image by how far its most unusual patch is from anything in the bank.
No defect images are needed to build the model; they are only used to evaluate it.

Run:  python src/anomaly.py --split official
      python src/anomaly.py --split clean
"""
from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from common import base_parser, evaluate, get_split, load_images, results_dir, seed_everything
from torch_utils import prep_imagenet


class PatchFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.layer2, self.layer3 = m.layer2, m.layer3

    @torch.no_grad()
    def forward(self, x):
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], dim=1)
        return F.avg_pool2d(f, 3, stride=1, padding=1)  # local neighbourhood context


def patch_embed(net, x: np.ndarray, bs: int = 32) -> torch.Tensor:
    """(N,H,W) uint8 -> (N, h*w, C) patch features."""
    out = []
    for i in range(0, len(x), bs):
        f = net(prep_imagenet(x[i:i + bs]))
        out.append(f.flatten(2).transpose(1, 2))
    return torch.cat(out)


def score(net, bank: torch.Tensor, x: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Image score = distance of its most unusual patch to the nearest ok patch in the bank."""
    scores, maps = [], []
    for i in range(0, len(x), 32):
        for f in patch_embed(net, x[i:i + 32]):
            d = torch.cdist(f, bank).min(dim=1).values
            scores.append(d.max().item())
            maps.append(d.view(grid, grid).numpy())
    return np.array(scores), np.stack(maps)


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--bank", type=int, default=30000, help="memory-bank size (random patch subsample)")
    args = p.parse_args()
    seed_everything()
    parts = get_split(args.split, args.smoke)
    data = {k: load_images(v, args.size) for k, v in parts.items()}
    (xtr, ytr), (xva, yva), (xte, yte) = data["train"], data["val"], data["test"]
    net = PatchFeatures()

    t0 = time.time()
    ok_train = xtr[ytr == 0]
    target = args.bank if not args.smoke else 3000
    per_img = max(1, int(np.ceil(target / len(ok_train))))  # random patches kept per ok image
    chunks, grid = [], None
    for i in range(0, len(ok_train), 32):
        f = patch_embed(net, ok_train[i:i + 32])
        grid = int(np.sqrt(f.shape[1]))
        idx = torch.stack([torch.randperm(f.shape[1])[:per_img] for _ in range(len(f))])
        chunks.append(torch.gather(f, 1, idx.unsqueeze(-1).expand(-1, -1, f.shape[-1])).reshape(-1, f.shape[-1]))
    bank = torch.cat(chunks)
    n_bank = len(bank)
    print(f"memory bank: {n_bank:,} patches from {len(ok_train)} ok images ({time.time() - t0:.0f}s)")

    s_val, _ = score(net, bank, xva, grid)
    s_te, maps = score(net, bank, xte, grid)
    default_thr = float(np.quantile(s_val[yva == 0], 0.95))  # 5 % false alarms on validation ok parts
    evaluate("patchcore_anomaly", args.split, yva, s_val, yte, s_te, default_thr,
             extra={"train_seconds": round(time.time() - t0, 1), "size": args.size, "bank": n_bank,
                    "trained_on": "ok images only"}, smoke=args.smoke)

    # anomaly maps for a few test defects and ok parts
    out = results_dir(args.smoke)
    rng = np.random.default_rng(0)
    pick = list(rng.choice(np.where(yte == 1)[0], 4, replace=False)) + \
        list(rng.choice(np.where(yte == 0)[0], 4, replace=False))
    vmax = np.quantile(maps[pick], 0.99)
    fig, axes = plt.subplots(2, len(pick), figsize=(2.2 * len(pick), 4.8))
    for j, i in enumerate(pick):
        axes[0, j].imshow(xte[i], cmap="gray")
        axes[0, j].set_title(f"{'DEF' if yte[i] else 'OK'}  s={s_te[i]:.1f}", fontsize=8)
        axes[1, j].imshow(xte[i], cmap="gray")
        up = np.kron(maps[i], np.ones((args.size // grid, args.size // grid)))
        axes[1, j].imshow(up, cmap="jet", alpha=0.45, vmin=0, vmax=vmax)
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle(f"PatchCore-style anomaly maps ({args.split} split)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"anomaly_maps_{args.split}.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
