"""PyTorch helpers shared by the CNN, transfer-learning and anomaly scripts."""
from __future__ import annotations

import copy
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def prep_gray(x_uint8: np.ndarray | torch.Tensor) -> torch.Tensor:
    """uint8 (N,H,W) -> float (N,1,H,W), roughly zero-centred."""
    t = torch.as_tensor(x_uint8).float().unsqueeze(1) / 255.0
    return (t - 0.5) / 0.25


def prep_imagenet(x_uint8: np.ndarray | torch.Tensor) -> torch.Tensor:
    """uint8 (N,H,W) grayscale -> float (N,3,H,W) normalised like ImageNet."""
    t = torch.as_tensor(x_uint8).float().unsqueeze(1).repeat(1, 3, 1, 1) / 255.0
    return (t - IMAGENET_MEAN) / IMAGENET_STD


def augment(xb: torch.Tensor) -> torch.Tensor:
    """Cheap on-the-fly augmentation for a uint8 batch (N,H,W). Impellers are rotation-symmetric."""
    if torch.rand(1) < 0.5:
        xb = torch.flip(xb, dims=[2])
    if torch.rand(1) < 0.5:
        xb = torch.flip(xb, dims=[1])
    xb = torch.rot90(xb, k=int(torch.randint(0, 4, (1,))), dims=[1, 2])
    gain = 1 + (torch.rand(xb.shape[0], 1, 1) - 0.5) * 0.3  # +-15 % brightness
    return (xb.float() * gain).clamp(0, 255).to(torch.uint8)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, prep, bs: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        out.append(torch.sigmoid(model(prep(x[i:i + bs])).squeeze(1)))
    return torch.cat(out).numpy()


def train(model: nn.Module, data: dict, prep, epochs: int, lr: float, bs: int = 32,
          params=None, log=print) -> tuple[nn.Module, list[dict]]:
    """Train with BCE, keep the weights with the best validation AUC."""
    xtr, ytr = data["train"]
    xva, yva = data["val"]
    xtr_t = torch.as_tensor(xtr)
    ytr_t = torch.as_tensor(ytr).float()
    opt = torch.optim.AdamW(params if params is not None else model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_auc, best_state, history = -1.0, None, []
    for ep in range(epochs):
        model.train()
        t0, total = time.time(), 0.0
        perm = torch.randperm(len(xtr_t))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            logits = model(prep(augment(xtr_t[idx]))).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        s_val = predict(model, xva, prep)
        auc = roc_auc_score(yva, s_val) if len(set(yva)) > 1 else 0.0
        acc = float(((s_val >= 0.5) == yva).mean())
        history.append({"epoch": ep + 1, "loss": total / len(perm), "val_auc": auc, "val_acc": acc,
                        "seconds": round(time.time() - t0, 1)})
        log(f"  epoch {ep + 1}/{epochs}  loss={total / len(perm):.4f}  val_auc={auc:.4f}  val_acc={acc:.4f}  "
            f"({time.time() - t0:.0f}s)")
        if auc > best_auc:
            best_auc, best_state = auc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history


def grad_cam(model: nn.Module, layer: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Grad-CAM heatmaps (N,H,W) in [0,1] for the defect logit."""
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    model.eval()
    model.zero_grad()
    model(x).sum().backward()
    h1.remove()
    h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return cam.detach().numpy()


def save_gradcam_grid(model, layer, x_uint8: np.ndarray, y: np.ndarray, scores: np.ndarray, prep,
                      path: Path, title: str, n: int = 8) -> None:
    """Show defect and ok examples (incl. the worst mistakes) with their Grad-CAM overlay."""
    def_idx = np.where(y == 1)[0]
    ok_idx = np.where(y == 0)[0]
    pick = list(def_idx[np.argsort(scores[def_idx])][: n // 4])           # defects scored lowest (hardest)
    pick += list(np.random.default_rng(0).choice(def_idx, n // 4, replace=False))
    pick += list(ok_idx[np.argsort(-scores[ok_idx])][: n // 4])           # ok parts scored highest
    pick += list(np.random.default_rng(1).choice(ok_idx, n // 4, replace=False))
    cams = grad_cam(model, layer, prep(x_uint8[pick]))
    fig, axes = plt.subplots(2, len(pick), figsize=(2.2 * len(pick), 4.8))
    for j, i in enumerate(pick):
        axes[0, j].imshow(x_uint8[i], cmap="gray")
        axes[0, j].set_title(f"{'DEF' if y[i] else 'OK'}  p={scores[i]:.2f}", fontsize=8)
        axes[1, j].imshow(x_uint8[i], cmap="gray")
        axes[1, j].imshow(cams[j], cmap="jet", alpha=0.45)
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
