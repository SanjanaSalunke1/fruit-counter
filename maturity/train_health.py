"""
Fine-tune an ImageNet-pretrained torchvision model to classify healthy / unhealthy
bell peppers, using the balanced split from maturity/split_dataset.py.

Usage:
    python maturity/train_health.py --arch convnext_tiny
    python maturity/train_health.py --arch efficientnet_v2_s
    python maturity/train_health.py --arch convnext_tiny --export   # copy to backend/model
    python maturity/train_health.py --eval maturity/runs/health_convnext_tiny/best.pth

Checkpoint format: {"arch", "classes", "img_size", "state_dict", "val_acc"}.
"""
import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

HERE = Path(__file__).resolve().parent
DATA = HERE / "health_data"
BACKEND_MODEL_DIR = HERE.parent / "backend" / "model"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

ARCHS = {
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 224),
    "efficientnet_v2_s": (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.IMAGENET1K_V1, 288),
}


def build_model(arch, num_classes, pretrained=True):
    fn, weights, _ = ARCHS[arch]
    m = fn(weights=weights if pretrained else None)
    if arch.startswith("convnext"):
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    else:
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


def eval_transform(size):
    return transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_transform(size):
    return transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        # no hue shift: discoloration is part of what "unhealthy" looks like
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.2),
    ])


@torch.inference_mode()
def evaluate(model, loader, classes):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, conf = 0.0, torch.zeros(len(classes), len(classes), dtype=torch.long)
    for x, y in loader:
        out = model(x)
        total_loss += loss_fn(out, y).item()
        for t, p in zip(y, out.argmax(1)):
            conf[t, p] += 1
    n = conf.sum().item()
    return conf.diag().sum().item() / n, total_loss / n, conf


def print_report(conf, classes):
    print("  confusion (rows=true, cols=pred):", "  ".join(classes))
    for i, c in enumerate(classes):
        tp, row, col = conf[i, i].item(), conf[i].sum().item(), conf[:, i].sum().item()
        print(f"  {c:10} {conf[i].tolist()}  recall {tp / max(row, 1):.3f}  precision {tp / max(col, 1):.3f}")


def train(a):
    torch.manual_seed(a.seed)
    _, _, size = ARCHS[a.arch]
    train_ds = datasets.ImageFolder(a.data / "train", train_transform(size))
    val_ds = datasets.ImageFolder(a.data / "val", eval_transform(size))
    classes = train_ds.classes
    assert classes == val_ds.classes, "train/val class folders differ"
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=32, num_workers=0)

    model = build_model(a.arch, len(classes))
    head = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    body = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": head, "lr": a.lr * 10}],
                            weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.lr, a.lr * 10], total_steps=a.epochs * len(train_dl), pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    out_dir = HERE / "runs" / f"health_{a.arch}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pth"
    best = (-1.0, float("inf"))
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * len(y)
            seen += len(y)
        acc, vloss, _ = evaluate(model, val_dl, classes)
        mark = ""
        if (acc, -vloss) > (best[0], -best[1]):  # best acc, ties broken by lower val loss
            best = (acc, vloss)
            torch.save({"arch": a.arch, "classes": classes, "img_size": size,
                        "state_dict": model.state_dict(), "val_acc": acc}, best_path)
            mark = "  *saved"
        print(f"epoch {epoch:2}/{a.epochs}  train_loss {run_loss / seen:.4f}  "
              f"val_loss {vloss:.4f}  val_acc {acc:.3f}  ({time.time() - t0:.0f}s){mark}", flush=True)
    print(f"best val_acc {best[0]:.3f} (val_loss {best[1]:.4f}) -> {best_path}")
    return best_path


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--arch", choices=list(ARCHS), default="convnext_tiny")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", type=Path, help="checkpoint to evaluate (no training)")
    ap.add_argument("--export", action="store_true", help="copy best checkpoint into backend/model")
    a = ap.parse_args()

    ckpt_path = a.eval or train(a)
    model, ckpt = load_checkpoint(ckpt_path)
    val_ds = datasets.ImageFolder(a.data / "val", eval_transform(ckpt["img_size"]))
    acc, vloss, conf = evaluate(model, DataLoader(val_ds, batch_size=32), ckpt["classes"])
    print(f"[{ckpt['arch']}] val acc {acc:.3f}  val loss {vloss:.4f}")
    print_report(conf, ckpt["classes"])
    if a.export:
        # fp16 weights halve the file (ConvNeXt-Tiny fp32 is ~111 MB, over GitHub's
        # 100 MB limit); load_state_dict casts them back to fp32 on load.
        dst = BACKEND_MODEL_DIR / "health_classifier.pth"
        half = {k: v.half() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()}
        torch.save({**ckpt, "state_dict": half}, dst)
        model.load_state_dict(half)
        acc, _, _ = evaluate(model.float(), DataLoader(val_ds, batch_size=32), ckpt["classes"])
        print(f"Exported -> {dst} ({dst.stat().st_size / 1e6:.0f} MB, fp16 val acc {acc:.3f})")
