"""
Train the YOLO11n Ripe/Overripe classifier used by backend/app.py.

Usage:
    python maturity/train.py                          # train on maturity/data, copy best.pt
                                                      # to backend/model/maturity_yolo11n_cls.pt
    python maturity/train.py --eval path/to/best.pt   # accuracy on held-out days (+ test folder)
    python maturity/train.py --eval path/to/best.pt --test-dir my_photos
        (test folder layout: my_photos/Ripe/*.jpg, my_photos/Overripe/*.jpg)
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
BACKEND_MODEL = HERE.parent / "backend" / "model" / "maturity_yolo11n_cls.pt"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def evaluate(weights, folders):
    model = YOLO(str(weights))
    for set_name, folder in folders.items():
        per_class = {}
        for cls_dir in sorted(d for d in folder.iterdir() if d.is_dir()):
            paths = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
            ok = 0
            for p in paths:
                r = model(str(p), imgsz=224, device="cpu", verbose=False)[0]
                ok += r.names[r.probs.top1] == cls_dir.name
            per_class[cls_dir.name] = (ok, len(paths))
        tot_ok = sum(o for o, _ in per_class.values())
        tot = sum(n for _, n in per_class.values())
        detail = "  ".join(f"{c} {o}/{n}" for c, (o, n) in per_class.items())
        print(f"  {set_name:14} acc {tot_ok}/{tot} = {tot_ok / max(tot, 1):.3f}  ({detail})")


def train(a):
    model = YOLO(a.model)
    model.train(
        data=str(a.data), epochs=a.epochs, imgsz=224, batch=32, patience=a.patience,
        device=a.device, workers=0, project=str(HERE / "runs"), name=a.name, exist_ok=True,
        seed=0,
        hsv_h=0.0,          # hue IS the label (green vs red/yellow) - never shift it
        hsv_s=0.4, hsv_v=0.4, degrees=15, scale=0.5, fliplr=0.5, flipud=0.2,
        erasing=0.3, dropout=0.2, plots=True,
    )
    best = HERE / "runs" / a.name / "weights" / "best.pt"
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--model", default="yolo11n-cls.pt")
    ap.add_argument("--name", default="maturity_yolo11n")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval", type=Path, help="weights to evaluate (no training)")
    ap.add_argument("--test-dir", type=Path, help="extra test folder with Ripe/ and Overripe/")
    ap.add_argument("--no-export", action="store_true", help="don't copy best.pt into backend/model")
    a = ap.parse_args()

    folders = {"held-out days": a.data / "val"}
    if a.test_dir:
        folders["test-dir"] = a.test_dir
    if a.eval:
        evaluate(a.eval, folders)
    else:
        best = train(a)
        evaluate(best, folders)
        if not a.no_export:
            shutil.copy2(best, BACKEND_MODEL)
            print(f"Copied {best} -> {BACKEND_MODEL}")
