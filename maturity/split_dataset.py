"""
Split dataset/{healthy,unhealthy} into a class-balanced train/val set.

- Exact duplicate files (same bytes) are counted once, so a copy can never
  land in both train and val.
- Both classes are trimmed to the size of the smaller class, then split with
  the same val fraction, so train and val each have equal counts per class.
- Originals are copied, never moved or modified.

Output: maturity/health_data/{train,val}/{healthy,unhealthy}

Usage:
    python maturity/split_dataset.py                 # 80/20 split, seed 0
    python maturity/split_dataset.py --val 0.2 --seed 0
"""
import argparse
import hashlib
import random
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "dataset"
OUT = HERE / "health_data"
CLASSES = ["healthy", "unhealthy"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def unique_images(folder, seen):
    out = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen:
            print(f"  skip duplicate: {p.relative_to(SRC)} (same as {seen[h]})")
            continue
        seen[h] = p.relative_to(SRC)
        out.append(p)
    return out


def main(src, out, val_frac, seed):
    seen = {}
    images = {c: unique_images(src / c, seen) for c in CLASSES}
    n = min(len(v) for v in images.values())
    n_val = round(n * val_frac)
    rng = random.Random(seed)

    if out.exists():
        shutil.rmtree(out)
    for c in CLASSES:
        paths = images[c][:]
        rng.shuffle(paths)
        paths = paths[:n]  # balance classes
        splits = {"val": paths[:n_val], "train": paths[n_val:]}
        for split, items in splits.items():
            dst = out / split / c
            dst.mkdir(parents=True, exist_ok=True)
            for p in items:
                shutil.copy2(p, dst / p.name)
        print(f"{c:10} unique {len(images[c]):4}  -> train {len(splits['train'])}  val {len(splits['val'])}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.src, a.out, a.val, a.seed)
