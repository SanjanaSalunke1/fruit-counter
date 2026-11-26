"""
Build a leak-free, class-balanced Ripe/Overripe dataset from VegNet bell peppers.

1. Collects every unique VegNet bell-pepper original (IMG_*.jpg) for Ripe, Old, Dried
   (Old + Dried -> Overripe). The numbered files in VegNet are re-encoded copies, skipped.
2. Splits by shooting DAY, so burst frames of one session never land in both train and val.
3. Balances train: Overripe is evenly subsampled to the Ripe count.

Output: maturity/data/{train,val}/{Ripe,Overripe}   (originals are never modified)

Usage:
    python maturity/build_dataset.py --vegnet "<path to VegNet>/New VegNet/1. Bell Pepper"
    (or set the VEGNET_BELL_PEPPER environment variable)
"""
import argparse
import hashlib
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOLDER_TO_CLASS = {"Ripe": "Ripe", "Old": "Overripe", "Dried": "Overripe"}
# held-out shooting days per source folder (~15-25% of each)
VAL_DAYS = {"Ripe": {"20220530", "20220606"}, "Old": {"20220609"}, "Dried": {"20220615"}}
DATE_RE = re.compile(r"IMG_(\d{8})_\d{6}")


def collect(vegnet):
    seen, items = set(), defaultdict(lambda: {"train": [], "val": []})
    for folder in FOLDER_TO_CLASS:
        for p in sorted((vegnet / folder).glob("IMG_*.jpg")):
            h = hashlib.md5(p.read_bytes()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            day = DATE_RE.search(p.name).group(1)
            items[folder]["val" if day in VAL_DAYS[folder] else "train"].append(p)
    return items


def evenly_spaced(paths, k):
    paths = sorted(paths, key=lambda p: p.name)
    return paths if k >= len(paths) else [paths[int(i * len(paths) / k)] for i in range(k)]


def main(vegnet, out):
    if not all((vegnet / f).is_dir() for f in FOLDER_TO_CLASS):
        raise SystemExit(f"{vegnet} must contain the folders {', '.join(FOLDER_TO_CLASS)}")
    if out.exists():
        shutil.rmtree(out)
    items = collect(vegnet)

    # balance train: Overripe (Old + Dried) subsampled to the Ripe count, proportionally
    target = len(items["Ripe"]["train"])
    n_old, n_dried = len(items["Old"]["train"]), len(items["Dried"]["train"])
    if n_old + n_dried > target:
        k_old = round(target * n_old / (n_old + n_dried))
        items["Old"]["train"] = evenly_spaced(items["Old"]["train"], k_old)
        items["Dried"]["train"] = evenly_spaced(items["Dried"]["train"], target - k_old)

    for folder, cls in FOLDER_TO_CLASS.items():
        for split in ("train", "val"):
            dst = out / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for p in items[folder][split]:
                shutil.copy2(p, dst / f"{folder.lower()}_{p.name}")

    for split in ("train", "val"):
        counts = {c.name: len(list(c.iterdir())) for c in sorted((out / split).iterdir())}
        print(f"{split}: {counts}")
    for folder in FOLDER_TO_CLASS:
        print(f"  {folder:5}: train {len(items[folder]['train'])}, "
              f"val {len(items[folder]['val'])} (days {', '.join(sorted(VAL_DAYS[folder]))})")
    print(f"Written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vegnet", default=os.getenv("VEGNET_BELL_PEPPER"),
                    help="VegNet '1. Bell Pepper' folder (contains Ripe/, Old/, Dried/)")
    ap.add_argument("--out", default=str(HERE / "data"))
    a = ap.parse_args()
    if not a.vegnet:
        ap.error("pass --vegnet or set VEGNET_BELL_PEPPER")
    main(Path(a.vegnet), Path(a.out))
