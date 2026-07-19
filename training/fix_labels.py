"""
fix_labels.py
=============
One-off utility to patch YOLO label files when custom class IDs were
accidentally written as 80, 81, 82 instead of 0, 1, 2.

Usage
-----
# Default: patches all .txt files under data/labeled/labels/
python training/fix_labels.py

# Custom directory
python training/fix_labels.py --labels-dir path/to/labels
"""

import argparse
from pathlib import Path


def fix_labels(labels_dir: Path) -> None:
    """Scan all YOLO .txt label files and remap class IDs 80→0, 81→1, 82→2."""
    if not labels_dir.exists():
        print(f"[FixLabels] Directory not found: {labels_dir}")
        return

    count = 0
    total = 0
    for f in labels_dir.rglob("*.txt"):
        total += 1
        original = f.read_text()
        patched = original

        # Remap mid-line occurrences
        patched = patched.replace("\n80 ", "\n0 ")
        patched = patched.replace("\n81 ", "\n1 ")
        patched = patched.replace("\n82 ", "\n2 ")

        # Remap if the file starts with these IDs
        for old, new in [("80 ", "0 "), ("81 ", "1 "), ("82 ", "2 ")]:
            if patched.startswith(old):
                patched = new + patched[len(old):]

        if patched != original:
            f.write_text(patched)
            count += 1

    print(f"[FixLabels] Scanned {total} label files — patched {count}.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Patch YOLO label files: remap class IDs 80/81/82 -> 0/1/2.",
    )
    p.add_argument(
        "--labels-dir",
        type=str,
        default="data/labeled/labels",
        dest="labels_dir",
        help="Root directory containing YOLO .txt label files. "
             "(default: data/labeled/labels)",
    )
    args = p.parse_args()
    fix_labels(Path(args.labels_dir))


if __name__ == "__main__":
    main()
