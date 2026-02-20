#!/usr/bin/env python3
"""Prepare New Plant Diseases dataset into train/valid folders.

Supports two source layouts:
1) source/train/<class_name>/*.jpg and source/valid/<class_name>/*.jpg
2) source/<class_name>/*.jpg (single directory to split)
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def class_dirs(path: Path) -> list[Path]:
    return sorted([d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")])


def copy_files(files: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, destination / src.name)


def clear_output(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare_from_presplit(source: Path, output: Path) -> None:
    for split in ("train", "valid"):
        split_root = source / split
        for class_dir in class_dirs(split_root):
            images = [p for p in class_dir.iterdir() if p.is_file() and is_image(p)]
            copy_files(images, output / split / class_dir.name)


def prepare_from_single(source: Path, output: Path, val_size: float, seed: int) -> None:
    rng = random.Random(seed)
    for class_dir in class_dirs(source):
        images = [p for p in class_dir.iterdir() if p.is_file() and is_image(p)]
        if len(images) < 2:
            continue
        rng.shuffle(images)
        train_files, valid_files = train_test_split(images, test_size=val_size, random_state=seed)
        copy_files(train_files, output / "train" / class_dir.name)
        copy_files(valid_files, output / "valid" / class_dir.name)


def count_images(split_dir: Path) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    total = 0
    for class_dir in class_dirs(split_dir):
        n = len([p for p in class_dir.iterdir() if p.is_file() and is_image(p)])
        counts[class_dir.name] = n
        total += n
    return total, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/valid dataset structure")
    parser.add_argument("--source", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/splits"))
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    clear_output(args.output)

    has_train = (args.source / "train").exists()
    has_valid = (args.source / "valid").exists()

    if has_train and has_valid:
        prepare_from_presplit(args.source, args.output)
        print("Detected pre-split dataset; copied train/valid into data/splits.")
    else:
        prepare_from_single(args.source, args.output, args.val_size, args.seed)
        print(f"Split dataset with validation size {args.val_size:.2f} into data/splits.")

    train_total, train_counts = count_images(args.output / "train")
    valid_total, valid_counts = count_images(args.output / "valid")

    print(f"Train images: {train_total}")
    print(f"Valid images: {valid_total}")
    print(f"Classes: {len(train_counts)}")
    for class_name in sorted(train_counts):
        print(f"{class_name}: train={train_counts[class_name]}, valid={valid_counts.get(class_name, 0)}")


if __name__ == "__main__":
    main()
