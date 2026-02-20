#!/usr/bin/env python3
"""Explore dataset and create visual reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def class_counts(split_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for class_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
        count = len([p for p in class_dir.iterdir() if p.is_file() and is_image(p)])
        rows.append({"class": class_dir.name, "count": count})
    return pd.DataFrame(rows)


def sample_resolutions(split_dir: Path, max_samples: int = 1500) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    i = 0
    for class_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
        for image_path in class_dir.iterdir():
            if not image_path.is_file() or not is_image(image_path):
                continue
            with Image.open(image_path) as img:
                width, height = img.size
            rows.append({"width": width, "height": height})
            i += 1
            if i >= max_samples:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dataset EDA visuals")
    parser.add_argument("--train-dir", type=Path, default=Path("data/splits/train"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    counts = class_counts(args.train_dir).sort_values("count", ascending=False)
    counts.to_csv(args.report_dir / "class_distribution.csv", index=False)

    plt.figure(figsize=(14, 6))
    sns.barplot(data=counts, x="class", y="count", color="#2A9D8F")
    plt.xticks(rotation=90)
    plt.title("Training Class Distribution")
    plt.tight_layout()
    plt.savefig(args.report_dir / "class_distribution.png", dpi=180)
    plt.close()

    res_df = sample_resolutions(args.train_dir)
    if not res_df.empty:
        plt.figure(figsize=(7, 6))
        sns.scatterplot(data=res_df, x="width", y="height", s=20, alpha=0.5, color="#E76F51")
        plt.title("Image Resolution Distribution (Sample)")
        plt.tight_layout()
        plt.savefig(args.report_dir / "image_resolution.png", dpi=180)
        plt.close()

    print(f"Saved report files to: {args.report_dir}")


if __name__ == "__main__":
    main()
