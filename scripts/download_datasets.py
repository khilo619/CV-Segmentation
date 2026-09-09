#!/usr/bin/env python3
"""Dataset Acquisition Script: Fetches open-source CV datasets from Kaggle / Hugging Face.

Assigned Engineer: Engineer 3
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download open-source resume datasets.")
    parser.add_argument(
        "--source",
        choices=["kaggle", "huggingface", "all"],
        default="all",
        help="Target data source to download from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save downloaded datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Initializing dataset fetcher targeting: {output_dir}")
    print("[*] Sources configured: Kaggle (Resume Dataset, Updated Resume), Hugging Face")
    print(
        "[INFO] Ensure KAGGLE_USERNAME and KAGGLE_KEY environment variables are set if downloading from Kaggle."
    )
    # Placeholder implementation ready for Engineer 3 to hook into kaggle/datasets API
    sys.exit(0)


if __name__ == "__main__":
    main()
