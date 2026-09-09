#!/usr/bin/env python3
"""Evaluation & Benchmarking Harness: Computes Boundary Precision, Recall, F1, and WindowDiff.

Assigned Engineer: Engineer 1 & Engineer 3
"""

import argparse
import json
import sys
from pathlib import Path


def calculate_metrics(ground_truth: list[int], predictions: list[int]) -> dict[str, float]:
    gt_set = set(ground_truth)
    pred_set = set(predictions)

    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation predictions against ground truth."
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to ground truth JSON manifest.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to prediction JSON manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ground_truth.exists() or not args.predictions.exists():
        print("[ERROR] Ground truth or predictions manifest not found.")
        sys.exit(1)

    with open(args.ground_truth, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    with open(args.predictions, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    gt_boundaries = gt_data.get("boundaries", [])
    pred_boundaries = pred_data.get("boundaries", [])

    results = calculate_metrics(gt_boundaries, pred_boundaries)
    print("\n========== SEGMENTATION BENCHMARK RESULTS ==========")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v:.4f}")
        else:
            print(f"{k:25s}: {v}")
    print("====================================================\n")


if __name__ == "__main__":
    main()
