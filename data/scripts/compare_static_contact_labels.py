#!/usr/bin/env python3
"""Compare geometric static-contact labels and an optional PhysX diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _scores(pred: np.ndarray, ref: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(pred, dtype=bool)
    ref = np.asarray(ref, dtype=bool)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {ref.shape}")
    tp = int(np.logical_and(pred, ref).sum())
    fp = int(np.logical_and(pred, ~ref).sum())
    fn = int(np.logical_and(~pred, ref).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * tp / max(2 * tp + fp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--physx-report", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with np.load(args.labels) as labels:
        result = {
            "labels": str(args.labels.resolve()),
            "source_vs_target": _scores(
                labels["source_contact"], labels["target_contact"]
            ),
            "intended_vs_target": _scores(
                labels["intended_contact"], labels["target_contact"]
            ),
            "intended_vs_compatible": _scores(
                labels["intended_contact"], labels["target_compatible"]
            ),
            "training_vs_target": _scores(
                labels["training_contact"], labels["target_contact"]
            ),
        }
    if args.physx_report is not None:
        report = json.loads(args.physx_report.read_text(encoding="utf-8"))
        result["physx_pair_diagnostic"] = report.get("pair_micro", report.get("micro"))

    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
