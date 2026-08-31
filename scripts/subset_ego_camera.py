#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Subset a packaged per-motion ego-camera trajectory file."""

import argparse
import copy
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    motions = payload.get("motions")
    if not isinstance(motions, list):
        raise ValueError(f"{args.input} does not contain a motions list")
    invalid = [index for index in args.indices if not 0 <= index < len(motions)]
    if invalid:
        raise ValueError(f"Camera indices out of range [0, {len(motions)}): {invalid}")

    output = copy.deepcopy(payload)
    output["motions"] = [copy.deepcopy(motions[index]) for index in args.indices]
    offsets = output.get("retarget_root_height_offsets_m")
    if torch.is_tensor(offsets):
        output["retarget_root_height_offsets_m"] = offsets[args.indices].clone()
    output["source_motion_indices"] = list(args.indices)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    torch.save(output, temporary)
    os.replace(temporary, args.output)
    print(f"Saved {len(args.indices)} camera trajectories to {args.output}")


if __name__ == "__main__":
    main()
