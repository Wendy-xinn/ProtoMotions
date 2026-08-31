import json

import torch

from protomotions.agents.peft.offline_dataset import OfflineSFTDataset


def _write_clip(root, split, valid):
    directory = root / split
    directory.mkdir()
    torch.save(
        {
            "format_version": 1,
            "metadata": {"motion_id": 0},
            "tensors": {
                "state": torch.arange(6, dtype=torch.float16).view(3, 2),
                "target_latent": torch.arange(3, dtype=torch.long).view(3, 1),
                "valid": torch.tensor(valid),
            },
        },
        directory / "motion_0000.pt",
    )
    (directory / "cache_manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "split": split,
                "clips": [{"file": "motion_0000.pt"}],
            }
        )
    )


def test_offline_sft_dataset_filters_split_and_invalid_frames(tmp_path):
    _write_clip(tmp_path, "train", [True, False, True])
    _write_clip(tmp_path, "val", [True, True, True])

    dataset = OfflineSFTDataset(tmp_path, {"train"})

    assert len(dataset) == 2
    assert dataset[0]["state"].dtype == torch.float32
    assert dataset[0]["target_latent"].dtype == torch.long
    assert torch.equal(dataset[1]["state"], torch.tensor([4.0, 5.0]))
