import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "data/scripts/align_soma23_retarget_foot_height.py"
)
SPEC = importlib.util.spec_from_file_location("soma23_grounding", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _motion(num_bodies: int, *, contacts: bool) -> dict:
    frames = 4
    positions = torch.zeros(frames, num_bodies, 3)
    rotations = torch.zeros(frames, num_bodies, 4)
    rotations[..., 3] = 1.0
    contact_tensor = torch.zeros(frames, num_bodies, dtype=torch.bool)
    if contacts:
        contact_tensor[:, 7] = True
    return {
        "gts": positions,
        "grs": rotations,
        "contacts": contact_tensor,
        "length_starts": torch.tensor([0]),
        "motion_num_frames": torch.tensor([frames]),
    }


def _foot_geometries() -> dict:
    geometry = MODULE.BoxGeometry(
        position=torch.zeros(3, dtype=torch.float64),
        rotation=torch.eye(3, dtype=torch.float64),
        half_size=torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64),
    )
    return {pair[3]: geometry for pair in MODULE.FOOT_BODY_PAIRS}


def test_unlabelled_contacts_preserve_height_without_labelled_reference():
    source = _motion(24, contacts=False)
    target = _motion(23, contacts=False)
    for _, target_id, _, _ in MODULE.FOOT_BODY_PAIRS:
        target["gts"][:, target_id, 2] = 0.2

    output, offsets = MODULE.align_retargeted_motion(
        source, target, _foot_geometries()
    )

    assert offsets.tolist() == pytest.approx([0.0])
    assert output["gts"][0, 0, 2].item() == pytest.approx(0.0)
    assert output["retarget_unlabelled_support_fallback_motion_ids"].tolist() == [0]
    assert output["retarget_unlabelled_support_fallback_mask"].tolist() == [True]
    assert output["retarget_unlabelled_support_fallback"] == "zero_offset"


def test_unlabelled_contacts_reuse_labelled_motion_calibration():
    source_a = _motion(24, contacts=False)
    source_b = _motion(24, contacts=True)
    target_a = _motion(23, contacts=False)
    target_b = _motion(23, contacts=False)
    source = {
        key: torch.cat([source_a[key], source_b[key]])
        for key in ("gts", "grs", "contacts")
    }
    target = {
        key: torch.cat([target_a[key], target_b[key]])
        for key in ("gts", "grs", "contacts")
    }
    for payload in (source, target):
        payload["length_starts"] = torch.tensor([0, 4])
        payload["motion_num_frames"] = torch.tensor([4, 4])
    for _, target_id, _, _ in MODULE.FOOT_BODY_PAIRS:
        target["gts"][:, target_id, 2] = 0.2

    output, offsets = MODULE.align_retargeted_motion(
        source, target, _foot_geometries()
    )

    assert offsets.tolist() == pytest.approx([-0.097, -0.097])
    assert output["retarget_unlabelled_support_fallback"] == (
        "median_labelled_motion_offset"
    )


def test_unlabelled_contacts_can_remain_strict():
    with pytest.raises(ValueError, match="no support-foot contact"):
        MODULE.align_retargeted_motion(
            _motion(24, contacts=False),
            _motion(23, contacts=False),
            _foot_geometries(),
            allow_unlabelled_support_fallback=False,
        )


def test_missing_contact_field_uses_zero_offset_fallback():
    source = _motion(24, contacts=False)
    del source["contacts"]

    output, offsets = MODULE.align_retargeted_motion(
        source, _motion(23, contacts=False), _foot_geometries()
    )

    assert offsets.tolist() == [0.0]
    assert output["retarget_unlabelled_support_fallback_mask"].tolist() == [True]
