import json
import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[2] / "data" / "scripts" / "select_egobody_sft_clips.py"
)


def _load_selector():
    spec = importlib.util.spec_from_file_location("select_egobody_sft_clips", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _candidate(recording: str, start: int, score: float) -> dict:
    return {"recording": recording, "start": start, "motion_score": score}


def test_balanced_top_spreads_clips_and_enforces_start_gap():
    selector = _load_selector()
    candidates = {
        "a": [
            _candidate("a", 0, 10.0),
            _candidate("a", 48, 9.0),
            _candidate("a", 96, 8.0),
        ],
        "b": [
            _candidate("b", 0, 7.0),
            _candidate("b", 96, 6.0),
        ],
    }

    selected = selector.balanced_top(candidates, count=4, min_start_gap=96)

    assert [(item["recording"], item["start"]) for item in selected] == [
        ("a", 0),
        ("a", 96),
        ("b", 0),
        ("b", 96),
    ]


def test_balanced_top_rejects_an_unavailable_clip_budget():
    selector = _load_selector()
    candidates = {"a": [_candidate("a", 0, 1.0)]}

    with pytest.raises(RuntimeError, match="Only 1 spaced clips"):
        selector.balanced_top(candidates, count=2, min_start_gap=96)


def test_body_text_segments_drop_the_repeated_boundary_frame(tmp_path):
    selector = _load_selector()
    text_dir = tmp_path / "recording" / "body_idx_0"
    text_dir.mkdir(parents=True)
    (text_dir / "000.json").write_text(
        json.dumps({"0": "a", "1": "b", "2": "c"})
    )
    (text_dir / "001.json").write_text(
        json.dumps({"0": "c", "1": "d", "2": "e"})
    )
    info = {"body_idx_fpv": "0 female", "start_frame": "10", "end_frame": "14"}

    body_name, timeline, exact = selector.load_body_text_timeline(
        tmp_path, "recording", info
    )

    assert body_name == "body_idx_0"
    assert timeline == ["a", "b", "c", "d", "e"]
    assert exact is True


def test_body_text_length_mismatch_is_explicitly_recording_level(tmp_path):
    selector = _load_selector()
    text_dir = tmp_path / "recording" / "body_idx_1"
    text_dir.mkdir(parents=True)
    (text_dir / "000.json").write_text(json.dumps({"0": "a", "1": "b"}))
    info = {"body_idx_fpv": "1 male", "start_frame": "10", "end_frame": "14"}

    _, timeline, exact = selector.load_body_text_timeline(
        tmp_path, "recording", info
    )

    assert timeline == ["a", "b"]
    assert exact is False
