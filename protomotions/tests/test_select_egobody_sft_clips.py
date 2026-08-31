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
