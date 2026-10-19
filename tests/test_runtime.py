import json

import numpy as np

from smart_vision.events.agreement import event_agreement
from smart_vision.events.engine import EventEngine
from smart_vision.runtime import MotionGate, SensorPipeline
from smart_vision.types import Detection, Frame


class FakeDetector:
    """Returns one tracked 'cup' for frames whose brightness marker is above 50."""

    name = "fake"
    last_speed: dict = {}

    def __init__(self):
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        if image.mean() > 50:
            return [Detection(41, "cup", 0.9, (1, 1, 5, 5), track_id=1)]
        return []


def frames(values):
    return [Frame(i, i * 0.1, np.full((32, 32, 3), v, np.uint8)) for i, v in enumerate(values)]


def test_gate_skips_static_and_runs_on_change():
    gate = MotionGate(threshold=2.0, max_skip=100)
    imgs = [np.full((32, 32, 3), v, np.uint8) for v in (10, 10, 10, 200, 200)]
    assert [gate.should_run(i) for i in imgs] == [True, False, False, True, False]


def test_gate_max_skip_forces_run():
    gate = MotionGate(threshold=50.0, max_skip=2)
    img = np.zeros((32, 32, 3), np.uint8)
    assert [gate.should_run(img) for _ in range(6)] == [True, False, False, True, False, False]


def test_pipeline_full_vs_adaptive_and_jsonl(tmp_path):
    seq = [10, 10, 10, 200, 200, 200, 200, 10, 10, 10]
    det = FakeDetector()
    out = tmp_path / "ev.jsonl"
    pipe = SensorPipeline(det, EventEngine(confirm_frames=2, lost_frames=1), None, out)
    list(pipe.run(frames(seq)))
    full_events = pipe.events
    assert det.calls == 10
    assert any(e["type"] == "appeared" for e in full_events)
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert 0 < len(lines) < 10  # report by exception: fewer records than frames

    det2 = FakeDetector()
    pipe2 = SensorPipeline(det2, EventEngine(confirm_frames=2, lost_frames=1), MotionGate(2.0, 50))
    list(pipe2.run(frames(seq)))
    assert det2.calls < 10
    summary = pipe2.telemetry.summary()
    assert summary["compute_fraction"] < 1.0 and summary["frames"] == 10
    assert summary["bandwidth_reduction"] > 0.9


def test_event_agreement():
    ref = [{"type": "appeared", "class": "cup", "frame": 10}, {"type": "appeared", "class": "dog", "frame": 50}]
    cand = [{"type": "appeared", "class": "cup", "frame": 12}, {"type": "appeared", "class": "cat", "frame": 50}]
    r = event_agreement(ref, cand, tolerance_frames=5)
    assert r["matched"] == 1 and r["precision"] == 0.5 and r["recall"] == 0.5
    assert event_agreement([], [])["f1"] == 0.0 or event_agreement([], [])["precision"] == 1.0
