"""The sensor pipeline: frame -> (optional motion gate) -> detector -> event engine -> record."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from smart_vision.events.engine import EventEngine
from smart_vision.runtime.policy import MotionGate
from smart_vision.types import Detection, Frame


@dataclass
class Telemetry:
    frames: int = 0
    inferred: int = 0
    inference_ms: list[float] = field(default_factory=list)
    frame_ms: list[float] = field(default_factory=list)
    records_emitted: int = 0
    bytes_emitted: int = 0
    raw_bytes: int = 0
    wall_s: float = 0.0

    def summary(self) -> dict:
        def med(v):
            return float(np.median(v)) if v else 0.0

        return {
            "frames": self.frames,
            "frames_inferred": self.inferred,
            "compute_fraction": round(self.inferred / self.frames, 4) if self.frames else 0.0,
            "median_inference_ms": round(med(self.inference_ms), 2),
            "mean_ms_per_frame": round(float(np.mean(self.frame_ms)), 2) if self.frame_ms else 0.0,
            "fps": round(self.frames / self.wall_s, 2) if self.wall_s > 0 else 0.0,
            "records_emitted": self.records_emitted,
            "bytes_emitted": self.bytes_emitted,
            "raw_frame_bytes": self.raw_bytes,
            "bandwidth_reduction": (round(1 - self.bytes_emitted / self.raw_bytes, 6)
                                    if self.raw_bytes else 0.0),
        }


class SensorPipeline:
    """Process frames and emit compact records ("report by exception").

    A record is written to the JSONL file only when the per-class counts changed
    or an event occurred, which is what keeps the output tiny compared to video.
    """

    def __init__(self, detector, engine: EventEngine | None = None, gate: MotionGate | None = None,
                 jsonl_path: str | Path | None = None, emit_on_change: bool = True):
        self.detector = detector
        self.engine = engine or EventEngine()
        self.gate = gate
        self.emit_on_change = emit_on_change
        self.telemetry = Telemetry()
        self._held: list[Detection] = []
        self._last_counts: dict[str, int] | None = None
        self._all_events: list[dict] = []
        self._fh = None
        if jsonl_path:
            path = Path(jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("w", encoding="utf-8")

    @property
    def events(self) -> list[dict]:
        return self._all_events

    def process(self, frame: Frame) -> tuple[list[Detection], dict]:
        t0 = time.perf_counter()
        run = True if self.gate is None else self.gate.should_run(frame.image)
        infer_ms = 0.0
        if run:
            t1 = time.perf_counter()
            self._held = self.detector.detect(frame.image)
            infer_ms = (time.perf_counter() - t1) * 1000
        record = self.engine.update(frame.index, frame.timestamp_s, self._held)
        record["inference"] = {"ran": run, "ms": round(infer_ms, 2)}

        tel = self.telemetry
        tel.frames += 1
        tel.raw_bytes += int(frame.image.nbytes)
        if run:
            tel.inferred += 1
            tel.inference_ms.append(infer_ms)
        self._all_events.extend(record["events"])
        changed = record["events"] or record["counts"] != self._last_counts
        if changed or not self.emit_on_change:
            self._last_counts = dict(record["counts"])
            line = json.dumps({k: record[k] for k in ("frame", "t", "counts", "events")},
                              separators=(",", ":"))
            tel.records_emitted += 1
            tel.bytes_emitted += len(line) + 1
            if self._fh:
                self._fh.write(line + "\n")
        tel.frame_ms.append((time.perf_counter() - t0) * 1000)
        return self._held, record

    def run(self, source: Iterable[Frame], max_frames: int | None = None) -> Iterator[tuple[Frame, list[Detection], dict]]:
        start = time.perf_counter()
        try:
            for n, frame in enumerate(source):
                if max_frames is not None and n >= max_frames:
                    break
                dets, record = self.process(frame)
                self.telemetry.wall_s = time.perf_counter() - start
                yield frame, dets, record
        finally:
            self.telemetry.wall_s = time.perf_counter() - start
            self.close()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
