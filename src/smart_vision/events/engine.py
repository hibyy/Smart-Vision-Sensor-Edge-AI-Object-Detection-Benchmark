"""Turn raw detections into compact, meaningful sensor output.

A smart sensor sends events and counts instead of video. This engine is
class-agnostic: it works for any object class the detector can output.

Per frame it reports instantaneous per-class counts. When detections carry
track ids it also reports debounced "appeared" / "disappeared" events:
- appeared: a track was seen in `confirm_frames` frames (filters flicker)
- disappeared: a confirmed track was not seen for more than `lost_frames` frames
"""
from __future__ import annotations

from dataclasses import dataclass

from smart_vision.types import Detection


@dataclass
class _Track:
    class_name: str
    first_frame: int
    last_frame: int
    hits: int = 0
    confirmed: bool = False


class EventEngine:
    def __init__(self, classes: set[str] | None = None, min_confidence: float = 0.25,
                 confirm_frames: int = 3, lost_frames: int = 5):
        if confirm_frames < 1 or lost_frames < 0:
            raise ValueError("confirm_frames must be >= 1 and lost_frames >= 0")
        self.classes = classes  # None means every class
        self.min_confidence = min_confidence
        self.confirm_frames = confirm_frames
        self.lost_frames = lost_frames
        self._tracks: dict[int, _Track] = {}

    def update(self, frame_index: int, timestamp_s: float, detections: list[Detection]) -> dict:
        kept = [d for d in detections
                if d.confidence >= self.min_confidence
                and (self.classes is None or d.class_name in self.classes)]

        counts: dict[str, int] = {}
        for d in kept:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1

        events: list[dict] = []
        for d in kept:
            if d.track_id is None:
                continue
            track = self._tracks.get(d.track_id)
            if track is None:
                track = self._tracks[d.track_id] = _Track(d.class_name, frame_index, frame_index)
            track.hits += 1
            track.last_frame = frame_index
            if not track.confirmed and track.hits >= self.confirm_frames:
                track.confirmed = True
                events.append({"type": "appeared", "class": track.class_name,
                               "track_id": d.track_id, "frame": frame_index})

        for tid in [t for t, tr in self._tracks.items() if frame_index - tr.last_frame > self.lost_frames]:
            track = self._tracks.pop(tid)
            if track.confirmed:
                events.append({"type": "disappeared", "class": track.class_name,
                               "track_id": tid, "frame": frame_index})

        return {"frame": frame_index, "t": round(timestamp_s, 3),
                "counts": counts, "events": events}
