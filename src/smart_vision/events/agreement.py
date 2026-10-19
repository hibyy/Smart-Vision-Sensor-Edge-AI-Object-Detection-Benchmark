"""Compare two event streams (e.g. adaptive pipeline vs full-rate reference).

Track ids differ between runs, so events are matched by class and time only:
an event of the candidate matches an unmatched reference event of the same
type and class within `tolerance_frames`. This measures agreement with the
reference pipeline, not accuracy against ground truth.
"""
from __future__ import annotations


def event_agreement(reference: list[dict], candidate: list[dict], tolerance_frames: int = 5,
                    event_type: str = "appeared") -> dict:
    ref = sorted((e for e in reference if e["type"] == event_type), key=lambda e: e["frame"])
    cand = sorted((e for e in candidate if e["type"] == event_type), key=lambda e: e["frame"])
    used: set[int] = set()
    matched = 0
    for c in cand:
        best, best_gap = None, None
        for i, r in enumerate(ref):
            if i in used or r["class"] != c["class"]:
                continue
            gap = abs(r["frame"] - c["frame"])
            if gap <= tolerance_frames and (best_gap is None or gap < best_gap):
                best, best_gap = i, gap
        if best is not None:
            used.add(best)
            matched += 1
    precision = matched / len(cand) if cand else (1.0 if not ref else 0.0)
    recall = matched / len(ref) if ref else (1.0 if not cand else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"reference_events": len(ref), "candidate_events": len(cand), "matched": matched,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
