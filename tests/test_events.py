from smart_vision.events.engine import EventEngine
from smart_vision.types import Detection


def det(name="cup", conf=0.9, tid=None, cid=41):
    return Detection(cid, name, conf, (0, 0, 10, 10), tid)


def test_counts_any_class_and_confidence_filter():
    eng = EventEngine(min_confidence=0.5)
    out = eng.update(0, 0.0, [det("cup"), det("cup"), det("dog", cid=16), det("cat", conf=0.2, cid=15)])
    assert out["counts"] == {"cup": 2, "dog": 1}
    assert out["events"] == []


def test_class_filter():
    eng = EventEngine(classes={"dog"})
    out = eng.update(0, 0.0, [det("cup"), det("dog", cid=16)])
    assert out["counts"] == {"dog": 1}


def test_appeared_only_after_confirmation():
    eng = EventEngine(confirm_frames=3)
    for i in range(2):
        assert eng.update(i, i * 0.1, [det(tid=7)])["events"] == []
    out = eng.update(2, 0.2, [det(tid=7)])
    assert out["events"] == [{"type": "appeared", "class": "cup", "track_id": 7, "frame": 2}]
    assert eng.update(3, 0.3, [det(tid=7)])["events"] == []  # no duplicate


def test_disappeared_after_lost_frames():
    eng = EventEngine(confirm_frames=1, lost_frames=2)
    eng.update(0, 0.0, [det(tid=1)])
    assert eng.update(1, 0.1, [])["events"] == []
    assert eng.update(2, 0.2, [])["events"] == []
    out = eng.update(3, 0.3, [])
    assert out["events"] == [{"type": "disappeared", "class": "cup", "track_id": 1, "frame": 3}]


def test_flicker_never_confirmed_emits_nothing():
    eng = EventEngine(confirm_frames=3, lost_frames=1)
    eng.update(0, 0.0, [det(tid=5)])
    for i in range(1, 5):
        assert eng.update(i, i * 0.1, [])["events"] == []
