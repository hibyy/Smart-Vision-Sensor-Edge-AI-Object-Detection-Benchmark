from smart_vision.env import collect_env


def test_collect_env_has_core_fields():
    info = collect_env()
    for key in ("os", "python", "cpu", "packages", "timestamp_utc"):
        assert key in info
