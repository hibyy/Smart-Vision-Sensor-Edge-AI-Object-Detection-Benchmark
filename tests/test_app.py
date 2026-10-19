"""Dashboard test using Streamlit's headless AppTest. Skipped when Ultralytics or weights are missing."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("ultralytics")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "dashboard.py")
needs_weights = pytest.mark.skipif(not (ROOT / "models" / "yolo11n.pt").exists(),
                                   reason="run scripts/02_download_weights.py first")


@needs_weights
def test_dashboard_renders():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value == "Smart Vision Sensor"


@needs_weights
def test_dashboard_runs_on_sample_images():
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.sidebar.radio[0].set_value("Sample images")
    at.sidebar.button[0].click()
    at.run()
    assert not at.exception
    assert any("Done:" in s.value for s in at.success)
