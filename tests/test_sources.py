import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from smart_vision.sensor.sources import ImageFolderSource, VideoSource


def test_image_folder_source(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i:03d}.jpg"), np.full((16, 16, 3), i * 40, np.uint8))
    frames = list(ImageFolderSource(tmp_path, fps=2.0))
    assert [f.index for f in frames] == [0, 1, 2]
    assert frames[1].timestamp_s == 0.5
    assert frames[0].image.shape == (16, 16, 3)


def test_video_source_stride(tmp_path):
    path = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
    if not writer.isOpened():
        pytest.skip("MJPG writer unavailable")
    for i in range(10):
        writer.write(np.full((32, 32, 3), i * 20, np.uint8))
    writer.release()
    frames = list(VideoSource(path, stride=2))
    assert [f.index for f in frames] == [0, 2, 4, 6, 8]


def test_empty_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ImageFolderSource(tmp_path)
