"""Model export and optimization: ONNX FP32 -> FP16 / INT8 (post-training static quantization)."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np


def letterbox(image: np.ndarray, size: int, color: int = 114) -> np.ndarray:
    """Resize keeping aspect ratio and pad to size x size (BGR uint8 in, BGR uint8 out)."""
    import cv2

    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def to_model_input(image_bgr: np.ndarray, size: int) -> np.ndarray:
    """BGR image -> float32 NCHW RGB in [0, 1], as the exported model expects."""
    img = letterbox(image_bgr, size)[:, :, ::-1]
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def export_onnx_fp32(weights_pt: str | Path, imgsz: int, out_path: str | Path, opset: int = 17) -> Path:
    from ultralytics import YOLO

    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exported = YOLO(str(weights_pt)).export(format="onnx", imgsz=imgsz, simplify=True,
                                            dynamic=False, half=False, device="cpu", opset=opset)
    shutil.move(str(exported), out_path)  # Ultralytics writes next to the weights; move per size
    return out_path


def convert_fp16(fp32_path: str | Path, out_path: str | Path) -> Path:
    """Convert weights and internal tensors to FP16, keeping float32 inputs/outputs."""
    import onnx
    # ONNX Runtime's own converter handles Resize/Upsample correctly; the generic
    # onnxconverter-common one produced graphs that ORT refuses to load for YOLO.
    from onnxruntime.transformers.float16 import convert_float_to_float16

    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    model = onnx.load(str(fp32_path))
    model16 = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model16, str(out_path))
    return out_path


def head_node_names(onnx_path: str | Path) -> list[str]:
    """Names of nodes in the last `/model.N/` block (the detection head) of an Ultralytics ONNX graph."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    pattern = re.compile(r"^/model\.(\d+)/")
    indices = [int(m.group(1)) for n in model.graph.node if (m := pattern.match(n.name))]
    if not indices:
        return []
    prefix = f"/model.{max(indices)}/"
    return [n.name for n in model.graph.node if n.name.startswith(prefix)]


def _calibration_reader(image_paths: list[Path], input_name: str, imgsz: int):
    from onnxruntime.quantization import CalibrationDataReader

    class Reader(CalibrationDataReader):
        def __init__(self):
            self._iter = iter(image_paths)

        def get_next(self):
            import cv2

            for path in self._iter:
                img = cv2.imread(str(path))
                if img is not None:
                    return {input_name: to_model_input(img, imgsz)}
            return None

    return Reader()


def quantize_int8(fp32_path: str | Path, out_path: str | Path, calibration_dir: str | Path,
                  imgsz: int, exclude_head: bool = False, max_images: int = 500) -> Path:
    """Static post-training INT8 quantization (QDQ, per-channel weights, MinMax calibration).

    exclude_head=True keeps the detection head in float ("mixed" variant); box
    regression is usually the most sensitive part of a detector.
    """
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    images = sorted(Path(calibration_dir).glob("*.jpg"))[:max_images]
    if not images:
        raise FileNotFoundError(f"No calibration images (*.jpg) in {calibration_dir}")

    fp32_path = Path(fp32_path)
    pre_path = out_path.with_name(out_path.stem + "_pre.onnx")
    try:
        quant_pre_process(str(fp32_path), str(pre_path), skip_symbolic_shape=True)
        source = pre_path
    except Exception:
        source = fp32_path

    input_name = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    exclude = head_node_names(source) if exclude_head else []
    quantize_static(
        str(source), str(out_path), _calibration_reader(images, input_name, imgsz),
        quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        per_channel=True, calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=exclude,
    )
    pre_path.unlink(missing_ok=True)
    return out_path


def file_size_mb(path: str | Path) -> float:
    return round(Path(path).stat().st_size / (1024 * 1024), 3)
