"""holabot_station.vision

Detector abstraction.

The private project uses YOLO (Ultralytics) with different deployment formats (pt/onnx/engine).
For the open-source baseline we implement:
- NoopDetector: always returns no detections (keeps pipeline runnable without weights)
- UltralyticsYoloDetector: loads a .pt model via ultralytics.YOLO and runs predict()

You can extend this module later with ONNX Runtime or TensorRT bindings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls_name: str

    @property
    def bbox(self):
        return (self.x1, self.y1, self.x2, self.y2)


class Detector:
    def detect(self, frame_bgr) -> List[Detection]:
        raise NotImplementedError


class NoopDetector(Detector):
    def detect(self, frame_bgr) -> List[Detection]:
        return []


class UltralyticsYoloDetector(Detector):
    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        allow_classes: Optional[List[str]] = None,
    ):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError("Ultralytics is not installed. Install with: pip install ultralytics") from e

        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Set model.path in config or place a file there."
            )

        self.model = YOLO(str(p))
        self.device = device
        self.conf = float(conf_threshold)
        self.iou = float(iou_threshold)
        self.allow = [c.lower() for c in (allow_classes or [])]

    def detect(self, frame_bgr) -> List[Detection]:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        r0 = results[0]
        names = getattr(r0, "names", {})
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return []

        xyxy = boxes.xyxy
        confs = boxes.conf
        clss = boxes.cls
        try:
            xyxy = xyxy.cpu().numpy()
            confs = confs.cpu().numpy()
            clss = clss.cpu().numpy()
        except Exception:
            pass

        dets: List[Detection] = []
        for (x1, y1, x2, y2), cf, cl in zip(xyxy, confs, clss):
            cls_id = int(cl)
            cls_name = str(names.get(cls_id, str(cls_id)))
            if self.allow and cls_name.lower() not in self.allow:
                continue
            dets.append(
                Detection(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    conf=float(cf),
                    cls_name=cls_name,
                )
            )
        return dets


def build_detector_from_config(cfg: Dict[str, Any]) -> Detector:
    model = cfg.get("model", {})
    if not isinstance(model, dict):
        return NoopDetector()

    mtype = str(model.get("type", "torch_yolo")).lower().strip()
    path = str(model.get("path", "model.pt"))
    device = str(model.get("device", "cpu"))
    conf = float(model.get("conf_threshold", 0.25))
    iou = float(model.get("iou_threshold", 0.45))

    allow = model.get("allow_classes", [])
    allow_classes = [str(x) for x in allow] if isinstance(allow, list) else []

    if mtype in ("torch_yolo", "ultralytics", "yolo"):
        return UltralyticsYoloDetector(
            model_path=path,
            device=device,
            conf_threshold=conf,
            iou_threshold=iou,
            allow_classes=allow_classes,
        )

    return NoopDetector()
