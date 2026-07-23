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

from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class PrimaryTracked:
    track_id: Optional[int]          # Ultralytics tracker id (bisa None)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    conf: float
    cls_name: str


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
        use_track: bool = True,
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
        self.use_track = bool(use_track)

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

    def detect_primary(self, frame_bgr) -> Optional[PrimaryTracked]:
        # 1) Run YOLO in track-mode (production-like) OR predict-mode
        if self.use_track:
            results = self.model.track(
                source=frame_bgr,
                persist=True,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
        else:
            results = self.model.predict(
                source=frame_bgr,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )

        if not results:
            return None

        r0 = results[0]
        names = getattr(r0, "names", {})
        boxes = getattr(r0, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy
        confs = boxes.conf
        clss = boxes.cls
        ids = getattr(boxes, "id", None)  # <- hanya ada/terisi saat track-mode

        try:
            xyxy = xyxy.cpu().numpy()
            confs = confs.cpu().numpy()
            clss = clss.cpu().numpy()
            if ids is not None:
                ids = ids.cpu().numpy()
        except Exception:
            pass

        best = None  # (area, track_id, bbox, conf, cls_name)
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            cls_id = int(clss[i])
            cls_name = str(names.get(cls_id, str(cls_id)))
            if self.allow and cls_name.lower() not in self.allow:
                continue

            area = float((x2 - x1) * (y2 - y1))
            tid = None
            if ids is not None:
                try:
                    tid = int(ids[i])
                except Exception:
                    tid = None

            bbox = (int(x1), int(y1), int(x2), int(y2))
            item = (area, tid, bbox, float(confs[i]), cls_name)
            if best is None or item[0] > best[0]:
                best = item

        if best is None:
            return None

        _area, tid, bbox, conf, cls_name = best
        return PrimaryTracked(track_id=tid, bbox=bbox, conf=conf, cls_name=cls_name)


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

    use_track = bool(model.get("use_track", True))

    if mtype in ("torch_yolo", "ultralytics", "yolo"):
        return UltralyticsYoloDetector(
            model_path=path,
            device=device,
            conf_threshold=conf,
            iou_threshold=iou,
            allow_classes=allow_classes,
            use_track=use_track,
        )

    return NoopDetector()
