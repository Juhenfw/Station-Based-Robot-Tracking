"""holabot_station.vision

Detector abstraction.

This open-source repo supports running without private weights, while still allowing a
production-like mode using Ultralytics YOLO tracking:
- NoopDetector: always returns no detections
- UltralyticsYoloDetector:
  - detect(): list of detections (predict or track)
  - detect_primary(): single "primary" detection (largest box), optionally with track_id

Config keys used:
model:
  type: torch_yolo|ultralytics|yolo
  path: model.pt
  device: cpu|cuda|cuda:0
  conf_threshold: 0.25
  iou_threshold: 0.45
  allow_classes: ["robot"]          # optional
  allow_class_ids: [0]              # optional
  use_track: true                   # if true, uses model.track(persist=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    bbox: BBox
    conf: float
    cls_id: int
    cls_name: str


@dataclass
class PrimaryTracked:
    track_id: Optional[int]
    bbox: BBox
    conf: float
    cls_id: int
    cls_name: str


class Detector:
    def detect(self, frame_bgr) -> List[Detection]:
        raise NotImplementedError

    def detect_primary(self, frame_bgr) -> Optional[PrimaryTracked]:
        """Return a single best detection (largest box). Default: None."""
        return None


class NoopDetector(Detector):
    def detect(self, frame_bgr) -> List[Detection]:
        return []

    def detect_primary(self, frame_bgr) -> Optional[PrimaryTracked]:
        return None


class UltralyticsYoloDetector(Detector):
    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        allow_classes: Optional[List[str]] = None,
        allow_class_ids: Optional[List[int]] = None,
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
        self.device = str(device)
        self.conf = float(conf_threshold)
        self.iou = float(iou_threshold)
        self.allow_classes = [c.lower() for c in (allow_classes or [])]
        self.allow_class_ids = [int(x) for x in (allow_class_ids or [])]
        self.use_track = bool(use_track)

    def _run(self, frame_bgr):
        if self.use_track:
            return self.model.track(
                source=frame_bgr,
                persist=True,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
        return self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

    def _accept(self, cls_id: int, cls_name: str) -> bool:
        if self.allow_class_ids and cls_id not in self.allow_class_ids:
            return False
        if self.allow_classes and cls_name.lower() not in self.allow_classes:
            return False
        return True

    def detect(self, frame_bgr) -> List[Detection]:
        results = self._run(frame_bgr)
        if not results:
            return []

        r0 = results[0]
        names = getattr(r0, "names", {})
        boxes = getattr(r0, "boxes", None)
        if boxes is None or len(boxes) == 0:
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
            if not self._accept(cls_id, cls_name):
                continue
            dets.append(
                Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    conf=float(cf),
                    cls_id=cls_id,
                    cls_name=cls_name,
                )
            )
        return dets

    def detect_primary(self, frame_bgr) -> Optional[PrimaryTracked]:
        results = self._run(frame_bgr)
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
        ids = getattr(boxes, "id", None)

        try:
            xyxy = xyxy.cpu().numpy()
            confs = confs.cpu().numpy()
            clss = clss.cpu().numpy()
            if ids is not None:
                ids = ids.cpu().numpy()
        except Exception:
            pass

        best = None  # (area, track_id, bbox, conf, cls_id, cls_name)
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            cls_id = int(clss[i])
            cls_name = str(names.get(cls_id, str(cls_id)))
            if not self._accept(cls_id, cls_name):
                continue

            area = float((x2 - x1) * (y2 - y1))
            tid: Optional[int] = None
            if ids is not None:
                try:
                    tid = int(ids[i])
                except Exception:
                    tid = None

            item = (
                area,
                tid,
                (int(x1), int(y1), int(x2), int(y2)),
                float(confs[i]),
                cls_id,
                cls_name,
            )
            if best is None or item[0] > best[0]:
                best = item

        if best is None:
            return None

        _area, tid, bbox, conf, cls_id, cls_name = best
        return PrimaryTracked(track_id=tid, bbox=bbox, conf=conf, cls_id=cls_id, cls_name=cls_name)


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

    allow_ids = model.get("allow_class_ids", [])
    allow_class_ids = [int(x) for x in allow_ids] if isinstance(allow_ids, list) else []

    use_track = bool(model.get("use_track", True))

    if mtype in ("torch_yolo", "ultralytics", "yolo"):
        return UltralyticsYoloDetector(
            model_path=path,
            device=device,
            conf_threshold=conf,
            iou_threshold=iou,
            allow_classes=allow_classes,
            allow_class_ids=allow_class_ids,
            use_track=use_track,
        )

    return NoopDetector()
