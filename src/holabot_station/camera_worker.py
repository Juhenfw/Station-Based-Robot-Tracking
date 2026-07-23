"""holabot_station.camera_worker

One worker process per camera.

This is designed to be spawned by:
  python -m holabot_station.run_station --config configs/config.yaml

But you can run a single camera directly:
  python -m holabot_station.camera_worker --config configs/config.yaml --camera cam1

Open-source safety goals:
- No RTSP/DB credentials are hardcoded; everything comes from config.
- Default event sink is local (sqlite/jsonl/none).
- If Ultralytics isn't installed or model isn't found, the worker can still run in "no-detect" mode
  (shows video + checkpoints) so contributors can validate wiring without private weights.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from holabot_station.db import build_sink
from holabot_station.vision import build_detector_from_config
from holabot_station.tracker import Detection as TrkDetection
from holabot_station.tracker import SimpleCentroidTracker, CheckpointEventLogic

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency PyYAML. Install with: pip install pyyaml") from e

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency opencv-python. Install with: pip install opencv-python") from e


# -------------------------
# Logging (simple, local)
# -------------------------

def _log(level: str, msg: str, level_cfg: str = "INFO") -> None:
    levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
    if levels.get(level, 20) >= levels.get(level_cfg, 20):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {level:<7} {msg}", flush=True)


# -------------------------
# Config loading
# -------------------------

def _load_cfg(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Copy configs/config.example.yaml to configs/config.yaml"
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/dict")
    return data


def _get_log_level(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("app", {}).get("log_level", "INFO")).upper()


def _pick_camera_cfg(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    cams = cfg.get("cameras", [])
    if not isinstance(cams, list):
        raise ValueError("config.cameras must be a list")
    for cam in cams:
        if isinstance(cam, dict) and str(cam.get("name")) == name:
            return cam
    available = [str(c.get("name")) for c in cams if isinstance(c, dict) and c.get("name")]
    raise ValueError(f"Camera '{name}' not found in config. Available: {available}")


# -------------------------
# Event sink (open-source safe)
# -------------------------

class EventSink:
    def write_event(self, event: Dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NoneSink(EventSink):
    def write_event(self, event: Dict[str, Any]) -> None:
        return


class JsonlSink(EventSink):
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", encoding="utf-8")

    def write_event(self, event: Dict[str, Any]) -> None:
        self._f.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


class SqliteSink(EventSink):
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts REAL NOT NULL,
              camera TEXT NOT NULL,
              track_id INTEGER NOT NULL,
              station TEXT,
              event_type TEXT NOT NULL,
              meta TEXT
            )
            """
        )
        self.conn.commit()

    def write_event(self, event: Dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO events (ts, camera, track_id, station, event_type, meta) VALUES (?, ?, ?, ?, ?, ?)",
            (
                float(event.get("ts", time.time())),
                str(event.get("camera", "")),
                int(event.get("track_id", -1)),
                (str(event.get("station")) if event.get("station") is not None else None),
                str(event.get("type", "")),
                json.dumps(event.get("meta", {}), ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


def _build_sink(cfg: Dict[str, Any]) -> EventSink:
    storage = cfg.get("storage", {})
    if not isinstance(storage, dict):
        return NoneSink()

    t = str(storage.get("type", "none")).lower()
    if t == "none":
        return NoneSink()
    if t == "jsonl":
        return JsonlSink(Path(str(storage.get("jsonl_path", "data/events.jsonl"))))
    if t == "sqlite":
        return SqliteSink(Path(str(storage.get("sqlite_path", "data/events.sqlite"))))

    return NoneSink()


# -------------------------
# Detection
# -------------------------

@dataclass
class Detection:
    xyxy: Tuple[int, int, int, int]
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
        model_path: str,
        device: str,
        conf: float,
        iou: float,
        allow_classes: Optional[List[str]],
        log_level: str,
    ):
        self.log_level = log_level
        self.allow_classes = [c.lower() for c in (allow_classes or [])]
        self.conf = float(conf)
        self.iou = float(iou)

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError("Ultralytics not installed. pip install ultralytics") from e

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. Put a model there or set model.path in config."
            )

        _log("INFO", f"Loading YOLO model: {model_path} (device={device})", self.log_level)
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, frame_bgr) -> List[Detection]:
        # Ultralytics expects BGR images as numpy arrays.
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
        dets: List[Detection] = []

        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return []

        # Convert to CPU numpy for safety
        xyxy = boxes.xyxy
        confs = boxes.conf
        clss = boxes.cls
        try:
            xyxy = xyxy.cpu().numpy()
            confs = confs.cpu().numpy()
            clss = clss.cpu().numpy()
        except Exception:
            pass

        for (x1, y1, x2, y2), cf, cl in zip(xyxy, confs, clss):
            cls_id = int(cl)
            cls_name = str(names.get(cls_id, str(cls_id)))
            if self.allow_classes and cls_name.lower() not in self.allow_classes:
                continue
            dets.append(
                Detection(
                    xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    conf=float(cf),
                    cls_name=cls_name,
                )
            )

        return dets


def _build_detector(cfg: Dict[str, Any], log_level: str) -> Detector:
    model = cfg.get("model", {})
    if not isinstance(model, dict):
        _log("WARNING", "Config model section missing/invalid; running with NoopDetector", log_level)
        return NoopDetector()

    mtype = str(model.get("type", "torch_yolo")).lower()
    path = str(model.get("path", "model.pt"))
    device = str(model.get("device", "cpu"))
    conf = float(model.get("conf_threshold", 0.25))
    iou = float(model.get("iou_threshold", 0.45))
    allow = model.get("allow_classes", [])
    allow_classes = [str(x) for x in allow] if isinstance(allow, list) else []

    if mtype in ("torch_yolo", "ultralytics", "yolo"):
        try:
            return UltralyticsYoloDetector(path, device, conf, iou, allow_classes, log_level)
        except Exception as e:
            _log("WARNING", f"Detector disabled: {e}. Running in no-detect mode.", log_level)
            return NoopDetector()

    _log("WARNING", f"Unknown model.type '{mtype}'. Running in no-detect mode.", log_level)
    return NoopDetector()


# -------------------------
# Tracking + checkpoint logic (simple, deterministic)
# -------------------------

@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    cls_name: str
    conf: float
    last_seen_ts: float
    station: Optional[str] = None


def _centroid(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _in_rect(pt: Tuple[float, float], rect: List[int]) -> bool:
    x, y = pt
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


class SimpleTracker:
    def __init__(self, max_missed_seconds: float):
        self.max_missed_seconds = float(max_missed_seconds)
        self._next_id = 1
        self._tracks: Dict[int, Track] = {}

    def update(self, dets: List[Detection], ts: float) -> List[Track]:
        # Very simple assignment by nearest centroid (greedy). Good enough for an open-source template.
        used = set()
        det_centroids = [(_centroid(d.xyxy), d) for d in dets]

        # Try to match existing tracks
        for tid, tr in list(self._tracks.items()):
            if ts - tr.last_seen_ts > self.max_missed_seconds:
                del self._tracks[tid]
                continue

            best_j = None
            best_dist = 1e18
            cx, cy = _centroid(tr.bbox)
            for j, (cpt, d) in enumerate(det_centroids):
                if j in used:
                    continue
                dx = cpt[0] - cx
                dy = cpt[1] - cy
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            # Threshold prevents random jumps; tuned for 640x480-ish.
            if best_j is not None and best_dist < (90.0 * 90.0):
                used.add(best_j)
                d = det_centroids[best_j][1]
                self._tracks[tid] = Track(
                    track_id=tid,
                    bbox=d.xyxy,
                    cls_name=d.cls_name,
                    conf=d.conf,
                    last_seen_ts=ts,
                    station=tr.station,
                )

        # Create new tracks for unmatched detections
        for j, (_, d) in enumerate(det_centroids):
            if j in used:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = Track(
                track_id=tid,
                bbox=d.xyxy,
                cls_name=d.cls_name,
                conf=d.conf,
                last_seen_ts=ts,
                station=None,
            )

        return list(self._tracks.values())


class CheckpointLogic:
    def __init__(self, checkpoints: Dict[str, List[int]], min_dwell_seconds: float):
        self.checkpoints = checkpoints
        self.min_dwell_seconds = float(min_dwell_seconds)
        # track_id -> (station, enter_ts)
        self._inside: Dict[int, Tuple[str, float]] = {}

    def evaluate(self, tracks: List[Track], ts: float) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        active_ids = {t.track_id for t in tracks}
        # Handle exits for tracks no longer present
        for tid, (st, enter_ts) in list(self._inside.items()):
            if tid not in active_ids:
                events.append({"ts": ts, "track_id": tid, "station": st, "type": "exit"})
                del self._inside[tid]

        for t in tracks:
            c = _centroid(t.bbox)
            in_station: Optional[str] = None
            for st_name, rect in self.checkpoints.items():
                if _in_rect(c, rect):
                    in_station = st_name
                    break

            if in_station is None:
                if t.track_id in self._inside:
                    st, _ = self._inside[t.track_id]
                    events.append({"ts": ts, "track_id": t.track_id, "station": st, "type": "exit"})
                    del self._inside[t.track_id]
                continue

            if t.track_id not in self._inside:
                # Enter (tentative)
                self._inside[t.track_id] = (in_station, ts)
                continue

            prev_station, enter_ts = self._inside[t.track_id]
            if prev_station != in_station:
                # Switch station -> treat as exit + new enter
                events.append({"ts": ts, "track_id": t.track_id, "station": prev_station, "type": "exit"})
                self._inside[t.track_id] = (in_station, ts)
                continue

            # Confirm entry after dwell
            if ts - enter_ts >= self.min_dwell_seconds:
                # Emit a single "enter" event when dwell threshold is passed.
                # To keep it simple, we mark enter_ts as -inf after firing.
                if enter_ts >= 0:
                    events.append({"ts": ts, "track_id": t.track_id, "station": in_station, "type": "enter"})
                    self._inside[t.track_id] = (in_station, -1.0)

        return events


# -------------------------
# Video capture + processing
# -------------------------

@dataclass
class FramePacket:
    ts: float
    frame: Any


def _open_capture(source_cfg: Dict[str, Any], log_level: str) -> cv2.VideoCapture:
    stype = str(source_cfg.get("type", "webcam")).lower()

    if stype == "webcam":
        idx = int(source_cfg.get("webcam_index", 0))
        cap = cv2.VideoCapture(idx)
        return cap

    if stype == "file":
        path = str(source_cfg.get("file_path", ""))
        cap = cv2.VideoCapture(path)
        return cap

    if stype == "rtsp":
        url = str(source_cfg.get("rtsp_url", ""))
        if not url:
            raise ValueError("source.rtsp_url is empty")
        cap = cv2.VideoCapture(url)
        return cap

    _log("WARNING", f"Unknown source.type '{stype}', falling back to webcam(0)", log_level)
    return cv2.VideoCapture(0)


def _resize_if_needed(frame, w: int, h: int):
    if frame is None:
        return frame
    if int(frame.shape[1]) == int(w) and int(frame.shape[0]) == int(h):
        return frame
    return cv2.resize(frame, (int(w), int(h)))


def _draw_checkpoints(frame, checkpoints: Dict[str, List[int]]):
    for name, rect in checkpoints.items():
        x1, y1, x2, y2 = rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 200, 50), 2)
        cv2.putText(
            frame,
            str(name),
            (x1 + 5, max(15, y1 + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (50, 200, 50),
            1,
            cv2.LINE_AA,
        )


def _draw_tracks(frame, tracks: List[Track]):
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 160, 255), 2)
        label = f"ID {t.track_id} {t.cls_name} {t.conf:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 160, 255),
            1,
            cv2.LINE_AA,
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Camera worker")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--camera", required=True, help="Camera name from config.cameras[].name")
    args = parser.parse_args(argv)

    cfg = _load_cfg(Path(args.config).resolve())
    log_level = _get_log_level(cfg)
    cam_cfg = _pick_camera_cfg(cfg, args.camera)

    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime", {}), dict) else {}
    tracking_cfg = cfg.get("tracking", {}) if isinstance(cfg.get("tracking", {}), dict) else {}

    display = bool(runtime_cfg.get("display", True))
    draw_checkpoints = bool(runtime_cfg.get("draw_checkpoints", True))
    qsize = int(runtime_cfg.get("frame_queue_size", 5))

    w = int(tracking_cfg.get("process_width", 640))
    h = int(tracking_cfg.get("process_height", 480))
    max_missed = float(tracking_cfg.get("max_missed_seconds", 2.0))
    min_dwell = float(tracking_cfg.get("min_dwell_seconds", 0.3))

    checkpoints_raw = cam_cfg.get("checkpoints", {})
    if not isinstance(checkpoints_raw, dict):
        checkpoints_raw = {}
    checkpoints: Dict[str, List[int]] = {}
    for k, v in checkpoints_raw.items():
        if isinstance(v, list) and len(v) == 4:
            checkpoints[str(k)] = [int(x) for x in v]

    sink = build_sink(cfg.get("storage", {}))
    detector = build_detector_from_config(cfg)
    tracker = SimpleCentroidTracker(max_missed_seconds=max_missed)
    cp_logic = CheckpointEventLogic(checkpoints=checkpoints, min_dwell_seconds=min_dwell)

    source_cfg = cam_cfg.get("source", {}) if isinstance(cam_cfg.get("source", {}), dict) else {}

    stop = False

    def _handle_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_stop)

    frame_q: "queue.Queue[FramePacket]" = queue.Queue(maxsize=qsize)

    def receive_loop() -> None:
        nonlocal stop
        backoff = 1.0
        cap: Optional[cv2.VideoCapture] = None

        while not stop:
            try:
                if cap is None or not cap.isOpened():
                    cap = _open_capture(source_cfg, log_level)
                    if not cap.isOpened():
                        _log("WARNING", f"[{args.camera}] Cannot open source. Retrying...", log_level)
                        time.sleep(backoff)
                        backoff = min(10.0, backoff * 1.5)
                        continue
                    backoff = 1.0

                ok, frame = cap.read()
                if not ok or frame is None:
                    _log("WARNING", f"[{args.camera}] Frame read failed. Reconnecting...", log_level)
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                    time.sleep(0.5)
                    continue

                frame = _resize_if_needed(frame, w, h)
                pkt = FramePacket(ts=time.time(), frame=frame)

                # Drop frames if queue is full (prefer realtime)
                try:
                    frame_q.put(pkt, timeout=0.01)
                except queue.Full:
                    try:
                        _ = frame_q.get_nowait()
                    except Exception:
                        pass
                    try:
                        frame_q.put(pkt, timeout=0.01)
                    except Exception:
                        pass

            except Exception as e:
                _log("ERROR", f"[{args.camera}] Receive loop error: {e}", log_level)
                time.sleep(1.0)

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    recv_thread = threading.Thread(target=receive_loop, name=f"Receive-{args.camera}", daemon=True)
    recv_thread.start()

    win_name = f"{args.camera}"
    last_fps_t = time.time()
    frames = 0
    fps = 0.0

    _log("INFO", f"[{args.camera}] Worker started. display={display}", log_level)

    try:
        while not stop:
            try:
                pkt = frame_q.get(timeout=0.5)
            except queue.Empty:
                continue

            ts = pkt.ts
            frame = pkt.frame

            dets = detector.detect(frame)
            trk_dets = [TrkDetection(bbox=d.bbox, conf=d.conf, cls_name=d.cls_name) for d in dets]
            tracks = tracker.update(trk_dets, ts=ts)
            events = cp_logic.evaluate(tracks, ts=ts)

            for ev in events:
                ev_out = {
                    "ts": ev["ts"],
                    "camera": args.camera,
                    "track_id": ev["track_id"],
                    "station": ev.get("station"),
                    "type": ev["type"],
                    "meta": {"source_type": str(source_cfg.get("type", ""))},
                }
                sink.write_event(ev_out)
                _log("INFO", f"[{args.camera}] event: {ev_out}", log_level)

            # Draw
            frames += 1
            now = time.time()
            if now - last_fps_t >= 1.0:
                fps = frames / (now - last_fps_t)
                frames = 0
                last_fps_t = now

            if draw_checkpoints:
                _draw_checkpoints(frame, checkpoints)
            _draw_tracks(frame, tracks)

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if display:
                cv2.imshow(win_name, frame)
                # Press 'q' to stop this worker
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    stop = True

    finally:
        stop = True
        try:
            sink.close()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        _log("INFO", f"[{args.camera}] Worker stopped.", log_level)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())