"""holabot_station.camera_worker

One worker process per camera.

Designed to be spawned by:
  python -m holabot_station.run_station --config configs/config.yaml

Single-robot production-like mode:
- Uses Ultralytics YOLO tracking (model.track(persist=True)) via holabot_station.vision
- Chooses ONE primary robot per frame (largest box)
- Applies checkpoint state machine with:
  - sec_stop (dwell/stop time before ENTER)
  - pending exit grace
  - last_entry_area validation

Open-source safety:
- No credentials hardcoded; everything comes from config.yaml
- Event sinks are local (sqlite/jsonl/none)
- If ultralytics/model is missing, runs in no-detect mode (still shows video + checkpoints)
"""

from __future__ import annotations

import argparse
import os
import queue
import signal
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency PyYAML. Install with: pip install pyyaml") from e

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency opencv-python. Install with: pip install opencv-python") from e

from holabot_station.db import build_sink
from holabot_station.vision import build_detector_from_config
from holabot_station.tracker import SingleRobotCheckpointSM


def _log(level: str, msg: str, level_cfg: str = "INFO") -> None:
    levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
    if levels.get(level, 20) >= levels.get(level_cfg, 20):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {level:<7} {msg}", flush=True)


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


def _open_capture(source_cfg: Dict[str, Any], log_level: str) -> cv2.VideoCapture:
    stype = str(source_cfg.get("type", "webcam")).lower()

    if stype == "webcam":
        idx = int(source_cfg.get("webcam_index", 0))
        return cv2.VideoCapture(idx)

    if stype == "file":
        path = str(source_cfg.get("file_path", ""))
        return cv2.VideoCapture(path)

    if stype == "rtsp":
        url = str(source_cfg.get("rtsp_url", ""))
        if not url:
            raise ValueError("source.rtsp_url is empty")
        return cv2.VideoCapture(url)

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


def _draw_primary(frame, primary) -> None:
    if primary is None:
        return
    x1, y1, x2, y2 = primary.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 160, 255), 2)
    label = f"ID {primary.track_id} {primary.cls_name} {primary.conf:.2f}"
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


class FramePacket:
    def __init__(self, ts: float, frame: Any):
        self.ts = ts
        self.frame = frame


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

    # Production-like parameters
    sec_stop = float(tracking_cfg.get("sec_stop", 3.0))
    exit_grace_seconds = float(tracking_cfg.get("exit_grace_seconds", 1.0))
    require_exit_matches = bool(tracking_cfg.get("require_exit_matches_last_entry", True))
    lock_track_id = bool(tracking_cfg.get("lock_track_id", True))

    checkpoints_raw = cam_cfg.get("checkpoints", {})
    if not isinstance(checkpoints_raw, dict):
        checkpoints_raw = {}
    checkpoints: Dict[str, List[int]] = {}
    for k, v in checkpoints_raw.items():
        if isinstance(v, list) and len(v) == 4:
            checkpoints[str(k)] = [int(x) for x in v]

    sink = build_sink(cfg.get("storage", {}))
    detector = build_detector_from_config(cfg)

    sm = SingleRobotCheckpointSM(
        checkpoints=checkpoints,
        sec_stop=sec_stop,
        exit_grace_seconds=exit_grace_seconds,
        require_exit_matches_last_entry=require_exit_matches,
        lock_track_id=lock_track_id,
    )

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

            # Production single-robot: choose one primary detection (largest box)
            primary = None
            try:
                primary = detector.detect_primary(frame)  # type: ignore[attr-defined]
            except Exception:
                # Detector may not implement detect_primary (noop), that's fine.
                primary = None

            events = sm.update(
                ts=ts,
                track_id=(primary.track_id if primary is not None else None),
                bbox=(primary.bbox if primary is not None else None),
            )

            for ev in events:
                status = "In" if ev["type"] == "enter" else "Out"
                ev_out = {
                    "ts": ev["ts"],
                    "camera": args.camera,
                    "track_id": (primary.track_id if primary and primary.track_id is not None else -1),
                    "station": ev.get("station"),
                    # Keep both: machine-friendly and human-friendly
                    "event_type": ev["type"],
                    "status": status,
                    "meta": {"source_type": str(source_cfg.get("type", ""))},
                }
                sink.write_event(ev_out)
                _log("INFO", f"[{args.camera}] event: {ev_out}", log_level)

            frames += 1
            now = time.time()
            if now - last_fps_t >= 1.0:
                fps = frames / max(now - last_fps_t, 1e-9)
                frames = 0
                last_fps_t = now

            if draw_checkpoints:
                _draw_checkpoints(frame, checkpoints)
            _draw_primary(frame, primary)

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
