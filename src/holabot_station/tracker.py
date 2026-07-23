"""holabot_station.tracker

Reusable tracking + checkpoint event logic.

This module intentionally stays lightweight and dependency-free (standard library only).
It provides:
- A simple, deterministic tracker that assigns IDs based on nearest-centroid matching.
- Checkpoint (station) logic that emits enter/exit events when tracks cross rectangular areas.

For a production deployment you might replace this with ByteTrack/DeepSORT, but this is a
clean open-source baseline that matches the original project shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class Detection:
    bbox: BBox
    conf: float
    cls_name: str = "object"


@dataclass
class Track:
    track_id: int
    bbox: BBox
    conf: float
    cls_name: str
    last_seen_ts: float


def centroid(b: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def in_rect(pt: Tuple[float, float], rect: Iterable[int]) -> bool:
    x, y = pt
    x1, y1, x2, y2 = list(rect)
    return x1 <= x <= x2 and y1 <= y <= y2


class SimpleCentroidTracker:
    """A minimal tracker.

    - Maintains track IDs over time.
    - Matches detections to existing tracks by nearest centroid (greedy).
    - Drops tracks when they haven't been seen for max_missed_seconds.

    Notes:
    - This is intentionally simple to keep the open-source repo easy to run and extend.
    - For heavy occlusion / crowded scenes, consider plugging in a stronger tracker.
    """

    def __init__(self, *, max_missed_seconds: float = 2.0, match_threshold_px: float = 90.0):
        self.max_missed_seconds = float(max_missed_seconds)
        self.match_threshold_sq = float(match_threshold_px) ** 2
        self._next_id = 1
        self._tracks: Dict[int, Track] = {}

    def update(self, detections: List[Detection], *, ts: float) -> List[Track]:
        det_centroids = [(centroid(d.bbox), d) for d in detections]
        used = set()

        # Drop stale tracks + try to match active ones
        for tid, tr in list(self._tracks.items()):
            if ts - tr.last_seen_ts > self.max_missed_seconds:
                del self._tracks[tid]
                continue

            best_j: Optional[int] = None
            best_dist = 1e18
            tcx, tcy = centroid(tr.bbox)

            for j, (cpt, d) in enumerate(det_centroids):
                if j in used:
                    continue
                dx = cpt[0] - tcx
                dy = cpt[1] - tcy
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j is not None and best_dist <= self.match_threshold_sq:
                used.add(best_j)
                d = det_centroids[best_j][1]
                self._tracks[tid] = Track(
                    track_id=tid,
                    bbox=d.bbox,
                    conf=d.conf,
                    cls_name=d.cls_name,
                    last_seen_ts=ts,
                )

        # Create new tracks for unmatched detections
        for j, (_, d) in enumerate(det_centroids):
            if j in used:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = Track(
                track_id=tid,
                bbox=d.bbox,
                conf=d.conf,
                cls_name=d.cls_name,
                last_seen_ts=ts,
            )

        return list(self._tracks.values())


class CheckpointEventLogic:
    """Emits enter/exit events for rectangular checkpoints.

    - checkpoints: mapping station_name -> [x1,y1,x2,y2] in processed-frame coordinates.
    - min_dwell_seconds: require a track to stay inside the station for this long before emitting 'enter'.

    Emits events:
      {"ts": ts, "track_id": id, "station": station, "type": "enter"|"exit"}

    Behavior:
    - If a track disappears while inside a station -> emits an exit.
    - If a track switches station -> emits exit(old) then (after dwell) enter(new).
    """

    def __init__(self, *, checkpoints: Dict[str, List[int]], min_dwell_seconds: float = 0.3):
        self.checkpoints = checkpoints
        self.min_dwell_seconds = float(min_dwell_seconds)
        # track_id -> (station, enter_ts). enter_ts becomes -1 after we emitted enter.
        self._inside: Dict[int, Tuple[str, float]] = {}

    def _station_for_track(self, tr: Track) -> Optional[str]:
        c = centroid(tr.bbox)
        for name, rect in self.checkpoints.items():
            if in_rect(c, rect):
                return name
        return None

    def evaluate(self, tracks: List[Track], *, ts: float) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        active = {t.track_id for t in tracks}
        for tid, (st, _enter_ts) in list(self._inside.items()):
            if tid not in active:
                events.append({"ts": ts, "track_id": tid, "station": st, "type": "exit"})
                del self._inside[tid]

        for tr in tracks:
            cur_station = self._station_for_track(tr)

            if cur_station is None:
                if tr.track_id in self._inside:
                    prev_station, _ = self._inside[tr.track_id]
                    events.append({"ts": ts, "track_id": tr.track_id, "station": prev_station, "type": "exit"})
                    del self._inside[tr.track_id]
                continue

            if tr.track_id not in self._inside:
                self._inside[tr.track_id] = (cur_station, ts)
                continue

            prev_station, enter_ts = self._inside[tr.track_id]

            if prev_station != cur_station:
                events.append({"ts": ts, "track_id": tr.track_id, "station": prev_station, "type": "exit"})
                self._inside[tr.track_id] = (cur_station, ts)
                continue

            # Same station: emit enter once after dwell
            if enter_ts >= 0 and (ts - enter_ts) >= self.min_dwell_seconds:
                events.append({"ts": ts, "track_id": tr.track_id, "station": cur_station, "type": "enter"})
                self._inside[tr.track_id] = (cur_station, -1.0)

        return events
