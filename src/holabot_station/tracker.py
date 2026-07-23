# src/holabot_station/tracker.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

BBox = Tuple[int, int, int, int]  # x1,y1,x2,y2


def _centroid(b: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _in_rect(pt: Tuple[float, float], rect: List[int]) -> bool:
    x, y = pt
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


@dataclass
class StationEvent:
    ts: float
    station: str
    type: str  # "enter" | "exit"


class SingleRobotCheckpointSM:
    """
    Production-like checkpoint state machine for *single robot* use-case.

    Features:
    - sec_stop: robot must remain inside a station for N seconds before emitting 'enter'
    - pending exit: emit 'exit' only after exit_grace_seconds
    - last_entry_area: optionally require exit station equals last entry station

    Inputs per frame:
      - ts: timestamp (seconds)
      - track_id: Ultralytics track id (optional; can be None)
      - bbox: (x1,y1,x2,y2) for the chosen robot detection, or None if robot not detected
    """

    def __init__(
        self,
        *,
        checkpoints: Dict[str, List[int]],
        sec_stop: float = 3.0,
        exit_grace_seconds: float = 1.0,
        require_exit_matches_last_entry: bool = True,
        lock_track_id: bool = True,
    ):
        self.checkpoints = checkpoints
        self.sec_stop = float(sec_stop)
        self.exit_grace_seconds = float(exit_grace_seconds)
        self.require_exit_matches_last_entry = bool(require_exit_matches_last_entry)
        self.lock_track_id = bool(lock_track_id)

        self.robot_locked_id: Optional[int] = None

        # Enter state
        self.entered_area: Optional[str] = None
        self.last_entry_area: Optional[str] = None

        # Stop detection (dwell) state
        self.stop_area: Optional[str] = None
        self.stop_started_ts: Optional[float] = None

        # Exit pending state
        self.pending_exit_area: Optional[str] = None
        self.pending_exit_started_ts: Optional[float] = None

    def _area_for_bbox(self, bbox: BBox) -> Optional[str]:
        c = _centroid(bbox)
        for name, rect in self.checkpoints.items():
            if _in_rect(c, rect):
                return name
        return None

    def update(self, *, ts: float, track_id: Optional[int], bbox: Optional[BBox]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        # Track-id locking (optional, matches your production style)
        if self.lock_track_id and track_id is not None and self.robot_locked_id is None:
            self.robot_locked_id = track_id

        if self.lock_track_id and self.robot_locked_id is not None and track_id is not None:
            if track_id != self.robot_locked_id:
                # Ignore other objects once locked
                return self._flush_pending(ts, events)

        # If robot not detected this frame: treat as potential exit if we had entered
        if bbox is None:
            if self.entered_area is not None and self.pending_exit_started_ts is None:
                self.pending_exit_area = self.entered_area
                self.pending_exit_started_ts = ts
            return self._flush_pending(ts, events)

        area = self._area_for_bbox(bbox)

        # If we were entered, and now area differs => start pending exit
        if self.entered_area is not None and area != self.entered_area and self.pending_exit_started_ts is None:
            self.pending_exit_area = self.entered_area
            self.pending_exit_started_ts = ts

        # Stop detection logic inside an area
        if area is not None:
            if self.stop_area != area:
                self.stop_area = area
                self.stop_started_ts = ts

            if self.stop_started_ts is not None and (ts - self.stop_started_ts) >= self.sec_stop:
                # Emit enter once when the robot has dwelled long enough
                if self.entered_area != area:
                    self.entered_area = area
                    self.last_entry_area = area
                    events.append({"ts": ts, "station": area, "type": "enter"})

        return self._flush_pending(ts, events)

    def _flush_pending(self, ts: float, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.pending_exit_started_ts is None or self.pending_exit_area is None:
            return events

        if (ts - self.pending_exit_started_ts) < self.exit_grace_seconds:
            return events

        if (
            self.require_exit_matches_last_entry
            and self.last_entry_area is not None
            and self.pending_exit_area != self.last_entry_area
        ):
            # Drop invalid exit
            self.pending_exit_started_ts = None
            self.pending_exit_area = None
            return events

        events.append({"ts": ts, "station": self.pending_exit_area, "type": "exit"})

        # Reset entered state after exit
        self.entered_area = None
        self.pending_exit_started_ts = None
        self.pending_exit_area = None
        return events