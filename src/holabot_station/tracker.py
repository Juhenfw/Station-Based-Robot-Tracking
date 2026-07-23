"""holabot_station.tracker

Tracking / checkpoint logic.

This module contains a production-like *single-robot* checkpoint state machine that matches
common station-monitoring logic:
- Emit ENTER only after the robot stays inside a station for `sec_stop` seconds
- Emit EXIT only after an exit grace period (`exit_grace_seconds`)
- Optionally require exit station to match `last_entry_area`

The single-robot approach fits deployments where there is only one robot of interest per camera
(or you want to track the "primary" robot only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


BBox = Tuple[int, int, int, int]  # x1,y1,x2,y2


def centroid(b: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def in_rect(pt: Tuple[float, float], rect: List[int]) -> bool:
    x, y = pt
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


@dataclass
class StationEvent:
    ts: float
    station: str
    type: str  # "enter" | "exit"


class SingleRobotCheckpointSM:
    """Production-like checkpoint logic for one robot.

    Call update() once per processed frame.

    Parameters:
      - checkpoints: dict station_name -> [x1,y1,x2,y2] in processed-frame coordinates
      - sec_stop: seconds robot must remain inside a station to count as ENTER
      - exit_grace_seconds: seconds to wait after leaving a station before counting EXIT
      - require_exit_matches_last_entry: guards against false exits
      - lock_track_id: once a YOLO track_id is seen, ignore detections from other ids

    State:
      - entered_area / last_entry_area
      - stop timer per current area
      - pending exit timer
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

        self.entered_area: Optional[str] = None
        self.last_entry_area: Optional[str] = None

        self.stop_area: Optional[str] = None
        self.stop_started_ts: Optional[float] = None

        self.pending_exit_area: Optional[str] = None
        self.pending_exit_started_ts: Optional[float] = None

    def _area_for_bbox(self, bbox: BBox) -> Optional[str]:
        c = centroid(bbox)
        for name, rect in self.checkpoints.items():
            if in_rect(c, rect):
                return name
        return None

    def update(
        self,
        *,
        ts: float,
        track_id: Optional[int],
        bbox: Optional[BBox],
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        # Track-id locking (optional)
        if self.lock_track_id and track_id is not None and self.robot_locked_id is None:
            self.robot_locked_id = track_id

        if self.lock_track_id and self.robot_locked_id is not None and track_id is not None:
            if track_id != self.robot_locked_id:
                return self._flush_pending(ts, events)

        # If robot is not detected this frame
        if bbox is None:
            # if we were entered, start pending exit
            if self.entered_area is not None and self.pending_exit_started_ts is None:
                self.pending_exit_area = self.entered_area
                self.pending_exit_started_ts = ts
            return self._flush_pending(ts, events)

        area = self._area_for_bbox(bbox)

        # Start pending exit if we were entered and left the station
        if self.entered_area is not None and area != self.entered_area and self.pending_exit_started_ts is None:
            self.pending_exit_area = self.entered_area
            self.pending_exit_started_ts = ts

        # Stop/dwell logic inside any station
        if area is not None:
            if self.stop_area != area:
                self.stop_area = area
                self.stop_started_ts = ts

            if self.stop_started_ts is not None and (ts - self.stop_started_ts) >= self.sec_stop:
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

        # reset entered state
        self.entered_area = None
        self.pending_exit_started_ts = None
        self.pending_exit_area = None
        return events
