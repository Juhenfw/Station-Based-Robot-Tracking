"""holabot_station.db

Open-source safe event storage.

In the original (private) system, events were written to a company MySQL database.
For the public repository, we provide local sinks that work everywhere:
- none: discard events
- jsonl: append JSON lines to a file
- sqlite: store events in a local SQLite database

Event schema (recommended):
{
  "ts": <float unix timestamp>,
  "camera": <str>,
  "track_id": <int>,
  "station": <str|None>,
  "type": "enter"|"exit"|..., 
  "meta": <dict>
}
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class EventSink:
    def write_event(self, event: Dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NoneSink(EventSink):
    def write_event(self, event: Dict[str, Any]) -> None:
        return


@dataclass
class JsonlSink(EventSink):
    path: Path

    def __post_init__(self) -> None:
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
        ts = float(event.get("ts", time.time()))
        camera = str(event.get("camera", ""))
        track_id = int(event.get("track_id", -1))
        station = event.get("station")
        station_str: Optional[str] = str(station) if station is not None else None
        ev_type = str(event.get("type", ""))
        meta = event.get("meta", {})

        self.conn.execute(
            "INSERT INTO events (ts, camera, track_id, station, event_type, meta) VALUES (?, ?, ?, ?, ?, ?)",
            (ts, camera, track_id, station_str, ev_type, json.dumps(meta, ensure_ascii=False)),
        )
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


def build_sink(storage_cfg: Any) -> EventSink:
    """Factory for sinks.

    storage_cfg is expected to be a dict-like object from YAML.
    """

    if not isinstance(storage_cfg, dict):
        return NoneSink()

    t = str(storage_cfg.get("type", "none")).lower().strip()
    if t == "none":
        return NoneSink()
    if t == "jsonl":
        return JsonlSink(Path(str(storage_cfg.get("jsonl_path", "data/events.jsonl"))))
    if t == "sqlite":
        return SqliteSink(Path(str(storage_cfg.get("sqlite_path", "data/events.sqlite"))))

    return NoneSink()
