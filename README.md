# Station-Based Robot Tracking (Open Source Template)

This repository is an **open-source-friendly template** for running **multi-camera robot detection + checkpoint (station) tracking**.

It is designed to match a production-style setup where:
- one **orchestrator** process starts and monitors
- multiple **camera worker** processes (one per camera)

Everything that can contain sensitive information (RTSP URLs, credentials, internal database settings) is moved into a local `config.yaml` that you **should not commit**.

## Features

- Run **N cameras in parallel** (one OS process per camera)
- Sources supported: **webcam**, **video file**, **RTSP**
- YOLO detection via **Ultralytics** (optional) with automatic fallback to **no-detect mode**
- Simple tracking (centroid-based) + **checkpoint enter/exit events**
- Open-source safe event storage: **SQLite** or **JSONL** (no MySQL required)

## Repository structure

- `configs/config.example.yaml` — example configuration (safe to commit)
- `configs/config.yaml` — your real configuration (**do not commit**)
- `src/holabot_station/run_station.py` — orchestrator (spawns workers)
- `src/holabot_station/camera_worker.py` — per-camera worker
- `src/holabot_station/db.py` — local event sinks
- `src/holabot_station/tracker.py` — reusable tracking + checkpoint logic
- `src/holabot_station/vision.py` — YOLO detector wrapper

## Quick start (Windows)

### 1) Create and activate a virtual environment

PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

### 2) Install dependencies

pip install -r requirements.txt

Recommended packages for full functionality:
- `opencv-python`
- `pyyaml`
- `ultralytics` (only needed if you want YOLO detection)

### 3) Create your config

Copy the example config:

Copy-Item .\configs\config.example.yaml .\configs\config.yaml

Edit `configs/config.yaml`:
- set `cameras[].source` to `webcam` or `file` first for testing
- optionally set `model.path` to your model file (example uses `model.pt`)

### 4) Ensure Python can import from `src/`

If you have not installed this package, add `./src` to `PYTHONPATH`:

$env:PYTHONPATH = "$PWD\src"

(You can add this to your PowerShell profile if you want.)

### 5) Run (orchestrator spawns camera workers)

python -m holabot_station.run_station --config configs/config.yaml

Stop with **Ctrl+C**.

## Running a single camera worker (debug)

$env:PYTHONPATH = "$PWD\src"
python -m holabot_station.camera_worker --config configs/config.yaml --camera cam1

Press **q** in the OpenCV window to stop the worker.

## Event output

Events are written based on `storage.type`:
- `sqlite` (default): `data/events.sqlite` with a table `events`
- `jsonl`: `data/events.jsonl` (one JSON per line)
- `none`: discard events

Example event:

{
  "ts": 1720000000.123,
  "camera": "cam1",
  "track_id": 3,
  "station": "Station_1",
  "type": "enter",
  "meta": {"source_type": "file"}
}

## Notes on models (open-source safety)

This template will **run even without a model**:
- If `ultralytics` is not installed, or `model.path` is missing, the worker enters **no-detect mode**.
- In no-detect mode the app still shows video + checkpoints so you can validate camera wiring and config.

If you want detection:
- place a YOLO `.pt` file in the repo (for example `model.pt`) or any path you choose
- set `model.path` accordingly in `configs/config.yaml`

## Troubleshooting (Windows)

- **`ModuleNotFoundError: holabot_station`**
  - Make sure: `$env:PYTHONPATH = "$PWD\src"` before running.

- **OpenCV cannot open camera / file**
  - Try switching `source.type` to `file` first and provide a valid `file_path`.

- **YOLO not running / model not found**
  - Install ultralytics: `pip install ultralytics`
  - Ensure `model.path` points to an existing file.

## License

See `LICENSE`.