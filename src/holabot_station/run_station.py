"""holabot_station.run_station

Windows-first orchestrator for running N camera workers in parallel (one OS process per camera).

Run on Windows (PowerShell):
  python -m holabot_station.run_station --config configs/config.yaml

If you are not installing the package, ensure PYTHONPATH includes ./src.
This orchestrator also injects PYTHONPATH for child processes automatically.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency PyYAML. Install with: pip install pyyaml\n"
        "If you use requirements.txt/pyproject.toml, add 'pyyaml'."
    ) from e


def _now() -> float:
    return time.time()


def find_repo_root(start: Path) -> Path:
    """Best-effort: walk up until we find a folder that contains 'src'."""

    cur = start.resolve()
    for _ in range(8):
        if (cur / "src").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # Fallback: assume this file is in <repo>/src/holabot_station
    return start.resolve().parents[2]


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            "Copy configs/config.example.yaml to configs/config.yaml and edit it."
        )

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/dict.")

    if "cameras" not in data or not isinstance(data["cameras"], list) or not data["cameras"]:
        raise ValueError("Config must include a non-empty 'cameras:' list.")

    return data


def pick_log_level(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("app", {}).get("log_level", "INFO")).upper()


def log(level: str, msg: str, cfg_level: str = "INFO") -> None:
    levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
    if levels.get(level, 20) >= levels.get(cfg_level, 20):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {level:<7} {msg}", flush=True)


@dataclass
class WorkerSpec:
    name: str
    camera_index: int


@dataclass
class ProcState:
    spec: WorkerSpec
    proc: subprocess.Popen
    restarts: int = 0
    next_restart_time: float = 0.0


def build_worker_cmd(python_exe: str, config_path: Path, camera_name: str) -> List[str]:
    # Use -m so imports behave consistently.
    return [
        python_exe,
        "-m",
        "holabot_station.camera_worker",
        "--config",
        str(config_path),
        "--camera",
        camera_name,
    ]


def build_child_env(repo_root: Path) -> Dict[str, str]:
    env = dict(os.environ)

    # Ensure child can import holabot_station from ./src without requiring installation.
    src_path = str((repo_root / "src").resolve())
    existing = env.get("PYTHONPATH", "")
    if existing:
        env["PYTHONPATH"] = src_path + os.pathsep + existing
    else:
        env["PYTHONPATH"] = src_path

    # Optional: make OpenCV windows behave better on Windows.
    env.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

    return env


def spawn_worker(
    *,
    python_exe: str,
    repo_root: Path,
    config_path: Path,
    camera_name: str,
    log_level: str,
) -> subprocess.Popen:
    cmd = build_worker_cmd(python_exe, config_path, camera_name)
    env = build_child_env(repo_root)

    log("INFO", f"Starting worker '{camera_name}': {' '.join(cmd)}", log_level)

    # On Windows, CREATE_NEW_PROCESS_GROUP helps us forward Ctrl+C more predictably.
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    return subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        creationflags=creationflags,
    )


def request_stop_process(proc: subprocess.Popen, log_level: str) -> None:
    try:
        if os.name == "nt":
            # Best-effort: send CTRL_BREAK_EVENT to the process group.
            try:
                proc.send_signal(getattr(signal, "CTRL_BREAK_EVENT"))
            except Exception:
                proc.terminate()
        else:
            proc.terminate()
    except Exception as e:
        log("WARNING", f"Failed to stop process PID={getattr(proc, 'pid', '?')}: {e}", log_level)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run station: spawn one worker process per camera")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config.yaml (copy from configs/config.example.yaml)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between orchestrator health checks",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=25,
        help="Maximum restarts per camera before giving up (safety)",
    )
    parser.add_argument(
        "--restart-backoff-seconds",
        type=float,
        default=2.0,
        help="Base backoff before restarting a crashed worker (grows with restart count)",
    )
    args = parser.parse_args(argv)

    # Resolve paths
    config_path = Path(args.config).resolve()
    repo_root = find_repo_root(Path(__file__).resolve())

    cfg = load_config(config_path)
    log_level = pick_log_level(cfg)

    cameras = cfg.get("cameras", [])
    worker_specs: List[WorkerSpec] = []
    for idx, cam in enumerate(cameras):
        if not isinstance(cam, dict) or "name" not in cam:
            raise ValueError("Each cameras[] item must be a dict with at least 'name'.")
        worker_specs.append(WorkerSpec(name=str(cam["name"]), camera_index=idx))

    python_exe = sys.executable

    # Spawn initial workers
    states: List[ProcState] = []
    for spec in worker_specs:
        proc = spawn_worker(
            python_exe=python_exe,
            repo_root=repo_root,
            config_path=config_path,
            camera_name=spec.name,
            log_level=log_level,
        )
        states.append(ProcState(spec=spec, proc=proc))

    stop_requested = False

    def _handle_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    # Ctrl+C / termination
    signal.signal(signal.SIGINT, _handle_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_stop)

    log("INFO", f"Running {len(states)} camera worker(s). Press Ctrl+C to stop.", log_level)

    try:
        while True:
            if stop_requested:
                log("INFO", "Stop requested. Shutting down workers...", log_level)
                break

            for i, st in enumerate(states):
                code = st.proc.poll()
                if code is None:
                    continue

                # Process exited
                log(
                    "WARNING" if code != 0 else "INFO",
                    f"Worker '{st.spec.name}' exited with code {code} (PID was {st.proc.pid}).",
                    log_level,
                )

                if st.restarts >= args.max_restarts:
                    log(
                        "ERROR",
                        f"Worker '{st.spec.name}' exceeded max restarts ({args.max_restarts}). Not restarting.",
                        log_level,
                    )
                    # Keep it dead; user can restart orchestrator.
                    continue

                # Backoff schedule (simple linear growth)
                backoff = args.restart_backoff_seconds * (1.0 + min(st.restarts, 10))
                if st.next_restart_time <= 0.0:
                    st.next_restart_time = _now() + backoff
                    log(
                        "INFO",
                        f"Will restart '{st.spec.name}' in {backoff:.1f}s (restart #{st.restarts + 1}).",
                        log_level,
                    )

                if _now() >= st.next_restart_time:
                    try:
                        new_proc = spawn_worker(
                            python_exe=python_exe,
                            repo_root=repo_root,
                            config_path=config_path,
                            camera_name=st.spec.name,
                            log_level=log_level,
                        )
                        states[i] = ProcState(spec=st.spec, proc=new_proc, restarts=st.restarts + 1)
                    except Exception as e:
                        log("ERROR", f"Failed to restart '{st.spec.name}': {e}", log_level)
                        # Try again later with increased backoff
                        st.restarts += 1
                        st.next_restart_time = _now() + backoff

            time.sleep(max(0.1, float(args.poll_interval)))

    finally:
        # Stop everything
        for st in states:
            if st.proc.poll() is None:
                request_stop_process(st.proc, log_level)

        # Wait a bit, then force-kill if needed
        deadline = _now() + 5.0
        for st in states:
            if st.proc.poll() is None:
                timeout = max(0.0, deadline - _now())
                try:
                    st.proc.wait(timeout=timeout)
                except Exception:
                    try:
                        st.proc.kill()
                    except Exception:
                        pass

        log("INFO", "All workers stopped.", log_level)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
