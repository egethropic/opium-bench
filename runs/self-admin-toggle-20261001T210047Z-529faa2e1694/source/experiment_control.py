"""Graceful local experiment restarts for the observer's explicit Restart action.

The controller never kills a runner or overwrites a recorded experiment. It
requests a cooperative stop, waits for completion, then starts a fresh directory.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import uuid


DEFAULT_SETTINGS = {
    "task_count": 24, "action_budget": 120, "token_budget": 8192,
    "prime_after_actions": 2, "pace_seconds": 4,
}
OPTIONAL_SETTINGS = {
    "joy_dose", "suppression", "half_life_tokens", "cutoff_half_lives",
    "turn_token_limit", "temperature", "top_p", "seed",
}
TERMINAL = {"complete", "completed", "failed", "error", "stopped", "cancelled", "canceled"}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _read_object(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


class ExperimentController:
    def __init__(self, reader, runner_python, vectors_run, hf_home, settings=None):
        self.reader = reader
        # Do not resolve this symlink: a venv's python path selects its environment.
        self.runner_python = Path(runner_python).expanduser().absolute()
        if not self.runner_python.is_file() or not os.access(self.runner_python, os.X_OK):
            raise ValueError("runner_python must be an executable local Python interpreter")
        self.vectors_run = Path(vectors_run).expanduser().absolute()
        self.hf_home = Path(hf_home).expanduser().absolute()
        self.runner_script = Path(__file__).resolve().with_name("self_admin.py")
        if not self.runner_script.is_file():
            raise ValueError("self_admin.py is missing beside experiment_control.py")
        supplied = {} if settings is None else dict(settings)
        unknown = set(supplied) - set(DEFAULT_SETTINGS) - OPTIONAL_SETTINGS
        if unknown:
            raise ValueError("Unsupported experiment settings: " + ", ".join(sorted(unknown)))
        self.settings = {**DEFAULT_SETTINGS, **supplied}
        self._validate_settings()
        with reader.lock:
            self.parent_directory = Path(reader.directory).absolute().parent
        self._lock = threading.Lock()
        self._pending = False
        self._state = "idle"
        self._error = None
        self._request_id = None
        self._requested_utc = None
        self._process = None
        self._process_directory = None
        self._next_directory = None
        self._log_file = None
        self._return_code = None
        self._stop_timeout = 180.0
        self._startup_timeout = 180.0
        self._poll_interval = 0.2

    def _validate_settings(self):
        integers = {"task_count", "action_budget", "token_budget", "prime_after_actions", "turn_token_limit", "seed"}
        for key, value in self.settings.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
            if key in integers and not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
            if key in {"prime_after_actions", "pace_seconds", "temperature", "joy_dose", "seed"}:
                if value < 0:
                    raise ValueError(f"{key} must be nonnegative")
            elif key == "suppression":
                if not 0 <= value <= 1:
                    raise ValueError("suppression must be in [0,1]")
            elif value <= 0:
                raise ValueError(f"{key} must be positive")
        if self.settings["task_count"] > 100:
            raise ValueError("task_count must be at most 100")
        if self.settings["pace_seconds"] > 10:
            raise ValueError("pace_seconds must be at most 10")
        if self.settings.get("top_p", 1) > 1:
            raise ValueError("top_p must be at most 1")

    def status(self):
        # Keep controller and reader locks separate; the observer may independently
        # take reader.lock while collecting its snapshot.
        with self._lock:
            process = self._process
            result = {
                "restart_pending": self._pending, "state": self._state,
                "launch_error": self._error, "error": self._error,
                "request_id": self._request_id, "requested_utc": self._requested_utc,
                "next_experiment_directory": str(self._next_directory) if self._next_directory else None,
                "log_file": str(self._log_file) if self._log_file else None,
                "pid": getattr(process, "pid", None), "return_code": self._return_code,
            }
        with self.reader.lock:
            result["experiment_directory"] = str(self.reader.directory)
        return result

    def restart(self):
        with self._lock:
            accepted = not self._pending
            if accepted:
                request_id = uuid.uuid4().hex
                self._pending = True
                self._state = "restart_requested"
                self._error = None
                self._request_id = request_id
                self._requested_utc = _utc()
                self._next_directory = None
                thread = threading.Thread(target=self._restart_worker, args=(request_id,), name=f"experiment-restart-{request_id[:8]}", daemon=True)
            else:
                thread = None
        if thread is not None:
            try:
                thread.start()
            except Exception as exc:
                self._update(request_id, pending=False, state="failed", error=f"Could not start restart worker: {exc}")
        return {"accepted": accepted, **self.status()}

    def _update(self, request_id, **values):
        with self._lock:
            if self._request_id != request_id:
                return
            for name, value in values.items():
                setattr(self, "_" + name, value)

    def _write_stop_request(self, directory, request_id):
        with self.reader.lock:
            control_path = directory / "control.json"
            events_path = directory / "control_events.jsonl"
            if control_path.is_symlink() or events_path.is_symlink():
                raise ValueError("Control files must not be symbolic links")
            control = _read_object(control_path) or {}
            revision = control.get("revision", 0)
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
                raise ValueError("Current control revision must be a nonnegative integer")
            now = _utc()
            updated = {**control, "stop_requested": True, "stop_request_id": request_id,
                       "stop_requested_utc": now, "revision": revision + 1}
            event = {"event_id": request_id, "operation": "restart_experiment",
                     "actor": "human", "ts": now, "revision": revision + 1,
                     "stop_requested": True}
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                                 prefix=".restart-control-", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    json.dump(updated, stream, allow_nan=False)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, control_path)
                temporary = None
                with events_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            if Path(self.reader.directory).absolute() == directory:
                self.reader.cache.pop("control.json", None)
                self.reader.cache.pop("control_events.jsonl", None)

    def _wait_for_old_run(self, directory, process, request_id):
        deadline = time.monotonic() + self._stop_timeout
        requested = False
        while True:
            manifest = _read_object(directory / "manifest.json")
            alive = process is not None and process.poll() is None
            state = str((manifest or {}).get("status", "")).lower()
            if (manifest is None or state in TERMINAL) and not alive:
                return
            # An owned process can die before recording a terminal manifest.
            if process is not None and process.poll() is not None:
                return
            if manifest is not None and state in TERMINAL and not alive:
                return
            if manifest is not None and state not in TERMINAL and not requested:
                if not (manifest.get("manual_control") or manifest.get("experiment_controls") or manifest.get("restart_supported")):
                    raise ValueError("The active run does not declare support for graceful experiment controls")
                self._write_stop_request(directory, request_id)
                self._update(request_id, state="waiting_for_previous_run")
                requested = True
            if time.monotonic() >= deadline:
                raise TimeoutError(f"The previous experiment did not finish within {self._stop_timeout:g} seconds. It was not killed; its stop request remains set. Retry after it finishes.")
            time.sleep(self._poll_interval)

    def _fresh_directory(self):
        self.parent_directory.mkdir(parents=True, exist_ok=True)
        while True:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            name = f"self-admin-toggle-{stamp}-{uuid.uuid4().hex[:12]}"
            directory = self.parent_directory / name
            log_file = self.parent_directory / (name + ".log")
            if not directory.exists() and not log_file.exists():
                return directory, log_file

    def _argv(self, directory):
        args = [str(self.runner_python), str(self.runner_script), "--interactive",
                "--vectors-run", str(self.vectors_run), "--out", str(directory),
                "--hf-home", str(self.hf_home)]
        for key, value in self.settings.items():
            args.extend(["--" + key.replace("_", "-"), str(value)])
        return args

    def _restart_worker(self, request_id):
        process = None
        try:
            with self.reader.lock:
                old_directory = Path(self.reader.directory).absolute()
            with self._lock:
                old_process = self._process if self._process_directory == old_directory else None
            self._wait_for_old_run(old_directory, old_process, request_id)
            directory, log_file = self._fresh_directory()
            self._update(request_id, state="initializing", next_directory=directory, log_file=log_file)
            # The runner requires a nonexistent --out directory. Its log is a
            # sibling, and no directory is created here before subprocess launch.
            with log_file.open("xb") as output:
                process = subprocess.Popen(self._argv(directory), stdin=subprocess.DEVNULL,
                                           stdout=output, stderr=subprocess.STDOUT,
                                           cwd=str(self.runner_script.parent), shell=False,
                                           start_new_session=True)
            self._update(request_id, process=process, process_directory=directory, return_code=None)
            with self.reader.lock:
                self.reader.directory = directory
                self.reader.cache.clear()
            # Keep duplicate restarts blocked until a manifest exists. A second
            # click during Python startup must not launch a second GPU process.
            deadline = time.monotonic() + self._startup_timeout
            while _read_object(directory / "manifest.json") is None:
                code = process.poll()
                if code is not None:
                    raise RuntimeError(f"Runner exited with code {code} before creating a manifest. See {log_file}")
                if time.monotonic() >= deadline:
                    # Slow imports must not enable another launch while this
                    # process still owns GPU initialization. Continue monitoring.
                    self._update(request_id, state="initializing", error=f"Runner is still initializing after {self._startup_timeout:g} seconds. Restart remains pending to prevent overlapping processes. See {log_file}")
                time.sleep(self._poll_interval)
            self._update(request_id, pending=False, state="running", error=None)
            # Monitoring runs on this daemon thread. A later restart may have its
            # own worker; request IDs prevent an old monitor overwriting its state.
            code = process.wait()
            manifest = _read_object(directory / "manifest.json") or {}
            failed = code != 0 or str(manifest.get("status", "")).lower() not in {"complete", "completed", "stopped", "cancelled", "canceled"}
            error = (f"Runner exited with code {code}. {manifest.get('error', '')} See {log_file}" if failed else None)
            self._update(request_id, return_code=code, pending=False, state="failed" if failed else "complete", error=error)
        except Exception as exc:
            self._update(request_id, pending=False, state="failed", error=f"{type(exc).__name__}: {exc}",
                         return_code=process.poll() if process is not None else None)
