"""Local job control. The model lives in a separate, cancellable process."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
import json
import gzip
import hashlib
import os
import math
import random
import secrets
import subprocess
import sys
import threading
import tempfile

from .storage import ROOT, Store, atomic_json, new_id, preflight, read_json, storage_info, utc_now, guarded_bytes
from .resources import ResourceStop

PROFILES = [
    dict(id="qwen3-4b", name="Qwen3 · 4B", model_id="Qwen/Qwen3-4B",
         revision="1cfa9a7208912126459214e8b04321603b3df60c", quantization="none",
         dtype="bfloat16", precision="BF16", device="cuda", allow_download=False,
         description="Reference model. Native GPU inference with full activation access."),
    dict(id="qwen38-27b-q4", name="Qwen3.8 · 27B NF4", model_id="greghavens/Qwen3.8-27B-bnb-4bit",
         revision="26157380225e427827263c34df23018a1e28dff5",
         quantization="4bit", dtype="bfloat16", precision="NF4 / BF16", device="cuda", allow_download=False,
         local_kernels=True, download_gib=20,
         description="Pinned NF4 conversion of official Qwen3.8-27B. Requires the separate 27B runtime and model-specific calibration."),
]
COMMANDS = {"load_model", "unload_model", "calibrate", "start_session", "chat",
            "control", "inject", "stop", "restart", "start_batch", "pause", "resume", "branch",
            "validate_calibration", "preview_recipe", "preview_session", "score_calibration", "save_preset", "load_preset", "preview_protocol", "run_diagnostics", "start_protocol", "resume_protocol", "analyze_protocol"}


def normalize_config(raw, default="opium"):
    from .protocol import validate_recipe
    if not isinstance(raw, dict):
        raise ValueError("config must be an object")
    raw = deepcopy(raw)
    version = raw.get("recipe_version", 1)
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("Unsupported recipe version")
    if version == 2:
        from .recipes_v2 import resolve_recipe
        if "recipe_id" in raw:
            raw["id"] = raw.pop("recipe_id")
        return resolve_recipe(raw)
    raw.pop("recipe_version", None)
    identifier = raw.pop("recipe_id", raw.get("id", default))
    raw["id"] = identifier
    if "enabled" in raw:
        raw["aux_enabled"] = raw.pop("enabled")
    if "duration" in raw:
        duration = raw.pop("duration")
        if duration not in {"pulse", "hold"}:
            raise ValueError("Duration must be pulse or hold")
        raw["decay"] = "constant" if duration == "hold" else raw.get("decay", "exponential")
    generation = {}
    for name, default_value, low, high, integer in [
        ("temperature", .6, 0, 2, False), ("top_p", .95, .01, 1, False),
        ("top_k", 20, 0, 1000, True), ("max_context_tokens", 8192, 256, 32768, True)]:
        value = raw.pop(name, default_value)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high or (integer and type(value) is not int):
            raise ValueError(f"Invalid {name}")
        generation[name] = value
    history = raw.pop("reasoning_history", "template")
    if history not in {"template", "drop"}:
        raise ValueError("Unknown reasoning history policy")
    result = validate_recipe(raw)
    result.update(generation, reasoning_history=history)
    return result


def bundled_history():
    """Discover published studies without following roots outside this checkout."""
    root = ROOT.resolve()
    paths = [root / "runs"]
    studies = root / "studies"
    if not studies.is_dir() or studies.resolve().parent != root:
        return paths
    for study in sorted(studies.iterdir()):
        if not study.is_dir() or study.resolve().parent != studies.resolve():
            continue
        runs = study / "runs"
        if runs.is_dir() and runs.resolve().parent == study.resolve():
            paths.append(runs)
    return paths


class LabService:
    def __init__(self, data_dir, cache_dir, python=None, historical=None):
        self.store = Store(data_dir, bundled_history() if historical is None else historical)
        self.cache_dir = str(Path(cache_dir).resolve())
        self.python = str(Path(python or sys.executable).absolute())
        self.csrf = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.send_lock = threading.Lock()
        self.events = deque(maxlen=12000)
        self.seq = 0
        self.process = None
        self.worker = dict(status="unloaded", model=None, error=None)
        self.job = None
        self.preview = None
        self.session = dict(id=None, status="idle", mode="chat", config={}, metrics={}, events=[], conversation=[])
        self.last_start = None
        self.last_start_run_id = None
        self.pending = set()
        self.closed = False
        self.instance_id = new_id("service")
        self.process_runs = {}
        self.process_commands = {}
        self.command_runs = {}
        self.research_commands = {}
        self.handled_processes = set()
        self.resource_failures = {}
        from .resources import ResourceGuard
        self.resources = ResourceGuard(dict(data=self.store.root, cache=self.cache_dir,
                                            temp=self.store.root / "tmp"))
        self.store.guard = self.resources
        self._recover_abandoned_runs()
        from .research_jobs import recover_dispatches
        recover_dispatches(self.store,self._pid_alive)

    def _research_terminal(self,command_id,status,reason,*,release=True):
        binding=self.research_commands.get(command_id)
        if not binding:return
        from .research_jobs import record_dispatch_stop,release_dispatch
        identifier,execution_id=binding
        if status in {'failed','stopped','resource_stopped'}:
            record_dispatch_stop(self.store,identifier,execution_id,status,reason,actor='supervisor')
        if release:
            try:release_dispatch(self.store,identifier,execution_id,actor='all')
            except ValueError:
                record_dispatch_stop(self.store,identifier,execution_id,'failed','Worker ended without a durable research terminal receipt',actor='supervisor')
                release_dispatch(self.store,identifier,execution_id,actor='all')
            self.research_commands.pop(command_id,None)

    @staticmethod
    def _pid_alive(pid):
        if type(pid) is not int or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except (PermissionError, OSError):
            return True  # Uncertain ownership is never permission to abort a run.
        return True

    def _claim_run(self, identifier, worker_pid=None):
        self.store.prepare_emergency(identifier)
        owner = dict(service_id=self.instance_id, service_pid=os.getpid())
        if type(worker_pid) is int and worker_pid > 0:
            owner["worker_pid"] = worker_pid
        self.store.update(identifier, owner=owner)

    def _recover_abandoned_runs(self):
        """Close only unfinished local records whose recorded owners are gone.

        Runs without ownership metadata are left untouched: an older service
        may still be using them. Historical imports and finished runs are never
        edited. Recovery appends an explicit event and preserves raw records.
        """
        for path in self.store.runs.iterdir():
            if (path / "_portable").is_dir():
                continue  # Imported owners and statuses are inert source evidence.
            manifest = read_json(path / "manifest.json", {}) if path.is_dir() else {}
            if manifest.get("status") not in {"queued", "running", "awaiting_user", "paused"}:
                continue
            owner = manifest.get("owner")
            if not isinstance(owner, dict) or not owner.get("service_id"):
                continue
            if self._pid_alive(owner.get("service_pid")) or self._pid_alive(owner.get("worker_pid")):
                continue
            try:
                self._finalize_interrupted_run(path.name, "service_restarted", intentional=False)
            except (OSError, ValueError):
                # A damaged record must not prevent reviewing unrelated runs.
                continue

    def _finalize_interrupted_run(self, identifier, reason, intentional=False):
        if (self.store.run_path(identifier) / "_portable").is_dir():
            return
        run = self.store.read_run(identifier)
        manifest, events = run["manifest"], run["events"]
        if manifest.get("status") not in {"queued", "running", "awaiting_user", "paused"}:
            return
        summary = deepcopy(run.get("summary") or {})
        for event in events:
            if event.get("type") == "metrics":
                summary.update(event.get("metrics", {}))
            elif event.get("type") == "session_finished":
                # Finish event may have reached disk immediately before the
                # manifest write was interrupted. Recover the recorded outcome.
                self.store.update(identifier, status=event.get("status", "complete"),
                                  summary=event.get("summary", {}), finished_at=event.get("time", utc_now()))
                if identifier == self.session.get("id"):
                    self.session.update(status=event.get("status", "complete"), metrics=event.get("summary", {}))
                return
        if identifier == self.session.get("id"):
            summary.update(self.session.get("metrics", {}))
        cfg = manifest.get("config", {})
        summary.setdefault("assigned", 0 if manifest.get("mode") == "chat" else cfg.get("task_count", 0))
        summary.setdefault("submitted", 0)
        summary.setdefault("correct", 0)
        summary.setdefault("actions", 0)
        token_events = [e for e in events if e.get("type") == "token"]
        indexed = [e.get("generation_index") for e in token_events if type(e.get("generation_index")) is int]
        recovered_tokens = max(len(token_events), max(indexed, default=-1) + 1)
        prior_tokens = summary.get("tokens", 0)
        summary["tokens"] = max(prior_tokens, recovered_tokens)
        summary["tokens_since_last_metrics"] = max(0, recovered_tokens - prior_tokens)
        summary["reasoning_tokens"] = max(summary.get("reasoning_tokens", 0), sum(e.get("phase") == "reasoning" for e in token_events))
        summary["output_tokens"] = max(summary.get("output_tokens", 0), sum(e.get("phase") != "reasoning" for e in token_events))
        completed_actions = [e.get("action") for e in events if e.get("type") in {"tool", "action"} and type(e.get("action")) is int]
        summary["actions"] = max(summary["actions"], max(completed_actions, default=0))
        summary.update(termination=reason, recovered=True)
        # Do not concatenate the recovery event onto a partial JSON line left
        # by a killed process; retain those bytes as their own skipped record.
        log = self.store.run_path(identifier, writable=True) / "events.jsonl"
        if log.exists() and log.stat().st_size:
            with log.open("rb") as stream:
                stream.seek(-1, 2)
                boundary = stream.read(1) == b"\n"
            if not boundary:
                with log.open("ab") as stream:
                    stream.write(b"\n")
        self.seq = max(self.seq, max((e.get("seq", 0) for e in events if type(e.get("seq", 0)) is int), default=0))
        queued = manifest.get("status") == "queued" and not any(e.get("type") == "session_started" for e in events)
        status = "cancelled" if queued else "stopped" if intentional else "failed"
        self.event(dict(type="session_finished", run_id=identifier, status=status, summary=summary,
                        recovery_note="Last saved metrics retained; emitted tokens after the last metrics snapshot recovered from the event log."))

    def recipes(self):
        from .protocol import list_recipes
        return list_recipes()

    def protocol_catalog(self):
        from .protocol_library import list_protocols
        return [{key: item[key] for key in ("id", "title", "question", "content_sha256", "endpoints")}
                for item in list_protocols()]

    def model_profiles(self):
        rows = deepcopy(PROFILES)
        model = self.worker.get("model") or {}
        loaded_id = model.get("model_id", model.get("model", ""))
        for p in rows:
            base = Path(self.cache_dir) / "hub" / ("models--" + p["model_id"].replace("/", "--")) / "snapshots"
            p["available"] = base.exists() and any(base.iterdir())
            p["loaded"] = loaded_id == p["model_id"]
        return rows

    def state(self):
        with self.lock:
            session = deepcopy(self.session)
            session["restart_available"] = bool(self.last_start and self.last_start_run_id == session.get("id"))
            session["events"] = [e for e in list(self.events)[-500:] if e.get("run_id") == session["id"]]
            result = dict(csrf=self.csrf, worker=deepcopy(self.worker), session=session,
                          job=deepcopy(self.job), preview=deepcopy(self.preview), cursor=self.seq)
        from .presets import catalog as preset_catalog
        from .protocol_runner import LocalWorkerBackend
        from .research_jobs import catalog as research_catalog
        result.update(presets=preset_catalog(self.store.root), models=self.model_profiles(), calibrations=self.store.calibration_catalog(),
                      runs=self.store.catalog(), recipes=self.recipes(), protocols=self.protocol_catalog(), research_jobs=research_catalog(self.store), research_capabilities=sorted(LocalWorkerBackend.capabilities), ratings=self.rating_catalog(),
                      storage=dict(storage_info(self.store.root, self.cache_dir), guard=self.resources.status()))
        return result

    def rating_path(self, identifier):
        from .storage import ID
        if not isinstance(identifier, str) or not ID.fullmatch(identifier):
            raise ValueError("Invalid ratings identifier")
        root = self.store.root / "ratings"
        path = root / (identifier + ".json")
        if root.is_symlink() or path.is_symlink() or not path.is_file():
            raise FileNotFoundError("Ratings not found")
        return path

    def rating_catalog(self):
        root = self.store.root / "ratings"
        if not root.is_dir() or root.is_symlink():
            return []
        rows = []
        for path in root.glob("*.json"):
            if path.is_symlink():
                continue
            record = read_json(path, {})
            if isinstance(record, dict) and record.get("id") == path.stem:
                rows.append({key: record.get(key) for key in ("id", "calibration_id", "created_at", "scorer", "calibration_sha256")})
        return sorted(rows, key=lambda row: row.get("created_at") or "", reverse=True)

    def since(self, after):
        with self.lock:
            return dict(events=[e for e in self.events if e["seq"] > after], cursor=self.seq,
                        reset=bool(after > self.seq or (self.events and after and after < self.events[0]["seq"] - 1)))

    def event(self, event):
        with self.lock:
            self.seq += 1
            e = dict(event, seq=self.seq, time=event.get("time", utc_now()))
            self.events.append(e)
            kind = e.get("type")
            identifier = e.get("run_id")
            if kind == "worker":
                self.worker.update({k: e[k] for k in ("status", "model", "error") if k in e})
            elif kind == "session_preview":
                self.preview = dict(command_id=e.get("command_id"), preview=e.get("preview"))
            elif kind == "research_run_registered":
                self._claim_run(identifier, getattr(self.process, "pid", None))
                if self.process:
                    self.process_runs.setdefault(self.process, set()).add(identifier)
                self.command_runs.setdefault(e.get("command_id"), set()).add(identifier)
            elif kind == "research_progress":
                self.job = dict(self.job or {}, command_id=e.get("command_id"), status="running", kind="research",
                    research_job_id=e.get("research_job_id"), episode_id=e.get("episode_id"), stage=e.get("stage"),
                    message=f"{e.get('episode_id')} · {e.get('stage')} · {e.get('status')}")
            elif kind == "job":
                self.job = {k: v for k, v in e.items() if k not in {"seq", "type"}}
                if e.get("status") in {"complete", "failed", "stopped", "resource_stopped"}:
                    self._research_terminal(e.get('command_id'),e['status'],e.get('resource_stop',e.get('message','worker_terminal')))
                    self.pending.discard(e.get("command_id"))
                    affected = self.command_runs.pop(e.get("command_id"), set())
                    if e.get("status") in {"failed", "resource_stopped"}:
                        for run_id in affected:
                            self._finalize_interrupted_run(run_id, "worker_job_failed")
            elif kind == "session_started":
                if identifier != self.last_start_run_id:
                    self.last_start = None
                    self.last_start_run_id = None
                self.session = dict(id=identifier, status="running", mode=e.get("mode", "experiment"),
                    config=deepcopy(e.get("config", {})), metrics={}, events=[], conversation=[])
                self.store.update(identifier, status="running", config=e.get("config", {}),
                                  model=e.get("model", {}), calibration_id=e.get("calibration_id"), started_at=utc_now())
            if identifier and kind not in {"research_run_registered", "research_progress"}:
                self.store.append(identifier, e)
                if identifier == self.session.get("id"):
                    if kind == "message":
                        self.session["conversation"].append({k: e[k] for k in ("role", "content", "reasoning") if k in e})
                    elif kind in {"metrics", "session_finished"}:
                        self.session["metrics"] = e.get("metrics", e.get("summary", {}))
                    elif kind == "status":
                        self.session["status"] = e.get("status", self.session["status"])
                    elif kind == "control":
                        settings = e.get("settings", {})
                        configurations = [self.session["config"]]
                        if self.last_start and self.last_start_run_id == identifier:
                            configurations.append(self.last_start["config"])
                        for config in configurations:
                            for key, value in settings.items():
                                if key == "reset":
                                    config.update(baseline_pain=0, baseline_joy=0, baseline_suppression=0)
                                    if config.get("recipe_version") == 2:
                                        config["baseline_joy_suppression"] = 0
                                elif key == "duration":
                                    config["decay"] = "constant" if value == "hold" else settings.get("decay", "exponential")
                                else:
                                    name = "baseline_" + key if key in {"pain", "joy", "suppression", "joy_suppression", "joy_direction"} else "aux_enabled" if key == "enabled" else key
                                    config[name] = value
                    if kind == "session_finished":
                        self.session["status"] = e.get("status", "complete")
                if kind == "session_finished":
                    self.store.update(identifier, status=e.get("status", "complete"),
                                      summary=e.get("summary", {}), finished_at=utc_now())
                elif kind == "status":
                    self.store.update(identifier, status=e.get("status", "running"))
            return e

    @contextmanager
    def portable_export(self, identifier):
        """Export a settled record; a running writer must be stopped first."""
        from .portability import export_bundle
        with self.lock:
            source = self.store.run_path(identifier)
            manifest = self.store.read_run(identifier)["manifest"]
            imported = (source / "_portable").is_dir()
            if not imported and manifest.get("status") in {"queued", "running", "paused", "awaiting_user"}:
                raise ValueError("Stop this session before exporting its complete portable record")
            calibration = None
            if manifest.get("calibration_id"):
                try:
                    calibration = self.store.calibration_path(manifest["calibration_id"])
                except ValueError:
                    pass  # Historical evidence may not include a local bundle.
            temporary = self.store.root / "tmp"
            temporary.mkdir(exist_ok=True)
            directory = tempfile.TemporaryDirectory(prefix="export-", dir=temporary)
            try:
                target = Path(directory.name) / (identifier + ".zip")
                export_bundle(source, target, calibration_dir=calibration, guard=self.resources)
            except BaseException:
                directory.cleanup()
                raise
        try:
            yield target  # Do not hold the service lock during a browser download.
        finally:
            directory.cleanup()

    def import_record(self, stream, size, content_type):
        """Receive bounded raw evidence without loading a model or trusting paths."""
        from .portability import import_bundle, import_json_export, Limits
        limits = Limits()
        maximum = limits.compressed_bytes if content_type == "application/zip" else limits.member_bytes
        if type(size) is not int or not 0 < size <= maximum:
            raise ValueError("Evidence upload exceeds the 128 MiB ZIP / 64 MiB JSON limit")
        temporary = self.store.root / "tmp"
        temporary.mkdir(exist_ok=True)
        operation = new_id("upload")
        self.resources.reserve(operation, {temporary: size})
        try:
            with tempfile.TemporaryDirectory(prefix="import-", dir=temporary) as directory:
                source = Path(directory) / "upload"
                with source.open("xb") as output:
                    remaining = size
                    while remaining:
                        chunk = stream.read(min(remaining, self.resources.max_write_bytes))
                        if not chunk:
                            raise ValueError("Evidence upload ended before its declared size")
                        self.resources.before_write(operation, temporary, len(chunk))
                        output.write(chunk)
                        self.resources.written(operation, temporary, len(chunk))
                        remaining -= len(chunk)
                importer = import_bundle if content_type == "application/zip" else import_json_export
                with self.lock:
                    reserved = {row["id"] for row in self.store.catalog()}
                    return importer(source, self.store.runs, guard=self.resources, reserved_ids=reserved)
        finally:
            self.resources.release(operation)

    def _launch(self):
        if self.process and self.process.poll() is None:
            return
        if self.process:
            self._worker_exited(self.process, self.process.poll())
        env = os.environ.copy()
        temp = self.store.root / "tmp"
        temp.mkdir(exist_ok=True)
        env.update(OPIUM_DATA_DIR=str(self.store.root), HF_HUB_DISABLE_XET="1", HF_HUB_ENABLE_HF_TRANSFER="0", HF_ENABLE_PARALLEL_LOADING="false", HF_HUB_DOWNLOAD_TIMEOUT="30", HF_HOME=self.cache_dir, TMPDIR=str(temp), TEMP=str(temp), TMP=str(temp),
                   HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false",
                   TORCH_HOME=str(self.store.root / "torch-cache"),
                   PIP_CACHE_DIR=str(self.store.root / "pip-cache"),
                   TRITON_CACHE_DIR=str(self.store.root / "triton-cache"),
                   CUDA_CACHE_PATH=str(self.store.root / "cuda-cache"),
                   TORCHINDUCTOR_CACHE_DIR=str(self.store.root / "inductor-cache"),
                   XDG_CACHE_HOME=str(self.store.root / "xdg-cache"), PYTHONUNBUFFERED="1")
        from .resources import EmergencyMetadata
        operation = new_id("worker-reserve")
        self.resources.reserve(operation, {self.cache_dir:256*1024**2, self.store.root:8*1024**2})
        try:
            emergency = EmergencyMetadata(self.store.root / "resource-stops").allocate(self.resources)
        except BaseException:
            self.resources.release(operation)
            raise
        try:
            self.process = subprocess.Popen([self.python, "-m", "lab.worker", "--cache-dir", self.cache_dir],
                cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
                start_new_session=True)
        except BaseException:
            self.resources.release(operation)
            emergency.release()
            raise
        process = self.process
        threading.Thread(target=self._read_worker, args=(process,), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(process,), daemon=True).start()
        threading.Thread(target=self._supervise_worker, args=(process,operation,emergency), daemon=True).start()

    def _resource_failure(self, process, error):
        detail = error.to_dict() if isinstance(error, ResourceStop) else dict(error)
        with self.lock:
            self.resource_failures.setdefault(process, detail)
            for command_id in self.process_commands.get(process,set()):
                self._research_terminal(command_id,'resource_stopped',detail,release=False)
            owned = set(self.process_runs.get(process, set()))
            if self.process is process and self.session.get("id"):
                owned.add(self.session["id"])
            for identifier in owned:
                try:
                    run = self.store.read_run(identifier)
                    if run["manifest"].get("status") in {"complete", "failed", "stopped", "cancelled", "resource_stopped"}:
                        continue
                    summary = self.session.get("metrics", {}) if self.session.get("id") == identifier else run.get("summary", {})
                    self.store.resource_stop(identifier, detail, summary)
                except (OSError, ValueError):
                    pass  # The supervisor also retains a separately preallocated record.
            if self.process is process:
                if self.session.get("id") in owned and self.session.get("status") not in {"complete","failed","stopped","cancelled"}:
                    self.session["status"] = "resource_stopped"
                self.worker.update(status="error", error=detail.get("reason", "Storage reserve reached"))
                self.job = dict(status="resource_stopped", message=detail.get("reason", "Storage reserve reached"))

    def _supervise_worker(self, process, operation, emergency):
        from .resources import supervise_process
        service = self
        class ObservedGuard:
            def check(self, **kwargs):
                try:
                    return service.resources.check(**kwargs)
                except ResourceStop as exc:
                    service._resource_failure(process, exc)
                    raise
        try:
            supervise_process(process, ObservedGuard(), operation=operation, emergency=emergency,
                              poll_seconds=1.0, process_group=True)
        finally:
            self.resources.release(operation)
            emergency.release()

    def _read_stderr(self, process):
        path = self.store.root / "worker.log"
        try:
            while True:
                chunk = process.stderr.read(4096)
                if not chunk:
                    return
                try:
                    if path.exists() and path.stat().st_size > 8*1024**2:
                        path.replace(path.with_suffix(".log.1"))
                    guarded_bytes(path, chunk.encode("utf-8", errors="replace"), mode="ab", guard=self.resources)
                except ResourceStop as exc:
                    self._resource_failure(process, exc)
                    # Keep draining until the supervisor cancels the owned child.
        except (OSError, ValueError):
            return

    def _read_worker(self, process):
        try:
            for line in process.stdout:
                try:
                    event = json.loads(line)
                    if isinstance(event, dict) and "type" in event:
                        with self.lock:
                            if self.process is process:
                                self.event(event)
                except ResourceStop as exc:
                    self._resource_failure(process, exc)
                except (ValueError, OSError) as exc:
                    try:
                        guarded_bytes(self.store.root / "worker.log", f"Worker output diagnostic: {str(exc)[:300]}\n".encode(), mode="ab", guard=self.resources)
                    except ResourceStop as stopped:
                        self._resource_failure(process, stopped)
        finally:
            code = process.wait()
            self._worker_exited(process, code)

    def _worker_exited(self, process, code):
        """Idempotently settle every episode/job owned by this worker."""
        with self.lock:
            if process in self.handled_processes:
                return
            self.handled_processes.add(process)
            current = self.process is process
            runs = self.process_runs.pop(process, set())
            commands = self.process_commands.pop(process, set())
            for command_id in commands:
                self._research_terminal(command_id,'resource_stopped' if self.resource_failures.get(process) else 'stopped' if self.closed else 'failed',
                    self.resource_failures.get(process,'service_shutdown' if self.closed else 'worker_exited'))
            for command_id in commands:
                self.command_runs.pop(command_id, None)
            if current and self.session.get("id"):
                runs.add(self.session["id"])
            resource_failure = self.resource_failures.get(process)
            reason = "storage_reserve" if resource_failure else "service_shutdown" if self.closed else "worker_exited"
            for identifier in runs:
                try:
                    self._finalize_interrupted_run(identifier, reason, intentional=self.closed)
                except (OSError, ValueError):
                    continue
            if current:
                active_job = (self.job or {}).get("status") in {"queued", "running"}
                clean = self.closed or (code == 0 and self.worker.get("status") == "unloaded" and not active_job and not self.pending)
                self.event(dict(type="worker", status="unloaded" if clean else "error", model=None,
                                error=None if clean else resource_failure.get("reason") if resource_failure else f"Worker exited ({code}); saved partial results. Load the model to continue."))
                if active_job:
                    self.event(dict(type="job", command_id=self.job.get("command_id"),
                                    status="resource_stopped" if resource_failure else "stopped" if self.closed else "failed", message=reason))
                self.pending.difference_update(commands)
                # No old worker command can still be executing after its exit.
                self.pending.clear()
                self.process = None

    def _send(self, command, payload):
        command_id = new_id("cmd")
        runs = {payload["run_id"]} if command in {"start_session", "restore_session", "run_diagnostics"} else {e["run_id"] for e in payload.get("entries", [])} if command in {"start_batch", "run_research_job"} else set()
        try:
            if command=='run_research_job':
                from .research_jobs import prepare_dispatch
                payload=dict(payload,execution_id=prepare_dispatch(self.store,payload['job_id']))
                self.research_commands[command_id]=(payload['job_id'],payload['execution_id'])
            self._launch()
            with self.lock:
                self.process_runs.setdefault(self.process, set()).update(runs)
                if command not in {"control", "inject"}:
                    self.process_commands.setdefault(self.process, set()).add(command_id)
                    self.pending.add(command_id)
                if runs:
                    self.command_runs[command_id] = set(runs)
                for identifier in runs:
                    self._claim_run(identifier, getattr(self.process, "pid", None))
                if command=='run_research_job':
                    from .research_jobs import bind_dispatch_worker
                    bind_dispatch_worker(self.store,payload['job_id'],payload['execution_id'],self.process.pid)
            with self.send_lock:
                self.process.stdin.write(json.dumps(dict(command=command, payload=payload, id=command_id), allow_nan=False) + "\n")
                self.process.stdin.flush()
        except ResourceStop as exc:
            self._research_terminal(command_id,'resource_stopped',exc)
            self.pending.discard(command_id)
            for identifier in runs:
                try:
                    self.store.resource_stop(identifier, exc)
                except OSError:
                    pass
            if self.process:
                self._resource_failure(self.process, exc)
            self.worker.update(status="error", error=str(exc))
            self.job = dict(status="resource_stopped", message=str(exc))
            raise
        except (OSError, ValueError) as exc:
            self._research_terminal(command_id,'failed',exc)
            self.pending.discard(command_id)
            for identifier in runs:
                self._finalize_interrupted_run(identifier, "worker_dispatch_failed")
            self.event(dict(type="worker", status="error", model=None, error=f"Could not send worker command: {exc}"))
            self.event(dict(type="job", command_id=command_id, status="failed", message="worker_dispatch_failed"))
            raise
        return dict(accepted=True, command_id=command_id)

    def command(self, command, payload):
        if command not in COMMANDS or not isinstance(payload, dict):
            raise ValueError("Unknown command or invalid payload")
        with self.lock:
            busy = self.worker.get("status") in {"loading", "calibrating", "running", "paused"} or (self.job or {}).get("status") == "running"
            if busy and command in {"load_model", "unload_model", "calibrate", "validate_calibration", "preview_session", "start_session", "start_batch", "restart", "branch", "run_diagnostics", "start_protocol", "resume_protocol"}:
                raise ValueError("Stop the current job before starting another")
            if command == "stop":
                if not self.process or self.process.poll() is not None:
                    return dict(accepted=True, message="Already stopped")
                return self._send(command, {})
            if command == "restart":
                if not self.last_start or self.last_start_run_id != self.session.get("id"):
                    raise ValueError("Restart is available only for the current standalone session. Use the frozen research plan or a saved boundary for research and branch runs.")
                command, payload = "start_session", deepcopy(self.last_start)
            if command == "load_model":
                profile_id = payload.get("profile_id", "qwen3-4b")
                profile = next((deepcopy(p) for p in PROFILES if p["id"] == profile_id), None)
                if profile is None:
                    raise ValueError("Unknown model profile")
                # A revision belongs to one repository. A new checkpoint must
                # not inherit the reference 4B repository's pinned commit.
                original_model_id = profile["model_id"]
                original_revision = profile.get("revision")
                for key in ("model_id", "revision", "quantization", "dtype", "allow_download", "local_kernels"):
                    if key in payload:
                        profile[key] = payload[key]
                if profile["model_id"] != original_model_id and not payload.get("revision"):
                    profile.pop("revision", None)
                catalog_checkpoint = (profile["model_id"] == original_model_id and
                                      profile.get("revision") == original_revision)
                if (not isinstance(profile["model_id"], str) or
                        (not catalog_checkpoint and not profile["model_id"].startswith("Qwen/"))):
                    if not (isinstance(profile["model_id"], str) and payload.get("acknowledge_custom_checkpoint") is True):
                        raise ValueError("Custom checkpoints require acknowledge_custom_checkpoint=true")
                if profile.get("allow_download"):
                    # Only a pinned catalog conversion gets its smaller download estimate.
                    pinned_conversion = catalog_checkpoint and profile.get("quantization") == "4bit"
                    gib = profile.get("download_gib") if pinned_conversion else None
                    estimate = (gib if gib is not None else (65 if "27B" in profile["model_id"] else 12)) * 2**30
                    preflight(self.cache_dir, estimate)
                self.worker.update(status="loading", error=None)
                return self._send(command, dict(profile=profile))
            if command == "unload_model":
                return self._send(command, {})
            if command == "analyze_protocol":
                from .research_jobs import analyze_job
                if set(payload) - {"research_job_id", "endpoint", "arm_a", "arm_b", "attempt_policy"}:
                    raise ValueError("Unknown research analysis option")
                return dict(accepted=True, analysis=analyze_job(self.store, payload.get("research_job_id"),
                    **{k:v for k,v in payload.items() if k != "research_job_id"}))
            if command == "preview_protocol":
                from .protocol_library import dry_run, load_protocol
                if set(payload) - {"protocol_id", "document", "mode", "seeds", "order_seed"}:
                    raise ValueError("Unknown protocol preview option")
                document = payload.get("document") if "document" in payload else load_protocol(payload.get("protocol_id"))
                preview = dry_run(document, payload.get("mode", "smoke"), seeds=payload.get("seeds"), order_seed=payload.get("order_seed", 1729))
                return dict(accepted=True, preview=preview, preview_json=json.dumps(preview, ensure_ascii=False, allow_nan=False, indent=2))
            if command in {"save_preset", "load_preset"}:
                from .presets import save, load
                if command == "save_preset":
                    asset = save(self.store.root, payload.get("kind"), payload.get("id"), payload.get("value"), self.resources)
                else:
                    asset = load(self.store.root, payload.get("kind"), payload.get("id"))
                return dict(accepted=True, asset=asset)
            if command == "preview_recipe":
                from .session_v2 import build_session
                config = normalize_config(payload.get("config", {}))
                if config.get("recipe_version") != 2:
                    raise ValueError("The editor preview requires an explicit version 2 recipe")
                grammar = (self.worker.get("model") or {}).get("tool_call_format", "json")
                result = build_session(config, payload.get("mode", "experiment"), grammar, include_task=True)
                return dict(accepted=True, preview=result,
                    config_json=json.dumps(result["config"], ensure_ascii=False, allow_nan=False, indent=2),
                    exact_template=False, tool_call_format=grammar)
            if command == "score_calibration":
                from .scoring import score_blinded_sheet
                source_id = payload.get("calibration_id")
                source = self.store.calibration_path(source_id)
                metadata = read_json(source / "calibration.json", {})
                key_file = source / "scoring-key.json"
                if key_file.is_symlink() or not key_file.is_file():
                    raise ValueError("This calibration has no independent scoring export")
                key_raw = key_file.read_bytes()
                if hashlib.sha256(key_raw).hexdigest() != metadata.get("evidence_sha256", {}).get("scoring-key.json"):
                    raise ValueError("Calibration scoring-key integrity check failed")
                scores = score_blinded_sheet(payload.get("sheet"), json.loads(key_raw))
                record = dict(id=new_id("ratings"), calibration_id=source_id, created_at=utc_now(),
                    calibration_sha256=hashlib.sha256((source / "calibration.json").read_bytes()).hexdigest(),
                    scorer=payload.get("scorer", "anonymous"), sheet=payload["sheet"], scores=scores)
                if not isinstance(record["scorer"], str) or len(record["scorer"]) > 160:
                    raise ValueError("Scorer label must contain at most 160 characters")
                directory = self.store.root / "ratings"
                raw = (json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
                directory.mkdir(exist_ok=True)
                path = directory / (record["id"] + ".json")
                temporary = path.with_suffix(".json.tmp")
                try:
                    with temporary.open("xb") as stream:
                        for offset in range(0, len(raw), self.resources.max_write_bytes):
                            chunk = raw[offset:offset + self.resources.max_write_bytes]
                            with self.resources.write(directory, len(chunk)):
                                stream.write(chunk)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                return dict(accepted=True, rating_id=record["id"], scores=scores)
            if self.worker.get("model") is None:
                raise ValueError("Load a model first")
            if command == "resume_protocol":
                from .research_jobs import load_plan
                if set(payload) - {"research_job_id", "expansion_sha256", "retry_failed"}:
                    raise ValueError("Unknown research resume option")
                if type(payload.get("retry_failed", False)) is not bool:
                    raise ValueError("retry_failed must be an explicit boolean")
                if self.session.get("status") in {"running", "awaiting_user", "generating", "queued", "paused"}:
                    raise ValueError("Stop the current session before resuming a research job")
                if not isinstance(payload.get("expansion_sha256"), str):
                    raise ValueError("Submit the frozen job expansion hash")
                receipt, _, _ = load_plan(self.store, payload.get("research_job_id"), verify_sources=True,
                    expected_expansion_sha256=payload["expansion_sha256"])
                entries = [dict(run_id=e["run_id"]) for e in receipt["entries"] if e["status"] != "complete"]
                self.worker["status"] = "running"
                result = self._send("run_research_job", dict(job_id=receipt["id"], data_dir=str(self.store.root),
                    expected_expansion_sha256=receipt["expansion_sha256"], resume=True,
                    retry_failed=payload.get("retry_failed", False), entries=entries))
                return dict(result, research_job_id=receipt["id"])
            if command == "start_protocol":
                from .protocol_library import dry_run, load_protocol
                from .research_jobs import create_job
                from .protocol_runner import LocalWorkerBackend
                allowed = {"protocol_id", "document", "mode", "seeds", "order_seed", "expansion_sha256", "calibration_id", "bindings"}
                if set(payload) - allowed:
                    raise ValueError("Unknown research launch option")
                if self.session.get("status") in {"running", "awaiting_user", "generating", "queued", "paused"}:
                    raise ValueError("Stop the current session before launching a research job")
                document = payload.get("document") if "document" in payload else load_protocol(payload.get("protocol_id"))
                preview = dry_run(document, payload.get("mode", "smoke"), seeds=payload.get("seeds"), order_seed=payload.get("order_seed", 1729))
                if payload.get("expansion_sha256") != preview["expansion_sha256"]:
                    raise ValueError("Preview the exact protocol and submit its unchanged expansion hash")
                receipt, entries = create_job(self.store, preview, payload.get("calibration_id"), bindings=payload.get("bindings"),
                    guard=self.resources, capabilities=LocalWorkerBackend.capabilities, model_info=self.worker["model"])
                for entry in entries:
                    self._claim_run(entry["run_id"])
                self.worker["status"] = "running"
                result = self._send("run_research_job", dict(job_id=receipt["id"], data_dir=str(self.store.root),
                    expected_expansion_sha256=receipt["expansion_sha256"], resume=False, retry_failed=False, entries=entries))
                return dict(result, research_job_id=receipt["id"], planned=len(entries))
            if command == "run_diagnostics":
                return self._start_diagnostics(payload)
            if command == "branch":
                return self._branch_session(payload)
            if command == "preview_session":
                config = normalize_config(payload.get("config", {}))
                if config.get("recipe_version") != 2:
                    raise ValueError("Exact editor preview requires a version 2 recipe")
                return self._send(command, dict(config=config, mode=payload.get("mode", "experiment"), include_task=True))
            if command in {"pause", "resume"}:
                if not self.session.get("id") or self.session.get("status") in {"complete", "failed", "stopped", "cancelled"}:
                    raise ValueError("Start an active session first")
                if payload:
                    raise ValueError("Pause and resume do not take settings")
                return self._send(command, {})
            if command == "calibrate":
                preflight(self.store.root, 64 * 2**20)
                identifier = new_id("cal")
                path = self.store.calibrations / identifier
                self.worker["status"] = "calibrating"
                return self._send(command, dict(config=payload, out_dir=str(path), calibration_id=identifier))
            if command == "validate_calibration":
                source = self.store.calibration_path(payload.get("calibration_id"))
                options = {k: v for k, v in payload.items() if k != "calibration_id"}
                allowed = {"doses", "validation_pairs_per_concept", "continuation_tokens", "max_kl", "max_relative_delta"}
                if set(options) - allowed:
                    raise ValueError("Unknown calibration validation option")
                options["calibration_dir"] = str(source)
                preflight(self.store.root, 256 * 2**20)
                identifier = new_id("cal")
                self.worker["status"] = "calibrating"
                return self._send(command, dict(config=options, out_dir=str(self.store.calibrations / identifier), calibration_id=identifier))
            if command in {"start_session", "start_batch"}:
                calibration_id = payload.get("calibration_id")
                self.store.calibration_path(calibration_id)
                from .protocol import validate_recipe
                raw = dict(payload.get("config", {}))
                recipe_id = raw.get("recipe_id", raw.get("id", "conversation" if payload.get("mode") == "chat" else "opium"))
                cfg = normalize_config(dict(raw, id=recipe_id))
                preflight(self.store.root, 64 * 2**20)
                if command == "start_session":
                    mode = payload.get("mode", "chat")
                    if mode not in {"chat", "experiment"}:
                        raise ValueError("Choose chat or experiment mode")
                    identifier, path = self.store.create(mode, cfg, payload.get("parent"))
                    self._claim_run(identifier)
                    self.last_start = dict(mode=mode, config=cfg, calibration_id=calibration_id)
                    if payload.get("initial_messages"):
                        self.last_start["initial_messages"] = deepcopy(payload["initial_messages"])
                    self.last_start_run_id = identifier
                    data = dict(mode=mode, config=cfg, calibration_id=calibration_id,
                                calibration_dir=str(self.store.calibration_path(calibration_id)), run_id=identifier,
                                out_dir=str(path), initial_messages=payload.get("initial_messages", []))
                    self.worker["status"] = "running"
                    result = self._send(command, data)
                    result["run_id"] = identifier
                    return result
                recipe_ids = payload.get("recipe_ids", [recipe_id])
                seeds = payload.get("seeds", [cfg.get("seed", 17)])
                thinking_modes = payload.get("thinking_modes")
                if thinking_modes is not None and (not isinstance(thinking_modes, list) or not thinking_modes or any(type(v) is not bool for v in thinking_modes) or len(thinking_modes) != len(set(thinking_modes))):
                    raise ValueError("thinking_modes must be a distinct boolean list")
                if not isinstance(recipe_ids, list) or not isinstance(seeds, list) or not 1 <= len(recipe_ids) * len(seeds) <= 160:
                    raise ValueError("Batch requires 1–160 recipe/seed combinations")
                planned = []
                for seed in seeds:
                    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
                        raise ValueError("Seeds must be unsigned integers")
                    for rid in recipe_ids:
                        clean = {k: v for k, v in raw.items() if k not in {"id", "recipe_id", "condition", "conditions"}}
                        base = normalize_config(dict(clean, id=rid, seed=seed))
                        for arm in base["conditions"]:
                            for thinking in thinking_modes if thinking_modes is not None else [base["thinking"]]:
                                planned.append(dict(base, condition=arm, thinking=thinking))
                if len(planned) > 160:
                    raise ValueError("Expanded batch exceeds 160 episodes")
                random.Random(cfg["seed"] ^ 0xBA7C).shuffle(planned)
                entries = []
                for recipe in planned:
                        identifier, path = self.store.create("experiment", recipe)
                        self._claim_run(identifier)
                        entries.append(dict(run_id=identifier, out_dir=str(path), mode="experiment", config=recipe,
                            calibration_id=calibration_id, calibration_dir=str(self.store.calibration_path(calibration_id))))
                self.worker["status"] = "running"
                return self._send(command, dict(entries=entries))
            if command in {"chat", "control", "inject"} and not self.session.get("id"):
                raise ValueError("Start a session first")
            if command == "chat":
                text = payload.get("text", "")
                if not isinstance(text, str) or not text.strip() or len(text) > 24000:
                    raise ValueError("Message must contain 1–24,000 characters")
                if self.session["mode"] != "chat" or self.session["status"] != "awaiting_user":
                    raise ValueError("Wait for the current response or start a chat session")
                self.worker["status"] = "running"
            if command in {"control", "inject"} and self.session.get("mode") == "diagnostic":
                raise ValueError("Diagnostic branches use their frozen effects and separate allowance")
            if command == "control":
                payload = deepcopy(payload)
                if "aux_enabled" in payload:
                    payload["enabled"] = payload.pop("aux_enabled")
                if self.session.get("config", {}).get("recipe_version") == 2:
                    from .controller_v2 import RecipeV2Controller
                    validator = RecipeV2Controller(self.session["config"])
                else:
                    from .protocol import EffectController
                    validator = EffectController()
                try:
                    if payload.get("reset") is True and set(payload) == {"reset"}:
                        validator.reset()
                    else:
                        validator.set_controls(**payload)
                except TypeError as exc:
                    raise ValueError("Unknown control setting") from exc
                # Persist only worker-acknowledged settings, not rejected requests.
            return self._send(command, payload)

    def _start_diagnostics(self, payload):
        from .checkpoints import validate
        from .discovery import prepare_branches
        from .diagnostic_runner import validate_diagnostic_policy
        from .lifecycle import read_boundary
        from .worker import Worker
        allowed = {"run_id", "checkpoint", "calibration_id", "spec", "answer_key", "pair_id", "token_budget", "turn_token_limit", "effect_policy", "criterion_evidence"}
        if set(payload) - allowed:
            raise ValueError("Unknown diagnostic option")
        if self.session.get("status") in {"running", "awaiting_user", "generating", "paused", "queued"}:
            raise ValueError("Stop the current session before running a separate diagnostic")
        source_id = payload.get("run_id")
        source = self.store.run_path(source_id)
        checkpoint = read_boundary(source, payload.get("checkpoint", "latest"))
        calibration_id = payload.get("calibration_id") or checkpoint["session"]["calibration_id"]
        calibration_dir = self.store.calibration_path(calibration_id)
        validate(checkpoint, self.worker["model"], Worker.read_calibration_identity(calibration_dir))
        pair_id = payload.get("pair_id", "diagnostic-pair")
        design = prepare_branches(checkpoint, payload.get("spec"), payload.get("answer_key"), pair_id=pair_id,
            criterion_evidence=payload.get("criterion_evidence"))
        token_budget = payload.get("token_budget", 2048)
        turn_limit = payload.get("turn_token_limit", 256)
        policy = payload.get("effect_policy", "carry_boundary_state")
        if type(token_budget) is not int or not 1 <= token_budget <= 100000 or type(turn_limit) is not int or not 1 <= turn_limit <= 16384:
            raise ValueError("Use bounded integer diagnostic token allowances")
        if policy not in {"carry_boundary_state", "sham"}:
            raise ValueError("Unknown diagnostic effect policy")
        validate_diagnostic_policy(checkpoint, policy)
        self.resources.preflight({self.store.root: token_budget * 8192 + len(json.dumps(design).encode()) + 8 * 1024**2})
        identifier, path = self.store.create("diagnostic", checkpoint["session"]["config"])
        self.store.update(identifier, diagnostic_source=dict(run_id=source_id, checkpoint_sha256=checkpoint["sha256"]),
            diagnostic_design=dict(spec=design["spec"], pair_id=pair_id, token_budget=token_budget, turn_token_limit=turn_limit, effect_policy=policy))
        self._claim_run(identifier)
        data = dict(run_id=identifier, mode="diagnostic", config=checkpoint["session"]["config"], checkpoint=checkpoint,
            spec=design["spec"], answer_key=payload["answer_key"], pair_id=pair_id, calibration_id=calibration_id,
            calibration_dir=str(calibration_dir), out_dir=str(path), token_budget=token_budget,
            turn_token_limit=turn_limit, effect_policy=policy, criterion_evidence=payload.get("criterion_evidence"))
        self.worker["status"] = "running"
        result = self._send("run_diagnostics", data)
        return dict(result, run_id=identifier)

    def _branch_session(self, payload):
        from .checkpoints import branch, validate
        from .lifecycle import read_boundary, event_prefix
        from .worker import Worker
        if set(payload) - {"run_id", "checkpoint", "policy", "calibration_id", "action_budget", "token_budget"}:
            raise ValueError("Unknown saved-boundary setting")
        if self.session.get("status") in {"running", "awaiting_user", "paused", "queued"}:
            raise ValueError("Stop the current session before continuing a saved boundary")
        source_id = payload.get("run_id")
        source = self.store.run_path(source_id)
        checkpoint = read_boundary(source, payload.get("checkpoint", "latest"))
        calibration_id = payload.get("calibration_id") or checkpoint["session"]["calibration_id"]
        calibration_dir = self.store.calibration_path(calibration_id)
        validate(checkpoint, self.worker["model"], Worker.read_calibration_identity(calibration_dir))
        cutoff = checkpoint["session"].get("event_cutoff")
        prefix, prefix_hash = event_prefix(source, cutoff)
        ancestors = self.store.read_parent_events(source)
        ancestor_raw = b"".join((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode() for row in ancestors)
        if len(ancestor_raw) > 128 * 1024**2:
            raise ValueError("Inherited replay history exceeds the 128 MiB branch limit")
        options = dict(policy=payload.get("policy", "continue_state"), action_budget=payload.get("action_budget"),
                       token_budget=payload.get("token_budget"), parent_run_id=source_id,
                       event_cutoff=cutoff, parent_prefix_sha256=prefix_hash)
        prepared = branch(checkpoint, **options)  # Validate before allocating a run.
        provenance = prepared["session"]["branch"]
        mode, config = prepared["session"]["mode"], prepared["session"]["config"]
        self.resources.preflight({self.store.root: len(prefix) + len(ancestor_raw) + 8 * 1024**2})
        identifier, path = self.store.create(mode, config, parent=provenance)
        self._claim_run(identifier)
        try:
            if ancestors:
                guarded_bytes(path / "ancestor-events.jsonl.gz", gzip.compress(ancestor_raw, mtime=0), guard=self.resources)
                self.store.update(identifier, replay_ancestry=dict(sha256=hashlib.sha256(ancestor_raw).hexdigest(),
                    event_count=len(ancestors), source_run_id=source_id))
            raw = gzip.compress(prefix, mtime=0)
            with (path / "parent-events.jsonl.gz").open("xb") as stream:
                for offset in range(0, len(raw), self.resources.max_write_bytes):
                    chunk = raw[offset:offset + self.resources.max_write_bytes]
                    with self.resources.write(path, len(chunk)):
                        stream.write(chunk)
            prepared = branch(checkpoint, run_id=identifier, **options)
            atomic_json(path / "source-checkpoint.json", checkpoint)
            data = dict(run_id=identifier, out_dir=str(path), calibration_id=calibration_id,
                        calibration_dir=str(calibration_dir), checkpoint=prepared)
            # A restart would discard the restored history and runtime controls.
            # Branch again from its saved boundary to preserve those semantics.
            self.last_start = None
            self.last_start_run_id = None
            self.worker["status"] = "running"
            accepted = self._send("restore_session", data)
            return dict(accepted, run_id=identifier, parent=provenance)
        except Exception:
            self.store.update(identifier, status="failed", summary=dict(termination="branch_dispatch_failed"))
            raise

    def close(self):
        self.closed = True
        if self.process and self.process.poll() is None:
            try:
                self._send("stop", {})
                self.process.stdin.close()
                self.process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
