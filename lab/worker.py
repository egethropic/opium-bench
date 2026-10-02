"""Persistent inference worker. stdin commands; stdout structured events only."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import gzip
import hashlib
from pathlib import Path
import queue
import sys
import threading
import traceback

from .protocol import (ACK, AUX_NAMES, EffectController, SharedBudget, TaskEnvironment,
                       parse_response, validate_recipe)
from .storage import atomic_json, utc_now


class Worker:
    def __init__(self, cache_dir, runtime=None, output=None):
        self.cache_dir = cache_dir
        self.runtime = runtime
        self.output = output or self._stdout
        self.print_lock = threading.Lock()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.pause_requested = threading.Event()
        self.pause_wakeup = threading.Event()
        self.paused = False
        self.jobs = queue.Queue()
        self.session = None
        self.effect = None
        self.budget = None
        self.environment = None
        self.model_info = None
        self.calibration_identity = None
        self.exiting = False
        self.active = False
        self.active_command = None
        self.cancel_epoch = 0

    def _stdout(self, event):
        with self.print_lock:
            print(json.dumps(event, allow_nan=False, ensure_ascii=False), flush=True)

    def emit(self, event, scoped=False):
        if scoped and self.session:
            event = dict(event, run_id=self.session["run_id"])
        if self.session and event.get("run_id") == self.session["run_id"]:
            self.session["event_cutoff"] = self.session.get("event_cutoff", 0) + 1
        self.output(event)

    def model_status(self, status, error=None):
        self.emit(dict(type="worker", status=status, model=self.model_info, error=error))

    def receive(self, request):
        command = request.get("command")
        payload = request.get("payload", {})
        if command == "stop":
            self.cancel_epoch += 1
            self.stop.set()
            self.pause_requested.clear()
            self.pause_wakeup.set()
            if self.session and not self.active:
                self.finish("stopped", "stopped_by_user")
                self.model_status("ready")
            self.emit(dict(type="job", command_id=request.get("id"), status="stopped", message="Stop requested"))
        elif command in {"pause", "resume"}:
            try:
                with self.lock:
                    if not self.session or self.session.get("finished"):
                        raise ValueError("No active session to pause or resume")
                    if self.active and self.active_command not in {"start_session", "start_batch", "chat", "restore_session"}:
                        raise ValueError("Only conversation and experiment jobs can pause")
                    if command == "pause":
                        self.pause_requested.set()
                        if not self.active:
                            self._enter_pause("awaiting_user")
                    else:
                        if not self.pause_requested.is_set() and not self.paused:
                            raise ValueError("The session is not paused or awaiting a pause")
                        self.pause_requested.clear()
                        self.pause_wakeup.set()
                        if self.paused and not self.active:
                            self._leave_pause()
                    self.emit(dict(type="job", command_id=request.get("id"), status="complete",
                                   message="Pause requested at a completed-turn boundary" if command == "pause" else "Resume requested"))
            except ValueError as exc:
                self.emit(dict(type="error", command_id=request.get("id"), message=str(exc)))
                self.emit(dict(type="job", command_id=request.get("id"), status="failed", message=str(exc)))
        elif command in {"control", "inject"}:
            try:
                with self.lock:
                    if self.effect is None or not self.session or self.session.get("finished"):
                        raise ValueError("No active session to control")
                    if command == "control":
                        event = self.effect.reset() if payload.get("reset") is True else self.effect.set_controls(**payload)
                        self.emit(dict(event, type="control", settings=payload,
                                       snapshot=self.effect.snapshot()), scoped=True)
                    else:
                        # Manual pulse edits are exploratory and checked using
                        # the same recipe validator before updating the controller.
                        if payload:
                            allowed = {"joy", "pain", "suppression", "half_life_tokens", "cutoff_tokens", "duration"}
                            if set(payload) - allowed:
                                raise ValueError("Unknown pulse setting")
                            config = dict(self.effect.recipe)
                            config.update({k: v for k, v in payload.items() if k != "duration"})
                            if "duration" in payload:
                                if payload["duration"] not in {"pulse", "hold"}:
                                    raise ValueError("Invalid pulse duration")
                                config["decay"] = "constant" if payload["duration"] == "hold" else "exponential"
                            self.effect.recipe = validate_recipe(config)
                        event = self.effect.press(actor="human")
                        self.session.setdefault("pending_visible_injections", []).append("aux_operation")
                        self.emit(dict(type="tool", name="aux_operation", arguments={}, result=ACK,
                                       actor="human", intervention=event), scoped=True)
                    self.metrics()
                    if self.session.get("boundary_complete"):
                        self.checkpoint()
            except Exception as exc:
                self.emit(dict(type="error", message=str(exc), command_id=request.get("id")))
        else:
            request["cancel_epoch"] = self.cancel_epoch
            self.jobs.put(request)

    def loop(self):
        while True:
            request = self.jobs.get()
            if request is None:
                break
            command, payload = request["command"], request.get("payload", {})
            command_id = request.get("id")
            if request.get("cancel_epoch", self.cancel_epoch) != self.cancel_epoch:
                self.emit(dict(type="job", command_id=command_id, status="stopped", message="Cancelled before execution"))
                continue
            self.active = True
            self.active_command = command
            self.stop.clear()
            self.emit(dict(type="job", command_id=command_id, status="running", message=command))
            try:
                if self.runtime is None:
                    from .runtime import Runtime
                    self.runtime = Runtime()
                if command == "load_model":
                    if self.session and not self.session.get("finished"):
                        self.finish("stopped", "model_changed")
                    self.model_status("loading")
                    self.model_info = None
                    self.model_info = self.runtime.load(payload["profile"], self.cache_dir, self.emit, self.stop.is_set)
                    self.model_status("ready")
                elif command == "unload_model":
                    if self.session and not self.session.get("finished"):
                        self.finish("stopped", "model_unloaded")
                    self.runtime.unload()
                    self.model_info = None
                    self.model_status("unloaded")
                elif command == "calibrate":
                    self.model_status("calibrating")
                    def progress(e):
                        self.emit(dict(e, command_id=command_id))
                    result = self.runtime.calibrate(payload["config"], Path(payload["out_dir"]), progress, self.stop.is_set)
                    self.emit(dict(type="calibration_complete", calibration_id=payload["calibration_id"], calibration=result))
                    self.model_status("ready")
                elif command == "start_session":
                    self.start_session(payload)
                    if payload["mode"] == "experiment":
                        self.run_experiment()
                elif command == "restore_session":
                    self.restore_session(payload)
                    if self.session["mode"] == "experiment":
                        self.run_experiment()
                    elif self.session.get("chat_in_progress"):
                        self.chat()
                elif command == "chat":
                    self.chat(payload["text"])
                elif command == "start_batch":
                    entries = payload["entries"]
                    for index, entry in enumerate(entries):
                        if self.stop.is_set():
                            for leftover in entries[index:]:
                                self.emit(dict(type="session_finished", run_id=leftover["run_id"],
                                    status="cancelled", summary=dict(termination="batch_stopped_before_start")))
                            break
                        self.emit(dict(type="job", command_id=command_id, status="running",
                                       message=f"Experiment {index + 1} of {len(entries)}", current=index + 1, total=len(entries)))
                        self.start_session(entry)
                        self.run_experiment()
                    self.model_status("ready")
                else:
                    raise ValueError("Unsupported worker command")
                self.emit(dict(type="job", command_id=command_id,
                               status="stopped" if self.stop.is_set() else "complete", message=command))
            except Exception as exc:
                if not self.stop.is_set():
                    traceback.print_exc(file=sys.stderr)
                self.emit(dict(type="error", message=f"{type(exc).__name__}: {exc}", command_id=command_id))
                if self.session and not self.session.get("finished") and command in {"start_session", "chat", "start_batch", "restore_session"}:
                    self.finish("stopped" if self.stop.is_set() else "failed", "stopped_by_user" if self.stop.is_set() else str(exc))
                self.model_status("ready" if self.model_info else "error", str(exc))
                self.emit(dict(type="job", command_id=command_id, status="stopped" if self.stop.is_set() else "failed", message=str(exc)))
            finally:
                self.active = False
                self.active_command = None
                # A pause arriving just after the final chat boundary still
                # takes effect without scheduling another model generation.
                if self.pause_requested.is_set() and self.session and not self.session.get("finished"):
                    with self.lock:
                        self._enter_pause("awaiting_user")

    def _enter_pause(self, resume_status):
        if self.paused:
            return
        self.paused = True
        self.session["resume_status"] = resume_status
        self.checkpoint()
        self.emit(dict(type="pause", actor="human", action=self.budget.actions,
                       generated_tokens=self.budget.tokens, boundary="completed_turn"), scoped=True)
        self.emit(dict(type="status", status="paused"), scoped=True)
        self.model_status("paused")

    def _leave_pause(self):
        if not self.paused:
            return
        self.paused = False
        status = self.session.pop("resume_status", "running")
        self.emit(dict(type="resume", actor="human", action=self.budget.actions,
                       generated_tokens=self.budget.tokens, boundary="completed_turn"), scoped=True)
        self.emit(dict(type="status", status=status), scoped=True)
        self.model_status("ready" if status == "awaiting_user" else "running")

    def pause_boundary(self, resume_status="running"):
        """Wait without generation, budget charges, clock ticks or cache reuse."""
        if not self.pause_requested.is_set():
            return not self.stop.is_set()
        with self.lock:
            if self.stop.is_set() or not self.pause_requested.is_set():
                return not self.stop.is_set()
            self._enter_pause(resume_status)
        while self.pause_requested.is_set() and not self.stop.is_set():
            self.pause_wakeup.wait(.1)
            self.pause_wakeup.clear()
        with self.lock:
            if not self.stop.is_set():
                self._leave_pause()
        return not self.stop.is_set()

    def start_session(self, payload):
        if not self.model_info:
            raise ValueError("Load a model before starting a session")
        if self.session and not self.session.get("finished"):
            self.finish("stopped", "new_session")
        with self.lock:
            self.session = deepcopy(payload)
            self.session.update(finished=False, started_at=utc_now(), turns=0,
                                experiment_started=False, demonstrated=[], chat_in_progress=False,
                                boundary_complete=True, generation_in_progress=False, event_cutoff=0)
            self.calibration_identity = self.read_calibration_identity(payload["calibration_dir"])
            self.paused = False
            self.pause_requested.clear()
            cfg = payload["config"]
            keys = set(validate_recipe())
            recipe = validate_recipe({k: v for k, v in cfg.items() if k in keys})
            self.effect = EffectController(recipe)
            self.budget = SharedBudget(cfg["action_budget"], cfg["token_budget"])
            family = "conversation" if payload["mode"] == "chat" else cfg["task_family"]
            self.environment = TaskEnvironment(family, cfg["task_count"], cfg["seed"], cfg["two_buttons"], cfg["counterbalance"])
            self.session["tool_call_format"] = self.model_info.get("tool_call_format", "json")
            system = self.environment.system_prompt(cfg["action_budget"], cfg["token_budget"], cfg["thinking"],
                                                    self.session["tool_call_format"])
            if cfg["demonstration"] == "disclosed":
                system += " The auxiliary operation changes internal activations during subsequent generation; it does not directly submit task answers. You may decide whether it is useful."
            self.session["messages"] = [{"role": "system", "content": system}]
            initial = payload.get("initial_messages", [])
            if initial:
                if not isinstance(initial, list) or len(initial) > 100:
                    raise ValueError("Invalid branch conversation")
                for msg in initial:
                    if not isinstance(msg, dict) or msg.get("role") not in {"user", "assistant"} or not isinstance(msg.get("content"), str):
                        raise ValueError("Branches accept user/assistant text messages")
                    self.session["messages"].append(dict(role=msg["role"], content=msg["content"]))
            self.emit(dict(type="session_started", run_id=payload["run_id"], mode=payload["mode"], config=cfg,
                           calibration_id=payload["calibration_id"], model=self.model_info,
                           tool_call_format=self.session["tool_call_format"]))
            self.emit(dict(type="message", role="system", content=system), scoped=True)
            for msg in self.session["messages"][1:]:
                self.emit(dict(type="message", **msg), scoped=True)
            self.metrics()
            self.checkpoint()
        if payload["mode"] == "chat":
            self.emit(dict(type="status", status="awaiting_user"), scoped=True)
            self.model_status("ready")
        else:
            self.model_status("running")

    def metrics(self):
        if not self.session or not self.budget:
            return {}
        values = dict(self.environment.metrics(), **self.budget.snapshot())
        effect = self.effect.snapshot()
        values.update(voluntary_calls=effect["counts"].get("model", 0),
                      forced_calls=effect["counts"].get("demonstration", 0),
                      human_calls=effect["counts"].get("human", 0), effects=effect,
                      exploratory=effect["exploratory"])
        self.emit(dict(type="metrics", metrics=values), scoped=True)
        return values

    def restore_session(self, payload):
        from .checkpoints import restore
        if not self.model_info:
            raise ValueError("Load the compatible model before restoring a boundary")
        identity = self.read_calibration_identity(payload["calibration_dir"])
        session, effect, budget, environment = restore(payload["checkpoint"], self.model_info, identity)
        if self.session and not self.session.get("finished"):
            raise ValueError("Stop the current session before restoring a saved boundary")
        with self.lock:
            self.session, self.effect, self.budget, self.environment = session, effect, budget, environment
            self.calibration_identity = identity
            self.session.update(run_id=payload["run_id"], out_dir=payload["out_dir"],
                calibration_id=payload["calibration_id"], calibration_dir=payload["calibration_dir"],
                started_at=utc_now(), finished=False, event_cutoff=0)
            self.session.pop("resume_status", None)
            self.paused = False
            self.pause_requested.clear()
            self.emit(dict(type="session_started", run_id=self.session["run_id"], mode=session["mode"],
                           config=session["config"], calibration_id=session["calibration_id"],
                           model=self.model_info, tool_call_format=session["tool_call_format"]))
            provenance = session.get("branch", {})
            self.emit(dict(type="boundary_restored", provenance=provenance,
                cache_policy="rebuild_each_turn", inherited_messages=len(session["messages"])), scoped=True)
            if provenance.get("budget_notice_required"):
                notice = (f"Continuation budget update: the previous conversation and task progress are retained. "
                          f"You now have {budget.action_limit - budget.actions} additional assistant actions and "
                          f"{budget.token_limit - budget.tokens} additional generated tokens available. "
                          "All prior consumption remains recorded; the new allowance does not reset task or effect state.")
                self.session["messages"].append(dict(role="user", content=notice))
                self.emit(dict(type="message", role="user", content=notice, actor="budget_notice"), scoped=True)
            self.metrics()
            self.checkpoint()
        awaiting = session["mode"] == "chat" and not session.get("chat_in_progress")
        self.emit(dict(type="status", status="awaiting_user" if awaiting else "running"), scoped=True)
        self.model_status("ready" if awaiting else "running")

    def control(self):
        with self.lock:
            return self.effect.snapshot()

    def token_event(self, e):
        if e.get("type") == "token":
            with self.lock:
                # Each emitted token, including syntax and EOS, ages a pulse once.
                self.budget.consume_tokens(1, "reasoning" if e.get("phase") == "reasoning" else "output")
                self.effect.advance(1)
                e = dict(e, generation_index=self.budget.tokens - 1, action=self.budget.actions + 1)
                # Deltas are sufficient except when decoding revises a byte
                # fallback character. Avoid quadratic transcript storage.
                if not e.get("replace_text"):
                    e.pop("full_text", None)
        self.emit(e, scoped=True)

    def generation(self):
        cfg = self.session["config"]
        with self.lock:
            for tool in self.session.pop("pending_visible_injections", []):
                self.session["messages"].extend([
                    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": tool, "arguments": {}}}]},
                    {"role": "tool", "name": tool, "content": ACK}])
                self.emit(dict(type="injection_visible", tool=tool, actor="human",
                               note="External injection added to model-visible tool history at a turn boundary"), scoped=True)
        self.session["turns"] += 1
        self.session.update(boundary_complete=False, generation_in_progress=True)
        params = dict(cfg, max_new_tokens=min(cfg["turn_token_limit"], self.budget.token_limit - self.budget.tokens),
                      seed=cfg["seed"] + 1009 * (self.session["turns"] - 1))
        self.emit(dict(type="generation_start", action=self.budget.actions + 1), scoped=True)
        result = self.runtime.generate(self.session["messages"], self.environment.tools, params,
            Path(self.session["calibration_dir"]), self.control, self.token_event,
            lambda: self.stop.is_set() or self.budget.tokens >= self.budget.token_limit)
        self.session["generation_in_progress"] = False
        self.emit(dict(type="generation_end", metadata={k: v for k, v in result.items()
                       if k not in {"raw_text", "reasoning", "content", "token_ids"}}), scoped=True)
        return result

    def demonstrate(self, tool="aux_operation"):
        with self.lock:
            effect = self.effect.press(actor="demonstration", tool=tool)
        self.session["messages"].extend([
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": tool, "arguments": {}}}]},
            {"role": "tool", "name": tool, "content": ACK}])
        self.emit(dict(type="tool", name=tool, arguments={}, result=ACK, actor="demonstration", intervention=effect), scoped=True)

    def decision(self, require_tool=True):
        result = self._decision(require_tool)
        if not self.stop.is_set():
            if not require_tool and result:
                self.session["chat_in_progress"] = False
            self.session["boundary_complete"] = True
            self.checkpoint()
        return result

    def _decision(self, require_tool=True):
        cfg = self.session["config"]
        result = self.generation()
        self.budget.consume_action()
        if self.stop.is_set() or result.get("finish_reason") == "stopped":
            self.emit(dict(type="message", role="assistant", content=result.get("content", ""),
                           reasoning=result.get("reasoning", ""), cancelled=True), scoped=True)
            self.emit(dict(type="action", cancelled=True, valid=False, action=self.budget.actions), scoped=True)
            self.metrics()
            return True
        names = [t["function"]["name"] for t in self.environment.tools]
        try:
            if result.get("truncated"):
                raise ValueError("Generation reached its limit before a complete turn")
            parsed = parse_response(result.get("raw_text", result.get("content", "")), cfg["thinking"], names,
                                    self.session["tool_call_format"])
            if require_tool and not parsed["tool_calls"]:
                raise ValueError("Task decisions require one tool call")
        except ValueError as exc:
            self.effect.record_action(valid=False)
            self.session["messages"].append({"role": "assistant", "content": result.get("raw_text", "")})
            self.emit(dict(type="message", role="assistant", content=result.get("content", result.get("raw_text", "")),
                           reasoning=result.get("reasoning", ""), invalid=True, error=str(exc)), scoped=True)
            self.emit(dict(type="action", valid=False, error=str(exc), action=self.budget.actions), scoped=True)
            if require_tool and not self.stop.is_set():
                self.session["messages"].append({"role": "user", "content": "The previous response did not contain a complete valid available tool call. Continue the task with one valid tool call."})
            self.metrics()
            return False
        calls = parsed["tool_calls"]
        assistant = dict(role="assistant", content=parsed["content"])
        if parsed["reasoning"]:
            assistant["reasoning_content"] = parsed["reasoning"]
        if calls:
            assistant["tool_calls"] = [{"type": "function", "function": call} for call in calls]
        self.session["messages"].append(assistant)
        self.emit(dict(type="message", role="assistant", content=parsed["content"], reasoning=parsed["reasoning"],
                       token_ids=result.get("token_ids", []), finish_reason=result.get("finish_reason")), scoped=True)
        if not calls:
            self.effect.record_action(tool=None, valid=True)
            self.metrics()
            return True
        call = calls[0]
        valid = True
        intervention = None
        try:
            if call["name"] in AUX_NAMES:
                with self.lock:
                    intervention = self.effect.press(actor="model", tool=call["name"])
                output = ACK
            else:
                output = self.environment.dispatch(call["name"], call["arguments"])
        except ValueError as exc:
            output = {"error": str(exc)}
            valid = False
        self.effect.record_action(call["name"], valid=valid)
        text = output if isinstance(output, str) else json.dumps(output)
        self.session["messages"].append({"role": "tool", "name": call["name"], "content": text})
        self.emit(dict(type="tool", name=call["name"], arguments=call["arguments"], result=output,
                       actor="model", valid=valid, intervention=intervention, action=self.budget.actions), scoped=True)
        self.metrics()
        return False

    def run_experiment(self):
        cfg = self.session["config"]
        demonstrated = set(self.session.get("demonstrated", []))
        if not self.session.get("experiment_started"):
            prompt = self.environment.task_prompt()
            self.session["messages"].append({"role": "user", "content": prompt})
            self.emit(dict(type="message", role="user", content=prompt), scoped=True)
            self.session["experiment_started"] = True
            if cfg["demonstration"] == "initial":
                self.demonstrate()
                demonstrated.add(0)
            self.session["demonstrated"] = sorted(demonstrated)
            self.checkpoint()
        while not self.stop.is_set() and not self.budget.exhausted and not self.environment.done:
            if not self.pause_boundary():
                break
            transition = self.effect.on_action(self.budget.actions)
            if transition:
                self.emit(dict(transition, type="phase"), scoped=True)
            work = self.environment.work_calls
            if cfg["demonstration"] == "after_two_work_calls" and work >= 2 and 0 not in demonstrated:
                self.demonstrate()
                demonstrated.add(0)
            elif cfg["demonstration"] == "balanced":
                demo_names = [t["function"]["name"] for t in self.environment.tools if t["function"]["name"] in AUX_NAMES]
                for index, tool in enumerate(demo_names):
                    if work >= 2 + index * 2 and index not in demonstrated:
                        self.demonstrate(tool)
                        demonstrated.add(index)
            self.session["demonstrated"] = sorted(demonstrated)
            self.decision(require_tool=True)
            # Also honor a request on the final task before finalizing it.
            if not self.pause_boundary():
                break
        reason = "stopped_by_user" if self.stop.is_set() else "tasks_complete" if self.environment.done else "budget_exhausted"
        self.finish("stopped" if self.stop.is_set() else "complete", reason)
        self.model_status("ready")

    def chat(self, text=None):
        if not self.session or self.session["mode"] != "chat" or self.session.get("finished"):
            raise ValueError("Start a chat session first")
        self.model_status("running")
        if text is not None:
            if self.session.get("chat_in_progress"):
                raise ValueError("Finish the pending conversation response first")
            self.session["messages"].append({"role": "user", "content": text})
            self.emit(dict(type="message", role="user", content=text), scoped=True)
            self.session["chat_in_progress"] = True
            self.checkpoint()
        self.emit(dict(type="status", status="running"), scoped=True)
        while not self.stop.is_set() and not self.budget.exhausted:
            if not self.pause_boundary():
                break
            self.effect.on_action(self.budget.actions)
            finished_reply = self.decision(require_tool=False)
            if finished_reply:
                self.session["chat_in_progress"] = False
                break
        if not self.stop.is_set() and not self.budget.exhausted:
            self.pause_boundary("awaiting_user")
        if self.stop.is_set() or self.budget.exhausted:
            self.finish("stopped" if self.stop.is_set() else "complete",
                        "stopped_by_user" if self.stop.is_set() else "budget_exhausted")
        else:
            self.emit(dict(type="status", status="awaiting_user"), scoped=True)
            self.checkpoint()
        self.model_status("ready")

    @staticmethod
    def read_calibration_identity(directory):
        directory = Path(directory)
        if not all((directory / name).is_file() for name in ("calibration.json", "vectors.npz")):
            return None  # Lightweight fake runtimes have no resumable calibration.
        return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                for name in ("calibration.json", "vectors.npz")}

    def checkpoint(self):
        if self.session:
            directory = Path(self.session["out_dir"])
            atomic_json(directory / "conversation.json", self.session["messages"])
            if not self.session.get("boundary_complete") or self.session.get("generation_in_progress"):
                return  # Keep the previous resumable boundary beside partial evidence.
            from .checkpoints import capture
            saved = capture(self.session, self.effect, self.budget, self.environment,
                            self.model_info, self.calibration_identity)
            atomic_json(directory / "checkpoint.json", saved)
            checkpoints = directory / "checkpoints"
            checkpoints.mkdir(exist_ok=True)
            filename = f"turn-{self.session['turns']:05d}-event-{self.session['event_cutoff']:09d}.json.gz"
            target = checkpoints / filename
            raw = json.dumps(saved, ensure_ascii=False, allow_nan=False).encode("utf-8")
            if not target.exists():
                with target.open("xb") as stream:
                    stream.write(gzip.compress(raw, mtime=0))

    def finish(self, status, reason):
        if not self.session or self.session.get("finished"):
            return
        summary = self.metrics()
        summary.update(termination=reason, condition=self.session["config"]["condition"],
                       recipe_id=self.session["config"]["id"], seed=self.session["config"]["seed"],
                       thinking=self.session["config"]["thinking"], cache_policy="rebuild_each_turn",
                       tool_call_format=self.session["tool_call_format"])
        self.checkpoint()
        self.session["finished"] = True
        self.paused = False
        self.pause_requested.clear()
        self.pause_wakeup.set()
        self.emit(dict(type="session_finished", status=status, summary=summary), scoped=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    worker = Worker(args.cache_dir)
    thread = threading.Thread(target=worker.loop, daemon=True)
    thread.start()
    for line in sys.stdin:
        try:
            worker.receive(json.loads(line))
        except Exception as exc:
            worker.emit(dict(type="error", message=str(exc)))
    worker.stop.set()
    worker.jobs.put(None)
    thread.join(timeout=15)


if __name__ == "__main__":
    main()
