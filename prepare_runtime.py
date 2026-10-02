#!/usr/bin/env python3
"""Run an explicit installer/download command with selected storage and supervision.

No command runs without --execute. Hugging Face and pip are offline by default;
--allow-network enables their normal network behavior. This routes common caches
and monitors an owned child; it is not an OS quota or a network sandbox for
arbitrary executables. Existing environments and partial downloads are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading

from lab.resources import EmergencyMetadata, ResourceGuard, ResourceStop, cancel_owned_process, supervise_process
from lab.storage import atomic_json, guarded_bytes, new_id, utc_now

GIB = 2**30


def command_plan(command, *, environment_dir, cache_dir, temp_dir, planned_bytes, allow_network=False):
    if not isinstance(command, list) or not command or any(type(value) is not str or not value or "\0" in value for value in command):
        raise ValueError("Supply one explicit command argv; shell command strings are unsupported")
    if not isinstance(planned_bytes, dict) or set(planned_bytes) != {"environment", "cache", "temp"} or any(type(value) is not int or value < 0 for value in planned_bytes.values()):
        raise ValueError("Declare nonnegative environment/cache/temp byte estimates")
    if type(allow_network) is not bool:
        raise ValueError("allow_network must be boolean")
    paths = {name:str(Path(path).expanduser().resolve()) for name,path in
             (("environment",environment_dir),("cache",cache_dir),("temp",temp_dir))}
    if any(Path(path) == Path(Path(path).anchor) for path in paths.values()):
        raise ValueError("Select dedicated storage directories")
    value=dict(schema_version=1, command=command.copy(), destinations=paths, planned_bytes=dict(planned_bytes),
               allow_network=allow_network, shell=False,
               policy="Owned child monitoring and common-cache routing; no OS quota or arbitrary-command network sandbox")
    value["sha256"]=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return value


def child_environment(plan, base=None):
    env=dict(os.environ if base is None else base)
    paths=plan["destinations"];cache=Path(paths["cache"])
    executable_dir=Path(paths["environment"])/("Scripts" if os.name=="nt" else "bin")
    env.update(VIRTUAL_ENV=paths["environment"],PATH=str(executable_dir)+os.pathsep+env.get("PATH",""),
        HF_HOME=str(cache/"huggingface"),HF_HUB_CACHE=str(cache/"huggingface"/"hub"),
        PIP_CACHE_DIR=str(cache/"pip"),TORCH_HOME=str(cache/"torch"),XDG_CACHE_HOME=str(cache/"xdg"),
        TRITON_CACHE_DIR=str(cache/"triton"),CUDA_CACHE_PATH=str(cache/"cuda"),TORCHINDUCTOR_CACHE_DIR=str(cache/"inductor"),
        TMPDIR=paths["temp"],TMP=paths["temp"],TEMP=paths["temp"],
        HF_HUB_DISABLE_XET="1",HF_HUB_ENABLE_HF_TRANSFER="0",HF_ENABLE_PARALLEL_LOADING="false",
        HF_HUB_DISABLE_PROGRESS_BARS="1",PYTHONUNBUFFERED="1",PYTHONDONTWRITEBYTECODE="1")
    for key in ("PIP_NO_INDEX","HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE"):
        if plan["allow_network"]:env.pop(key,None)
        else:env[key]="1"
    return env


def run_command(plan, *, record_dir=None, guard=None, cancel_event=None, poll_seconds=.25, log_limit=8*1024**2):
    if not isinstance(plan, dict) or not isinstance(plan.get("destinations"), dict):
        raise ValueError("Expected a reviewed command plan")
    resolved=command_plan(plan.get("command"),environment_dir=plan.get("destinations",{}).get("environment"),
        cache_dir=plan.get("destinations",{}).get("cache"),temp_dir=plan.get("destinations",{}).get("temp"),
        planned_bytes=plan.get("planned_bytes"),allow_network=plan.get("allow_network"))
    if resolved != plan:
        raise ValueError("Command plan changed after its storage review")
    if type(log_limit) is not int or not 0 <= log_limit <= 64*1024**2:
        raise ValueError("Log limit must lie between zero and 64 MiB")
    paths=plan["destinations"]
    record_dir=Path(record_dir or Path(paths["environment"]).parent/"opium-runtime-jobs").resolve()
    guard=guard or ResourceGuard({**paths,"records":record_dir})
    identifier=new_id("prepare");emergency=None;process=None;reader=None;errors=[]
    log_path=record_dir/(identifier+".log")
    result=dict(status="failed",command_plan_sha256=plan["sha256"],started_at=utc_now(),log=str(log_path))
    counts={"written":0,"discarded":0}
    try:
        emergency=EmergencyMetadata(record_dir).allocate(guard)
        writes={}
        for name,path in paths.items():writes[path]=writes.get(path,0)+plan["planned_bytes"][name]
        guard.preflight(writes)
        burst={path:256*1024**2 for path in set(paths.values())}
        guard.reserve(identifier,burst)
        Path(paths["temp"]).mkdir(parents=True,exist_ok=True)
        Path(paths["cache"]).mkdir(parents=True,exist_ok=True)
        atomic_json(record_dir/(identifier+"-plan.json"),plan,guard=guard)
        if cancel_event is not None and cancel_event.is_set():
            result.update(status="cancelled",reason="cancelled_before_start",returncode=None)
            return result
        process=subprocess.Popen(plan["command"],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            env=child_environment(plan),shell=False,start_new_session=os.name!="nt")
        def drain():
            while data:=process.stdout.read(16384):
                allowed=min(len(data),log_limit-counts["written"])
                if allowed and not errors:
                    try:
                        guarded_bytes(log_path,data[:allowed],mode="ab",guard=guard)
                        counts["written"]+=allowed
                    except (ResourceStop,OSError,ValueError) as exc:errors.append(exc)
                counts["discarded"]+=len(data)-allowed if not errors else len(data)
        reader=threading.Thread(target=drain,daemon=True);reader.start()
        outcome=supervise_process(process,guard,cancel_event=cancel_event,operation=identifier,
                                  poll_seconds=poll_seconds,process_group=os.name!="nt")
        reader.join(timeout=3)
        if reader.is_alive():
            raise RuntimeError("Owned child log stream remained open after exit")
        result.update(outcome)
        if errors and outcome["status"]=="complete":
            if isinstance(errors[0],ResourceStop):raise errors[0]
            raise OSError(str(errors[0]))
    except KeyboardInterrupt:
        if process:cancel_owned_process(process,process_group=os.name!="nt")
        result.update(status="cancelled",reason="interrupted_by_user",returncode=process.returncode if process else None)
    except ResourceStop as exc:
        if process:cancel_owned_process(process,process_group=os.name!="nt")
        result.update(exc.to_dict(),returncode=process.returncode if process else None)
    except (OSError,ValueError,RuntimeError) as exc:
        if process:cancel_owned_process(process,process_group=os.name!="nt")
        result.update(status="failed",reason=str(exc),returncode=process.returncode if process else None)
    finally:
        guard.release(identifier)
        if reader:reader.join(timeout=3)
        if process and process.stdout:process.stdout.close()
        result.update(finished_at=utc_now(),log_bytes=counts["written"],discarded_log_bytes=counts["discarded"])
        if emergency:
            try:
                if result["status"]=="resource_stopped":
                    result["record"]=str(emergency.finalize(result))
                else:
                    try:
                        target=record_dir/(identifier+"-result.json");atomic_json(target,result,guard=guard);result["record"]=str(target)
                    except ResourceStop as exc:
                        result.update(exc.to_dict());result["record"]=str(emergency.finalize(result))
            except (OSError,ValueError) as exc:
                result["record_error"]=str(exc)
            finally:
                emergency.release()
    return result


def _gib(value):
    value=float(value)
    if not math.isfinite(value) or not 0 <= value <= 100000:raise argparse.ArgumentTypeError("GiB must be finite and nonnegative")
    return int(value*GIB)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-dir",required=True,type=Path)
    parser.add_argument("--cache-dir",required=True,type=Path)
    parser.add_argument("--temp-dir",required=True,type=Path)
    parser.add_argument("--environment-gib",required=True,type=_gib)
    parser.add_argument("--cache-gib",required=True,type=_gib)
    parser.add_argument("--temp-gib",required=True,type=_gib)
    parser.add_argument("--record-dir",type=Path)
    parser.add_argument("--allow-network",action="store_true")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("command",nargs=argparse.REMAINDER,help="Explicit argv after --; no shell expansion")
    args=parser.parse_args();command=args.command[1:] if args.command[:1]==["--"] else args.command
    try:
        plan=command_plan(command,environment_dir=args.environment_dir,cache_dir=args.cache_dir,temp_dir=args.temp_dir,
            planned_bytes=dict(environment=args.environment_gib,cache=args.cache_gib,temp=args.temp_gib),allow_network=args.allow_network)
        result=run_command(plan,record_dir=args.record_dir) if args.execute else plan
    except (ValueError,TypeError,OSError) as exc:parser.error(str(exc))
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0 if not args.execute or result["status"]=="complete" else 130 if result["status"]=="cancelled" else 2


if __name__=="__main__":raise SystemExit(main())
