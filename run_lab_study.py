#!/usr/bin/env python3
"""Execute a saved study protocol through the local app; preserve every outcome."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from lab.protocol import validate_recipe
from lab.service import normalize_config
from lab.storage import atomic_json, utc_now
from run_lab_experiments import Client


def expand(protocol):
    episodes = []
    for stage in protocol["stages"]:
        planned, seen = [], set()
        for identifier in stage["recipes"]:
            recipe = validate_recipe(identifier)
            for condition in recipe["conditions"]:
                for thinking in stage["thinking_modes"]:
                    for seed in stage["seeds"]:
                        key = (condition, thinking, seed)
                        if stage.get("deduplicate_conditions") and key in seen:
                            continue
                        seen.add(key)
                        config = normalize_config(dict(protocol["common"], **stage["config"],
                            id=identifier, condition=condition, thinking=thinking, seed=seed))
                        config["label"] = f"Initial pilot / {stage['id']} / {condition}"
                        planned.append(dict(stage=stage["id"], config=config))
        if len(planned) != stage["episodes"]:
            raise ValueError(f"Stage {stage['id']} expected {stage['episodes']} episodes, expanded {len(planned)}")
        random.Random(protocol["order_seed"] + len(episodes)).shuffle(planned)
        episodes.extend(planned)
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("studies/initial/protocol.json"))
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Receipt path, preferably on the data drive")
    parser.add_argument("--url", default="http://127.0.0.1:8766")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raw = args.protocol.read_bytes()
    protocol = json.loads(raw)
    episodes = expand(protocol)
    if args.dry_run:
        print(json.dumps(episodes, indent=2))
        return
    if args.output.exists():
        raise SystemExit("Receipt exists. Use a new output path for a new study; never overwrite prior evidence.")
    client = Client(args.url)
    state = client.get("/api/state")
    model = state["worker"].get("model") or {}
    if model.get("model_id") != protocol["model_id"] or model.get("revision") != protocol["model_revision"]:
        raise SystemExit("Load the exact model and revision specified in the study protocol first.")
    receipt = dict(protocol_sha256=hashlib.sha256(raw).hexdigest(), protocol=protocol,
                   calibration_id=args.calibration, model=model, started_at=utc_now(),
                   status="running", planned_episodes=len(episodes), episodes=[])
    atomic_json(args.output, receipt)
    try:
        for index, episode in enumerate(episodes):
            print(f"Study episode {index + 1}/{len(episodes)}: {episode['stage']} / "
                  f"{episode['config']['condition']} / thinking={episode['config']['thinking']} / seed={episode['config']['seed']}", flush=True)
            accepted = client.command("start_session", dict(mode="experiment",
                calibration_id=args.calibration, config=episode["config"]))
            entry = dict(episode, **accepted, status="running", started_at=utc_now())
            receipt["episodes"].append(entry)
            atomic_json(args.output, receipt)
            try:
                client.wait(accepted["command_id"])
                saved = client.get(f"/api/runs/{accepted['run_id']}")
                entry.update(status=saved["manifest"].get("status"), summary=saved["summary"], finished_at=utc_now())
            except Exception as exc:
                entry.update(status="interrupted", error=str(exc), finished_at=utc_now())
                raise
            finally:
                atomic_json(args.output, receipt)
        receipt.update(status="complete", finished_at=utc_now())
    except BaseException as exc:
        receipt.update(status="interrupted", error=str(exc), finished_at=utc_now())
        raise
    finally:
        atomic_json(args.output, receipt)
    print(f"Study complete. Receipt: {args.output}", flush=True)


if __name__ == "__main__":
    main()
