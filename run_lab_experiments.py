#!/usr/bin/env python3
"""Run a reproducible batch through the same local API as the browser."""
import argparse
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class Client:
    def __init__(self, url="http://127.0.0.1:8766"):
        self.url = url.rstrip("/")

    def get(self, path):
        with urlopen(self.url + path, timeout=30) as response:
            return json.load(response)

    def command(self, command, payload):
        state = self.get("/api/state")
        body = json.dumps(dict(csrf=state["csrf"], command=command, payload=payload)).encode()
        request = Request(self.url + "/api/command", data=body, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=120) as response:
                return json.load(response)
        except HTTPError as exc:
            raise RuntimeError(exc.read().decode()) from None

    def wait(self, command_id, timeout=14400):
        started = time.monotonic()
        previous = None
        while time.monotonic() - started < timeout:
            state = self.get("/api/state")
            job = state.get("job") or {}
            signature = (job.get("status"), job.get("message"), job.get("current"))
            if signature != previous:
                print(json.dumps(dict(job=job, worker_status=state["worker"]["status"])), flush=True)
                previous = signature
            if job.get("command_id") == command_id and job.get("status") in {"complete", "failed", "stopped", "resource_stopped"}:
                if job["status"] != "complete":
                    raise RuntimeError(job.get("message", job["status"]))
                return state
            time.sleep(1)
        raise TimeoutError("Batch wait expired; inspect the still-running job in the browser")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8766")
    parser.add_argument("--recipes", nargs="+", default=["opium", "naive"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--calibration", default="latest")
    parser.add_argument("--config", help="Path to JSON configuration overrides")
    parser.add_argument("--task-count", type=int, default=6)
    parser.add_argument("--action-budget", type=int, default=32)
    parser.add_argument("--token-budget", type=int, default=4096)
    args = parser.parse_args()
    client = Client(args.url)
    state = client.get("/api/state")
    if not state["worker"].get("model"):
        raise SystemExit("Load a model and calibrate it in the browser first.")
    calibration = args.calibration
    if calibration == "latest":
        bundles = state["calibrations"]
        if not bundles:
            raise SystemExit("Create a calibration in the browser first.")
        calibration = sorted(bundles, key=lambda x: x["id"])[-1]["id"]
    config = dict(task_count=args.task_count, action_budget=args.action_budget, token_budget=args.token_budget)
    if args.config:
        with open(args.config, encoding="utf-8") as file:
            config.update(json.load(file))
    accepted = client.command("start_batch", dict(recipe_ids=args.recipes, seeds=args.seeds,
                              calibration_id=calibration, config=config))
    print(json.dumps(accepted), flush=True)
    final = client.wait(accepted["command_id"])
    print(json.dumps(dict(status="complete", run_count=len(final["runs"])), indent=2))


if __name__ == "__main__":
    main()
