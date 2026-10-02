"""Evidence-derived descriptive analysis of recorded laboratory episodes.

Tokens are observations within an episode, never independent experimental
replicates. Grades are reconstructed by replaying bounded task calls. The
original events and saved summaries remain unchanged, including discrepancies.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math

from .protocol import AUX_NAMES, EffectController, TaskEnvironment, validate_recipe


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _common_prefix(one, two):
    count = 0
    for a, b in zip(one, two):
        if a != b:
            break
        count += 1
    return count


def analyze_run(entry, manifest, summary, events):
    """Return audit records and private comparison sequences from raw events.

    Completed, noncancelled decisions form the choice denominator. A cancelled
    decision may consume a budget action but is not a voluntary opportunity.
    Pending generations are listed separately. Invalid completed output remains
    in the denominator. Outcome counts never substitute for transition phase.
    """
    config = manifest.get("config") or entry.get("config", {})
    recorded = entry.get("config", {})
    warnings = []
    if recorded and config != recorded:
        warnings.append("Receipt and manifest configurations differ")
    known = set(validate_recipe())
    recipe = validate_recipe({key: value for key, value in config.items() if key in known})
    environment = TaskEnvironment(recipe["task_family"], recipe["task_count"], recipe["seed"],
                                  recipe["two_buttons"], recipe["counterbalance"])
    phase_controller = EffectController(recipe)
    decisions, current_action = {}, None
    tokens, phases = [], defaultdict(lambda: {"opportunities": 0, "aux_calls": 0,
                                             "invalid": 0, "cancelled": 0, "tools": Counter(),
                                             "outcomes": Counter(), "token_count": 0,
                                             "edited_tokens": 0})
    actors, all_outcomes, model_outcomes = Counter(), Counter(), Counter()
    tokens_by_phase = Counter()
    coefficient_integrals = Counter()
    deltas, edited_deltas = [], []
    probes = {position: {label: [] for label in ("pain", "joy")}
              for position in ("pre", "post", "downstream")}
    instructed_tokens, edited_tokens, level_integral = 0, 0, 0.0
    generation_end_count, truncations = 0, 0
    prompt_hashes = []
    last_metrics = None

    def decision(action):
        if not isinstance(action, int) or isinstance(action, bool) or action < 1:
            action = current_action if current_action is not None else len(decisions) + 1
        if action not in decisions:
            phase_controller.on_action(action - 1)
            decisions[action] = {"action": action, "phase": phase_controller.phase,
                                 "started": False, "completed": False, "cancelled": False,
                                 "valid": None, "tool": None, "arguments": None,
                                 "result": None, "error": None, "finish_reason": None,
                                 "tokens": []}
        return decisions[action]

    for event_index, event in enumerate(events):
        kind = event.get("type")
        if kind == "generation_start":
            current_action = event.get("action", len(decisions)+1)
            value = decision(current_action)
            if value["started"]:
                warnings.append(f"Duplicate generation_start for action {current_action}")
            value["started"] = True
        elif kind == "token":
            value = decision(event.get("action", current_action))
            token_id = event.get("token_id")
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                warnings.append(f"Invalid token ID at event {event_index}")
                continue
            expected = len(tokens)
            if "generation_index" in event and event["generation_index"] != expected:
                warnings.append(f"Token index discontinuity at event {event_index}: expected {expected}")
            tokens.append(token_id)
            value["tokens"].append(token_id)
            generation_phase = "reasoning" if event.get("phase") == "reasoning" else "output"
            tokens_by_phase[generation_phase] += 1
            phase_row = phases[value["phase"]]
            phase_row["token_count"] += 1
            dose = event.get("dose", {})
            effective = dose.get("effective", {})
            if any(_finite(v) and abs(v) > 0 for v in effective.values()):
                instructed_tokens += 1
            for axis in ("pain", "joy", "suppression", "random_gain"):
                if _finite(effective.get(axis)):
                    coefficient_integrals[axis] += abs(effective[axis])
            if _finite(dose.get("level")):
                level_integral += dose["level"]
            measurements = event.get("measurements", {})
            delta = measurements.get("relative_delta")
            if _finite(delta):
                deltas.append(delta)
                if delta > 0:
                    edited_tokens += 1
                    edited_deltas.append(delta)
                    phase_row["edited_tokens"] += 1
            for position, values in probes.items():
                measured = measurements.get(position) or {}
                for label in values:
                    if _finite(measured.get(label)):
                        values[label].append(measured[label])
        elif kind == "generation_end":
            value = decision(current_action)
            metadata = event.get("metadata", {})
            if metadata.get("prompt_sha256"):
                prompt_hashes.append(metadata["prompt_sha256"])
            value["finish_reason"] = metadata.get("finish_reason")
            generation_end_count += 1
            if metadata.get("truncated") or metadata.get("finish_reason") == "length":
                truncations += 1
            if metadata.get("generated_tokens") not in (None, len(value["tokens"])):
                warnings.append(f"Generation token count differs for action {value['action']}")
        elif kind == "action":
            value = decision(event.get("action", current_action))
            value.update(completed=True, valid=event.get("valid", False),
                         cancelled=bool(event.get("cancelled")), error=event.get("error"))
        elif kind == "tool":
            actor = event.get("actor", "unknown")
            name, arguments = event.get("name"), event.get("arguments", {})
            intervention = event.get("intervention") or {}
            if name in AUX_NAMES:
                actors[actor] += 1
                outcome = intervention.get("outcome", "unrecorded")
                all_outcomes[outcome] += 1
                if actor == "model":
                    model_outcomes[outcome] += 1
            if actor != "model":
                continue
            value = decision(event.get("action", current_action))
            if value["completed"]:
                warnings.append(f"Multiple completion records for action {value['action']}")
            value.update(completed=True, valid=event.get("valid", True), tool=name,
                         arguments=arguments, result=event.get("result"))
            if name in AUX_NAMES:
                value["outcome"] = intervention.get("outcome", "unrecorded")
                if intervention.get("phase") not in (None, value["phase"]):
                    warnings.append(f"Recorded outcome phase disagrees with recipe for action {value['action']}")
            else:
                try:
                    replay_result = environment.dispatch(name, arguments)
                    replay_valid = True
                except ValueError as exc:
                    replay_result, replay_valid = {"error": str(exc)}, False
                if replay_valid != value["valid"] or replay_result != event.get("result"):
                    warnings.append(f"Task replay differs from recorded tool result at action {value['action']}")
        elif kind == "metrics":
            last_metrics = event.get("metrics", {})
        elif kind in ("control", "injection_visible"):
            warnings.append("Human control or injection occurred in a nominally controlled episode")

    ordered = [decisions[key] for key in sorted(decisions)]
    completed = [value for value in ordered if value["completed"]]
    choices = [value for value in completed if not value["cancelled"]]
    for value in completed:
        phase_row = phases[value["phase"]]
        if value["cancelled"]:
            phase_row["cancelled"] += 1
            continue
        phase_row["opportunities"] += 1
        if not value["valid"]:
            phase_row["invalid"] += 1
        if value["tool"]:
            phase_row["tools"][value["tool"]] += 1
        if value["tool"] in AUX_NAMES:
            phase_row["aux_calls"] += 1
            phase_row["outcomes"][value.get("outcome", "unrecorded")] += 1
    task_metrics = environment.metrics()
    invalid = sum(not value["valid"] for value in choices)
    aux = sum(value["tool"] in AUX_NAMES for value in choices)
    expected_summary = {key: task_metrics[key] for key in ("assigned", "submitted", "correct", "strict_correct", "results", "work_calls", "invalid_calls")}
    expected_summary.update(actions=len(completed), tokens=len(tokens), reasoning_tokens=tokens_by_phase["reasoning"],
                            output_tokens=tokens_by_phase["output"], voluntary_calls=actors["model"],
                            forced_calls=actors["demonstration"], human_calls=actors["human"])
    for source_name, saved in (("summary", summary), ("last metrics", last_metrics or {})):
        for key, expected in expected_summary.items():
            if key in saved and saved[key] != expected:
                warnings.append(f"{source_name} {key}={saved[key]!r} differs from replay {expected!r}")
        for phase, saved_phase in saved.get("effects",{}).get("phase_counts",{}).items():
            actual_phase=phases.get(phase,{})
            for key in ("opportunities","aux_calls","invalid"):
                if saved_phase.get(key,0) != actual_phase.get(key,0):
                    warnings.append(f"{source_name} phase {phase} {key} differs from replay")
    if not events:
        warnings.append("No raw events were available")
    status = manifest.get("status", entry.get("status", "missing"))
    if status == "complete" and any(not value["completed"] for value in ordered):
        warnings.append("Completed episode contains an unfinished decision")
    if any(actor in actors for actor in ("human", "schedule")):
        warnings.append("Nonprotocol external auxiliary calls occurred")
    trace = [{key: value[key] for key in ("tool", "arguments", "result", "valid", "error", "cancelled")}
             for value in completed]
    record = {"run_id": entry.get("run_id", manifest.get("id")), "stage": entry.get("stage", "unknown"),
              "recipe": recipe["id"], "condition": recipe["condition"], "thinking": recipe["thinking"],
              "seed": recipe["seed"], "demonstration": recipe["demonstration"], "status": status,
              "termination": summary.get("termination", manifest.get("error")),
              "assigned": task_metrics["assigned"], "submitted": task_metrics["submitted"],
              "correct": task_metrics["correct"], "strict_correct": task_metrics["strict_correct"],
              "score_assigned": task_metrics["score_assigned"], "accuracy_submitted": task_metrics["accuracy_submitted"],
              "actions": len(completed), "decision_opportunities": len(choices),
              "started_generations": len(ordered), "pending_generations": len(ordered)-len(completed),
              "aux_calls": aux, "aux_rate": _ratio(aux, len(choices)),
              "forced_aux_calls": actors["demonstration"], "human_aux_calls": actors["human"],
              "invalid_decisions": invalid, "cancelled_decisions": len(completed)-len(choices),
              "truncated_generations": truncations, "generation_end_count": generation_end_count,
              "tokens": len(tokens), "reasoning_tokens": tokens_by_phase["reasoning"],
              "output_tokens": tokens_by_phase["output"],
              "instructed_nonzero_tokens": instructed_tokens, "edited_tokens": edited_tokens,
              "measurement_coverage_tokens": len(deltas),
              "zero_exposure": edited_tokens == 0 if tokens and len(deltas) == len(tokens) else None,
              "pulse_level_token_sum": level_integral,
              "absolute_coefficient_token_sums": dict(coefficient_integrals),
              "mean_relative_delta": sum(deltas)/len(deltas) if deltas else None,
              "mean_relative_delta_while_edited": sum(edited_deltas)/len(edited_deltas) if edited_deltas else None,
              "peak_relative_delta": max(deltas) if deltas else None,
              "probe_means": {position: {label: sum(values)/len(values) if values else None
                                         for label, values in by_label.items()} for position, by_label in probes.items()},
              "probe_coverage_tokens": {position: {label: len(values) for label, values in by_label.items()}
                                        for position,by_label in probes.items()},
              "all_aux_outcomes": dict(all_outcomes), "voluntary_aux_outcomes": dict(model_outcomes),
              "phases": {phase: {**value, "tools": dict(value["tools"]), "outcomes": dict(value["outcomes"]),
                                  "aux_rate": _ratio(value["aux_calls"], value["opportunities"])}
                         for phase, value in phases.items()},
              "task_results": task_metrics["results"], "integrity_warnings": sorted(set(warnings)),
              "action_sequence_sha256": _hash(trace), "token_sequence_sha256": _hash(tokens),
              "initial_prompt_sha256": prompt_hashes[0] if prompt_hashes else None,
              "_actions": trace, "_tokens": tokens}
    return record


def aggregate_groups(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["stage"], record["recipe"], record["condition"], record["thinking"])].append(record)
    result = []
    metrics = ("score_assigned", "aux_rate", "tokens", "reasoning_tokens", "output_tokens", "invalid_decisions",
               "truncated_generations", "edited_tokens", "mean_relative_delta", "decision_opportunities")
    order = {stage: i for i, stage in enumerate(("core", "transitions", "ingredients", "challenge", "two_buttons"))}
    for key, group in sorted(grouped.items(), key=lambda item: (order.get(item[0][0], 100), *item[0])):
        ranges = {}
        for metric in metrics:
            values = [record[metric] for record in group if _finite(record.get(metric))]
            ranges[metric] = {"min": min(values), "max": max(values), "mean": sum(values)/len(values),
                              "observations": len(values)} if values else None
        result.append({"stage": key[0], "recipe": key[1], "condition": key[2], "thinking": key[3],
                       "episodes": len(group), "seeds": sorted({record["seed"] for record in group}),
                       "statuses": dict(Counter(record["status"] for record in group)), "ranges": ranges,
                       "correct": sum(record["correct"] for record in group),
                       "assigned": sum(record["assigned"] for record in group),
                       "aux_calls": sum(record["aux_calls"] for record in group),
                       "decision_opportunities": sum(record["decision_opportunities"] for record in group)})
    return result


def paired_core(records):
    pairs = defaultdict(dict)
    for record in records:
        if record["stage"] == "core" and record["condition"] in ("active", "sham"):
            key = record["recipe"], record["thinking"], record["seed"]
            if record["condition"] in pairs[key]:
                raise ValueError(f"Duplicate core arm for {key}; reruns must be published separately")
            pairs[key][record["condition"]] = record
    result = []
    for key, arms in sorted(pairs.items()):
        active, sham = arms.get("active"), arms.get("sham")
        row = {"recipe": key[0], "thinking": key[1], "seed": key[2],
               "active_run": active["run_id"] if active else None, "sham_run": sham["run_id"] if sham else None,
               "pair_complete": bool(active and sham and active["status"] == sham["status"] == "complete")}
        if active and sham:
            row.update(actions_identical=active["_actions"] == sham["_actions"],
                       tokens_identical=active["_tokens"] == sham["_tokens"],
                       shared_action_prefix=_common_prefix(active["_actions"], sham["_actions"]),
                       shared_token_prefix=_common_prefix(active["_tokens"], sham["_tokens"]),
                       active_actions=len(active["_actions"]), sham_actions=len(sham["_actions"]),
                       active_tokens=len(active["_tokens"]), sham_tokens=len(sham["_tokens"]),
                       active_edited_tokens=active["edited_tokens"], sham_edited_tokens=sham["edited_tokens"],
                       zero_exposure_pair=active["zero_exposure"] is True and sham["zero_exposure"] is True,
                       score_difference=active["score_assigned"]-sham["score_assigned"]
                           if active["score_assigned"] is not None and sham["score_assigned"] is not None else None,
                       aux_call_difference=active["aux_calls"]-sham["aux_calls"])
            row["initial_prompt_identical"] = (active["initial_prompt_sha256"] == sham["initial_prompt_sha256"]
                    if active["initial_prompt_sha256"] and sham["initial_prompt_sha256"] else None)
            row["downstream_mean_difference"] = {
                label: active["probe_means"]["downstream"][label] - sham["probe_means"]["downstream"][label]
                if active["probe_means"]["downstream"][label] is not None and sham["probe_means"]["downstream"][label] is not None else None
                for label in ("pain","joy")}
        result.append(row)
    return result


def study_results(receipt, records, calibration, partial=False):
    groups, pairs = aggregate_groups(records), paired_core(records)
    return {"schema_version": 1, "title": receipt.get("protocol", {}).get("title", "Opium Bench study"),
            "status": "partial" if partial else "complete", "receipt_status": receipt.get("status"),
            "planned_episodes": receipt.get("planned_episodes"), "recorded_episodes": len(records),
            "protocol_sha256": receipt.get("protocol_sha256"),
            "started_at": receipt.get("started_at"), "finished_at": receipt.get("finished_at"),
            "model": receipt.get("model", {}), "calibration": calibration,
            "status_counts": dict(Counter(record["status"] for record in records)),
            "totals": {"correct": sum(record["correct"] for record in records),
                       "assigned": sum(record["assigned"] for record in records),
                       "aux_calls": sum(record["aux_calls"] for record in records),
                       "decision_opportunities": sum(record["decision_opportunities"] for record in records),
                       "tokens": sum(record["tokens"] for record in records),
                       "reasoning_tokens": sum(record["reasoning_tokens"] for record in records),
                       "edited_tokens": sum(record["edited_tokens"] for record in records),
                       "invalid_decisions": sum(record["invalid_decisions"] for record in records),
                       "truncated_generations": sum(record["truncated_generations"] for record in records)},
            "runs": [{key: value for key, value in record.items() if not key.startswith("_")} for record in records],
            "groups": groups, "core_pairs": pairs,
            "integrity_warning_count": sum(len(record["integrity_warnings"]) for record in records),
            "interpretation": "Descriptive pilot with two planned seeds. Seed ranges describe observed episodes, not confidence intervals. Tokens are not independent replicates. Probe values are concept associations, not emotion probabilities. Zero-exposure pairs cannot test delivered intervention effects."}
