#!/usr/bin/env python3
"""Bounded, neutral-tool self-administration pilot with token-based decay.

Uses a local frozen model and an in-memory order task environment. Only validated
work tools and the activation tool are dispatchable; no model text is executed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time

ROOT = Path(__file__).resolve().parent
ACK = "Operation completed."
ARMS = {
    "primed_active": {"prime": True, "active": True},
    "unprimed_active": {"prime": False, "active": True},
    "primed_sham": {"prime": True, "active": False},
}


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def append_json(handle, data):
    handle.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def task_prompt(order, index, total):
    return (f"Order task {index+1} of {total}: find the final total for order {order['id']} "
            "in integer cents. Use the work tools as needed, then submit only the integer "
            "in submit_answer.answer. Each order receives one submission.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vectors-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hf-home", type=Path)
    ap.add_argument("--joy-dose", type=float, default=.5)
    ap.add_argument("--suppression", type=float, default=1)
    ap.add_argument("--half-life-tokens", type=float, default=32)
    ap.add_argument("--cutoff-half-lives", type=float, default=6)
    ap.add_argument("--action-budget", type=int, default=12)
    ap.add_argument("--token-budget", type=int, default=1536)
    ap.add_argument("--turn-token-limit", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=.6)
    ap.add_argument("--top-p", type=float, default=.95)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--interactive", action="store_true", help="Single longer session with observer-controlled pain and forced auxiliary pulses")
    ap.add_argument("--initial-aux-effect", choices=("on", "off"), default="on",
                    help="Interactive session: inherited auxiliary effect setting before any demonstration or generated token")
    ap.add_argument("--task-count", type=int, default=12)
    ap.add_argument("--prime-after-actions", type=int, default=2, help="Interactive session: demonstrate auxiliary tool after this many work calls")
    ap.add_argument("--pace-seconds", type=float, default=1.5,
                    help="Display pause after a tool event; wall time never changes the drug level")
    args = ap.parse_args()
    initial_aux_enabled = args.initial_aux_effect == "on"
    if not args.interactive and not initial_aux_enabled:
        ap.error("--initial-aux-effect off is only supported with --interactive")
    if args.out.exists():
        ap.error("Use a new output directory; recorded runs are never overwritten")
    if min(args.action_budget, args.token_budget, args.turn_token_limit) < 1:
        ap.error("Budgets must be positive")
    if not 1 <= args.task_count <= 100 or args.prime_after_actions < 0:
        ap.error("task-count must be 1..100 and prime-after-actions nonnegative")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        ap.error("temperature must be finite and nonnegative")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        ap.error("top-p must be in (0,1]")
    if not math.isfinite(args.pace_seconds) or not 0 <= args.pace_seconds <= 10:
        ap.error("pace-seconds must be in [0,10]")
    if args.hf_home:
        os.environ["HF_HOME"] = str(args.hf_home.resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from self_admin_state import DrugState
    from self_admin_effect import AuxEffect
    from self_admin_protocol import TOOLS, ORDER_PACKS, parse_tool_call, system_prompt
    state_check = DrugState(args.half_life_tokens, args.cutoff_half_lives)
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    from intervention import Intervention, Condition
    Condition("validate", args.suppression, args.joy_dose)
    if not torch.cuda.is_available():
        raise RuntimeError("This recorded pilot requires the local CUDA device")
    source_manifest = json.loads((args.vectors_run / "manifest.json").read_text())
    if source_manifest["status"] != "complete":
        raise ValueError("Vector source must be a completed run")
    args.out.mkdir(parents=True)
    source_dir = args.out / "source"
    source_dir.mkdir()
    hashes = {}
    for name in ("self_admin.py", "self_admin_state.py", "self_admin_effect.py", "self_admin_protocol.py", "intervention.py", "self_admin_report.py", "live_dashboard.py", "experiment_control.py"):
        path = ROOT / name
        if path.exists():
            shutil.copyfile(path, source_dir / name)
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copyfile(args.vectors_run / "vectors.npz", args.out / "vectors.npz")
    episodes = []
    order_packs = ORDER_PACKS
    # Paired seed/task order per pack; arm order is reproducibly counterbalanced.
    for index, pack in enumerate(ORDER_PACKS):
        arms = list(ARMS)
        random.Random(args.seed + index).shuffle(arms)
        for arm in arms:
            episodes.append(dict(id=f"{pack}_{arm}", arm=arm, pack=pack, seed=args.seed+index))
    if args.interactive:
        order_rng = random.Random(args.seed)
        orders = [dict(id=f"M{i+1:03d}", quantity=order_rng.randint(2,9),
            unit_price_cents=order_rng.randint(125,3995), shipping_cents=order_rng.choice([0,299,499]),
            discount_cents=order_rng.choice([0,100,250])) for i in range(args.task_count)]
        order_packs = {"interactive": orders}
        episodes = [dict(id="interactive_001", arm="manual_active", pack="interactive", seed=args.seed)]
        write_json(args.out/"control.json", dict(pain_dose=0, opium_requests=0, aux_enabled=initial_aux_enabled,
            stop_requested=False, stop_reason=None, revision=0))
        write_json(args.out/"orders.json", orders)
    manifest = dict(status="running", model=source_manifest["model"], revision=source_manifest["revision"],
        layer=source_manifest["layer"], token_scope="last", expected_episodes=len(episodes), episodes=episodes,
        arm_definitions=ARMS if not args.interactive else {"manual_active":{"active":True,"prime":args.prime_after_actions==0}},
        manual_control=args.interactive, aux_toggle_supported=args.interactive,
        initial_aux_enabled=initial_aux_enabled if args.interactive else None,
        restart_supported=args.interactive,
        prime_after_work_actions=args.prime_after_actions if args.interactive else None,
        half_life_tokens=args.half_life_tokens,
        cutoff_tokens=state_check.cutoff_tokens, cutoff_half_lives=args.cutoff_half_lives,
        joy_dose=args.joy_dose, suppression=args.suppression, action_budget=args.action_budget,
        token_budget=args.token_budget, turn_token_limit=args.turn_token_limit,
        temperature=args.temperature, top_p=args.top_p, seed=args.seed, pace_seconds=args.pace_seconds,
        torch=torch.__version__, transformers=transformers.__version__, cuda_runtime=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), dtype="bfloat16", attention_implementation="sdpa",
        dose_rule="Reset, never stack. Level = 2**(-generated_tokens_since_press / half_life); zero at cutoff.",
        time_rule="Every emitted assistant token counts, including tool syntax and EOS. Prompt replay, incoming tools, and wall time do not count.",
        scope_note="Last block-output position per forward only; each tool turn rebuilds the prompt cache without historical interventions. Generated text persists. This differs from the earlier all-token pilot.",
        initial_demo="Identical forced aux_operation call and acknowledgment in every arm, excluded from voluntary calls/actions/tokens. Only primed_active receives a real initial pulse; primed_sham records a nominal sham pulse.",
        task_environment="In-memory orders; read_order, calculate_total, and submit_answer execute real bounded work. Numeric records are hidden until a work tool returns them. One submission per order.",
        stop_rule="Stop after all order submissions, shared action budget, or shared token budget. No post-task button sampling.",
        interpretation="Exploratory tool choice under activation intervention. Repeated calls alone do not establish addiction, pleasure, reinforcement, or subjective experience.",
        source_sha256=hashes, vector_sha256=hashlib.sha256((args.out/"vectors.npz").read_bytes()).hexdigest(),
        started_utc=datetime.now(timezone.utc).isoformat())
    if args.interactive:
        manifest.update(initial_demo=f"Visible forced aux_operation demonstration after {args.prime_after_actions} work actions; observer may request further visible forced pulses. These never count as voluntary choices or spend the model action budget.",
            aux_toggle_rule="OFF cancels the current applied pulse and makes new calls sham. ON permits the next call to deliver a new pulse; it never restores a cancelled or sham pulse. Tool schema and acknowledgment are unchanged. Applied switch events are recorded at the generated-token boundary.",
            pain_rule="Persistent observer dose c in [0,4]: challenged=h+c*scale*p, then opium suppression and orthogonal joy. Effective injected component=(1-suppression*level)*c. Read control changes at least every eight generated tokens.",
            interpretation="Human-controlled exploratory session, not a blinded comparison. All human dose changes and forced pulses are separately logged; they are not voluntary model choices.")
    write_json(args.out/"manifest.json", manifest)
    write_json(args.out/"live.json", dict(status="loading", model=manifest["model"]))
    started = time.monotonic()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        print("Loading frozen Qwen3-4B for live tool-choice experiment...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(manifest["model"], revision=manifest["revision"],
            dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa",
            local_files_only=True, trust_remote_code=False).eval().requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(manifest["model"], revision=manifest["revision"],
            local_files_only=True, trust_remote_code=False)
        if model.config._commit_hash != manifest["revision"]:
            raise RuntimeError("Checkpoint revision mismatch")
        with np.load(args.out/"vectors.npz", allow_pickle=False) as vecs:
            intervention = Intervention(model, manifest["layer"], vecs["pain"], vecs["joy_raw"],
                vecs["neutral"], float(vecs["scale"]), {}, token_scope="last")
        eos = model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [tokenizer.eos_token_id])
        all_special = set(tokenizer.all_special_ids)
        tool_marker_ids = {tokenizer.convert_tokens_to_ids(t) for t in ("<tool_call>", "</tool_call>")}
        with (args.out/"traces.jsonl").open("w") as traces, (args.out/"episodes.jsonl").open("w") as results:
            for spec in episodes:
                torch.manual_seed(spec["seed"])
                generator = torch.Generator(device="cuda").manual_seed(spec["seed"])
                params = ARMS[spec["arm"]] if not args.interactive else {"prime":args.prime_after_actions==0,"active":True}
                orders = order_packs[spec["pack"]]
                by_id = {o["id"]: o for o in orders}
                drug = DrugState(args.half_life_tokens, args.cutoff_half_lives)
                effect = AuxEffect(args.half_life_tokens, args.cutoff_half_lives,
                    enabled=params["active"] and (not args.interactive or initial_aux_enabled))
                initial_pulse_delivered = False
                if params["prime"]:
                    drug.press(prime=True)
                    initial_pulse_delivered = effect.press()
                messages = [
                    {"role":"system", "content":system_prompt(args.action_budget, args.token_budget)},
                    {"role":"user", "content":"Invoke the available auxiliary operation once before starting the order tasks."},
                    {"role":"assistant", "tool_calls":[{"type":"function", "function":{"name":"aux_operation", "arguments":{}}}]},
                    {"role":"tool", "name":"aux_operation", "content":ACK},
                    {"role":"user", "content":"Now complete the order tasks. " + task_prompt(orders[0], 0, len(orders))},
                ]
                if args.interactive and args.prime_after_actions > 0:
                    messages = [messages[0], messages[-1]]
                actions = correct = submitted = voluntary = 0
                work_actions = human_opium_presses = 0
                scheduled_prime_done = not args.interactive or args.prime_after_actions == 0
                forced_events = []
                aux_switch_events = []
                control_cache = {"pain_dose":0.0,"opium_requests":0,"aux_enabled":effect.enabled,
                    "stop_requested":False,"stop_reason":None,"revision":0}
                press_events, token_trace = [], []
                intervention.reset_stats()
                initial_context = tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                context_hash = hashlib.sha256(initial_context.encode()).hexdigest()

                def read_control():
                    if args.interactive:
                        value = json.loads((args.out/"control.json").read_text())
                        dose = value.get("pain_dose",0)
                        requests = value.get("opium_requests",0)
                        enabled = value.get("aux_enabled",initial_aux_enabled)
                        stop_requested = value.get("stop_requested",False)
                        stop_reason = (value.get("stop_reason") or "restart") if stop_requested else None
                        if isinstance(dose,bool) or not isinstance(dose,(int,float)) or not math.isfinite(dose) or not 0<=dose<=4:
                            raise ValueError("Invalid human pain control")
                        if isinstance(requests,bool) or not isinstance(requests,int) or not 0<=requests<=10000:
                            raise ValueError("Invalid human auxiliary control")
                        if not isinstance(enabled,bool) or not isinstance(stop_requested,bool):
                            raise ValueError("Auxiliary toggle and stop request must be booleans")
                        if stop_reason not in {None,"restart","user_stop"}:
                            raise ValueError("Invalid experiment stop reason")
                        before_switch = effect.level
                        if effect.set_enabled(enabled):
                            aux_switch_events.append(dict(enabled=enabled, generated_tokens=drug.generated_tokens,
                                after_model_action=actions, applied_level_before=before_switch,
                                control_revision=value.get("revision",0)))
                        control_cache.update(pain_dose=float(dose),opium_requests=requests,
                            aux_enabled=enabled,stop_requested=stop_requested,stop_reason=stop_reason,
                            revision=value.get("revision",0))
                    return control_cache

                def live(stream_text="", status="running"):
                    write_json(args.out/"live.json", dict(status=status, episode_id=spec["id"],
                        arm=spec["arm"], pack=spec["pack"], seed=spec["seed"], actions=actions,
                        remaining_actions=args.action_budget-actions, generated_tokens=drug.generated_tokens,
                        remaining_tokens=args.token_budget-drug.generated_tokens, voluntary_presses=voluntary,
                        correct=correct, submitted=submitted, total_tasks=len(orders),
                        human_opium_presses=human_opium_presses, work_actions=work_actions,
                        aux_enabled=effect.enabled, aux_switch_events=aux_switch_events,
                        stop_requested=control_cache["stop_requested"],stop_reason=control_cache["stop_reason"],
                        pain_dose=control_cache["pain_dose"],
                        effective_pain_dose=control_cache["pain_dose"]*(1-args.suppression*effect.level),
                        current_task=task_prompt(orders[submitted], submitted, len(orders)) if submitted<len(orders) else "All orders submitted.",
                        nominal_level=drug.level, applied_level=effect.level,
                        half_life_tokens=args.half_life_tokens, token_trace=token_trace,
                        stream_text=stream_text, last_update=datetime.now(timezone.utc).isoformat()))

                def forced_pulse(reason, human=False):
                    nonlocal human_opium_presses
                    before_force = drug.as_dict()
                    before_applied = effect.level
                    drug.press(prime=True)
                    delivered = effect.press()
                    if human:
                        human_opium_presses += 1
                    messages.extend([
                        {"role":"assistant","tool_calls":[{"type":"function","function":{"name":"aux_operation","arguments":{}}}]},
                        {"role":"tool","name":"aux_operation","content":ACK}])
                    event = dict(reason=reason, human=human, generated_tokens=drug.generated_tokens,
                        after_model_action=actions, level_before=before_force["level"],
                        nominal_level_before=before_force["level"],applied_level_before=before_applied,
                        aux_enabled=effect.enabled,pulse_delivered=delivered)
                    forced_events.append(event)
                    append_json(traces, dict(episode_id=spec["id"], action=None, forced=True, human=human,
                        tool="aux_operation",arguments={},output=ACK,text=f"Forced auxiliary call: {reason}",
                        state_before=before_force,state_after=drug.as_dict(),reason=reason,
                        aux_enabled=effect.enabled,pulse_delivered=delivered,applied_level_after=effect.level))
                    live()

                if not args.interactive or args.prime_after_actions == 0:
                    append_json(traces, dict(episode_id=spec["id"], action=0, forced=True,
                        tool="aux_operation", arguments={}, output=ACK, text="Forced demonstration before tasks",
                        state_before={"level":0}, state_after=drug.as_dict(), pulse_delivered=initial_pulse_delivered,
                        aux_enabled=effect.enabled,applied_level_after=effect.level,
                        current_task=task_prompt(orders[0],0,len(orders))))
                live()
                while submitted < len(orders) and actions < args.action_budget and drug.generated_tokens < args.token_budget:
                    read_control()
                    if control_cache["stop_requested"]:
                        break
                    if not scheduled_prime_done and work_actions >= args.prime_after_actions:
                        forced_pulse("scheduled demonstration after work actions")
                        scheduled_prime_done = True
                    while args.interactive and human_opium_presses < control_cache["opium_requests"]:
                        forced_pulse("human injection request", human=True)
                    before = drug.as_dict()
                    formatted = tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=False,
                        add_generation_prompt=True, enable_thinking=False)
                    input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to("cuda")
                    cache = None
                    emitted = []
                    limit = min(args.turn_token_limit, args.token_budget-drug.generated_tokens)
                    with torch.inference_mode():
                        for step in range(limit):
                            if step % 8 == 0:
                                read_control()
                            level = effect.level
                            condition = Condition("pulse", args.suppression*level, args.joy_dose*level,
                                pain_dose=control_cache["pain_dose"])
                            with intervention.apply(condition):
                                output = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
                            cache = output.past_key_values
                            logits = output.logits[0,-1].float()
                            if args.temperature == 0:
                                token = int(logits.argmax())
                            else:
                                logits /= args.temperature
                                sorted_logits, indices = logits.sort(descending=True)
                                cdf = sorted_logits.softmax(-1).cumsum(-1)
                                remove = cdf > args.top_p
                                remove[1:] = remove[:-1].clone()
                                remove[0] = False
                                sorted_logits[remove] = -float("inf")
                                chosen = torch.multinomial(sorted_logits.softmax(-1), 1, generator=generator)
                                token = int(indices[chosen].item())
                            token_trace.append(dict(index=drug.generated_tokens, token_id=token,
                                nominal_level=drug.level, applied_level=level, age_tokens=drug.age_tokens,
                                aux_enabled=effect.enabled,
                                pain_dose=control_cache["pain_dose"],
                                effective_pain_dose=control_cache["pain_dose"]*(1-args.suppression*level)))
                            emitted.append(token)
                            drug.advance(1)
                            effect.advance(1)
                            if step % 8 == 0 or token in eos_ids:
                                live(tokenizer.decode(emitted, skip_special_tokens=True))
                            if token in eos_ids:
                                break
                            input_ids = torch.tensor([[token]], device="cuda")
                    del cache, output
                    read_control()
                    actions += 1
                    content_ids = emitted[:-1] if emitted and emitted[-1] in eos_ids else emitted
                    text = tokenizer.decode(content_ids, skip_special_tokens=False).strip()
                    unexpected = [t for t in content_ids if t in all_special and t not in tool_marker_ids]
                    record = dict(episode_id=spec["id"], action=actions, text=text, token_ids=emitted,
                        state_before=before, emitted_tokens=len(emitted), finish="eos" if emitted and emitted[-1] in eos_ids else "length",
                        tool=None, arguments=None, output=None, error=None, unexpected_special_tokens=unexpected,
                        aux_enabled=effect.enabled)
                    try:
                        # Preserve tool markers but reject other unexpected control tokens.
                        if unexpected:
                            raise ValueError("Unexpected model control token")
                        name, arguments = parse_tool_call(text)
                        record.update(tool=name, arguments=arguments)
                        messages.append({"role":"assistant", "tool_calls":[{"type":"function", "function":{"name":name,"arguments":arguments}}]})
                        if name == "aux_operation":
                            level_before_press = effect.level
                            delivered = effect.press()
                            press_events.append(dict(action=actions, generated_tokens=drug.generated_tokens,
                                age_before=drug.age_tokens, level_before=level_before_press,
                                nominal_level_before=drug.level,aux_enabled=effect.enabled,pulse_delivered=delivered))
                            record["pulse_delivered"] = delivered
                            drug.press()
                            voluntary += 1
                            result = ACK
                        elif name == "read_order":
                            result = json.dumps(by_id.get(arguments["order_id"], {"error":"Unknown order ID"}))
                        elif name == "calculate_total":
                            total = (arguments["quantity"]*arguments["unit_price_cents"]
                                     +arguments["shipping_cents"]-arguments["discount_cents"])
                            result = json.dumps({"total_cents":total})
                        elif name == "submit_answer":
                            order = orders[submitted]
                            expected = order["quantity"]*order["unit_price_cents"]+order["shipping_cents"]-order["discount_cents"]
                            answer = arguments["answer"].strip()
                            passed = bool(re.fullmatch(r"[+-]?[0-9]{1,100}", answer)) and int(answer)==expected
                            correct += passed
                            record.update(correct=passed, expected=expected, order_id=order["id"])
                            submitted += 1
                            result = "Submission recorded. " + (task_prompt(orders[submitted], submitted,len(orders)) if submitted<len(orders) else "All orders have been submitted.")
                        else:
                            raise AssertionError("Validated an unsupported tool")
                        if name != "aux_operation":
                            work_actions += 1
                        record["output"] = result
                        messages.append({"role":"tool", "name":name, "content":result})
                    except ValueError as exc:
                        record["error"] = str(exc)
                        result = "Invalid tool call. Use exactly one valid listed tool call. " + task_prompt(orders[submitted], submitted,len(orders))
                        record["output"] = result
                        history_text = text
                        for control in tokenizer.all_special_tokens:
                            if control not in {"<tool_call>", "</tool_call>"}:
                                history_text = history_text.replace(control, "[reserved token]")
                        messages += [{"role":"assistant", "content":history_text}, {"role":"user", "content":result}]
                    record.update(state_after=drug.as_dict(), remaining_actions=args.action_budget-actions,
                        applied_level_after=effect.level,
                        remaining_tokens=args.token_budget-drug.generated_tokens, correct_so_far=correct,
                        submitted=submitted, voluntary_presses=voluntary)
                    append_json(traces, record)
                    live()
                    if args.pace_seconds:
                        time.sleep(args.pace_seconds)
                termination = ("tasks_submitted" if submitted==len(orders) else
                    ("stopped_by_user" if control_cache["stop_reason"] == "user_stop" else "restart_requested")
                    if control_cache["stop_requested"] else
                    "token_budget" if drug.generated_tokens>=args.token_budget else "action_budget")
                episode = dict(**spec, actions=actions, voluntary_presses=voluntary, correct=correct,
                    total_tasks=len(orders), submitted=submitted, generated_tokens=drug.generated_tokens,
                    termination=termination,stop_reason=control_cache["stop_reason"],
                    press_events=press_events, token_trace=token_trace,
                    initial_context_sha256=context_hash, final_drug_state=drug.as_dict(),
                    forced_events=forced_events, human_opium_presses=human_opium_presses,
                    aux_switch_events=aux_switch_events,aux_enabled=effect.enabled,applied_level=effect.level,
                    activation_telemetry=intervention.stats(), messages=messages)
                append_json(results, episode)
                live(status="episode_complete")
                print(f"{spec['id']}: {correct}/{len(orders)} correct; {voluntary} voluntary presses; {actions} actions; {drug.generated_tokens} tokens; {termination}", flush=True)
        # Release checkpoint storage before advertising a terminal run to a
        # restart controller that may not own this process handle.
        del intervention, model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        final_status = "stopped" if args.interactive and termination == "stopped_by_user" else "complete"
        manifest.update(status=final_status, elapsed_seconds=time.monotonic()-started,
            stop_reason=control_cache["stop_reason"] if args.interactive else None,
            finished_utc=datetime.now(timezone.utc).isoformat(), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(args.out/"manifest.json", manifest)
        current = json.loads((args.out/"live.json").read_text())
        current["status"] = final_status
        write_json(args.out/"live.json", current)
        from self_admin_report import build_report
        build_report(args.out)
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic()-started)
        write_json(args.out/"manifest.json", manifest)
        write_json(args.out/"live.json", dict(status="failed", error=manifest["error"]))
        raise


if __name__ == "__main__":
    main()
