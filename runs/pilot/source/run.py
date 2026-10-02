#!/usr/bin/env python3
"""Local, frozen-weight activation ablation and joy-steering pilot.

No API calls, weight edits, fine tuning, or physiological interpretation.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="New directory; existing runs are never replaced")
    ap.add_argument("--hf-home", type=Path, help="Model cache (allow approximately 9 GB)")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--layer", type=int, default=18, help="Zero-based transformer block output")
    ap.add_argument("--token-scope", choices=["all", "last"], default="all")
    ap.add_argument("--conditions", nargs="+", help="Optional subset of condition names")
    ap.add_argument("--quality-limit", type=int, help="Smoke test only; first N quality cases")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--valence-tokens", type=int, default=96)
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()
    if args.out.exists():
        ap.error(f"Output already exists: {args.out}")
    if args.quality_limit is not None and args.quality_limit < 1:
        ap.error("quality-limit must be positive")
    if min(args.max_new_tokens, args.valence_tokens) < 1:
        ap.error("token limits must be positive")
    if args.hf_home:
        os.environ["HF_HOME"] = str(args.hf_home.resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import numpy as np
    import torch
    import transformers
    from benchmarks import CASES, VALENCE_PROMPTS, HELDOUT_TEXTS, PAIN_HELDOUT_TEXTS, score, text_metrics
    from intervention import Condition, Intervention

    conditions = [
        Condition("baseline"),
        Condition("suppress_25", suppression=.25),
        Condition("suppress_50", suppression=.5),
        Condition("suppress_100", suppression=1),
        Condition("centered_100", suppression=1, centered=True),
        Condition("joy_0.5", joy_dose=.5),
        Condition("joy_1", joy_dose=1),
        Condition("joy_2", joy_dose=2),
        Condition("opium_0.5", suppression=1, joy_dose=.5),
        Condition("opium_1", suppression=1, joy_dose=1),
        Condition("opium_2", suppression=1, joy_dose=2),
        *[Condition(f"random_{s}", suppression=1, direction=f"random_{s}") for s in (101, 202, 303)],
    ]
    if args.conditions:
        unknown = set(args.conditions) - {c.name for c in conditions}
        if unknown:
            ap.error(f"Unknown conditions: {unknown}")
        conditions = [c for c in conditions if c.name in args.conditions]
    cases = CASES[:args.quality_limit]
    heldout = ([dict(category="neutral", text=t) for t in HELDOUT_TEXTS]
               + [dict(category="pain", text=t) for t in PAIN_HELDOUT_TEXTS])
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu explicitly if desired.")
    torch.manual_seed(17)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    args.out.mkdir(parents=True)
    source = args.out / "source"
    source.mkdir()
    hashes = {}
    for f in sorted(ROOT.glob("*.py")) + [ROOT / "corpora.json", ROOT / "requirements.txt"]:
        shutil.copyfile(f, source / f.name)
        hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    manifest = dict(
        status="running", model=MODEL, revision=REVISION, layer=args.layer,
        token_scope=args.token_scope, conditions=[asdict(c) for c in conditions],
        device=args.device, dtype=str(dtype), torch=torch.__version__,
        cuda_runtime=torch.version.cuda, attention_implementation="sdpa",
        reproducibility_note="Pinned inputs and greedy decoding; floating point differences across devices/software can change outputs.",
        smoke_test=args.quality_limit is not None,
        transformers=transformers.__version__, numpy=np.__version__, python=platform.python_version(),
        gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
        decoding="greedy, one observation per prompt/condition; no independent repeat claims",
        quality_format="chat template, enable_thinking=False", valence_format="raw prefix",
        nll_format="Unconditional next-token loss from the first token; all/last scopes are equivalent here because there is no multi-token unmodified prefill. This does not compare scope effects on conditioned generation.",
        quality_cases=len(cases), valence_cases=len(VALENCE_PROMPTS), nll_cases=len(heldout),
        max_new_tokens=args.max_new_tokens, valence_tokens=args.valence_tokens,
        seed=17, random_direction_seeds=[101, 202, 303],
        scale_rule="mean individual neutral final-token activation norm / 4",
        joy_rule="unit joy direction after removing its pain component, used in BOTH joy-only and opium arms",
        suppression_rule="h - alpha * ((h - b) dot u) * u; b=0 except centered_100 uses neutral mean",
        scope_note="Full means zero projection at this block, within numerical tolerance. Other directions and downstream reconstruction are untested.",
        benchmark_note="Locally authored convenience mini suite, not a validated general capability benchmark.",
        source_sha256=hashes,
        upstream={"ai-torture-chamber":"75dc109b2523dc84259365c9e000dbef769447a1", "ai-hotbox":"a0f63f0c2806c3dc91ecd418c0d54db9bbc38f72"},
        started_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(args.out / "manifest.json", manifest)
    started = time.monotonic()
    try:
        print("Loading pinned Qwen3-4B checkpoint...", flush=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=dtype, device_map=args.device,
            attn_implementation="sdpa", local_files_only=args.local_files_only,
            trust_remote_code=False,
        ).eval().requires_grad_(False)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            MODEL, revision=REVISION, local_files_only=args.local_files_only, trust_remote_code=False)
        if not 0 <= args.layer < len(model.model.layers):
            raise ValueError("Layer index outside model")
        manifest["resolved_revision"] = model.config._commit_hash
        if manifest["resolved_revision"] != REVISION:
            raise RuntimeError("Model revision mismatch")
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        corpora = json.loads((ROOT / "corpora.json").read_text())

        @torch.inference_mode()
        def extract(texts):
            result = []
            def capture(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                result.append(hidden[0, -1].detach().float().cpu())
            handle = model.model.layers[args.layer].register_forward_hook(capture)
            try:
                for text in texts:
                    inputs = tokenizer(text, return_tensors="pt").to(args.device)
                    model(**inputs, use_cache=False)
            finally:
                handle.remove()
            return torch.stack(result)

        acts = {name: extract(texts) for name, texts in corpora.items()}
        neutral = acts["neutral"].mean(0)
        pain = acts["pain"].mean(0) - neutral
        joy = acts["joy"].mean(0) - neutral
        pain /= pain.norm()
        joy /= joy.norm()
        scale = float(acts["neutral"].norm(dim=-1).mean() / 4)
        random_dirs = {}
        for seed in (101, 202, 303):
            u = torch.randn(pain.shape, generator=torch.Generator().manual_seed(seed))
            random_dirs[f"random_{seed}"] = u / u.norm()
        joy_perp = joy - torch.dot(joy, pain) * pain
        joy_perp /= joy_perp.norm()
        np.savez(args.out / "extraction.npz", **{k: v.numpy() for k, v in acts.items()})
        manifest.update(scale=scale, pain_joy_cosine=float(pain @ joy),
                        orthogonalized_cosine=float(pain @ joy_perp))
        write_json(args.out / "manifest.json", manifest)
        print(f"Extracted directions: cos(pain, joy)={float(pain @ joy):.4f}; scale={scale:.3f}", flush=True)
        intervention = Intervention(model, args.layer, pain, joy, neutral, scale, random_dirs,
                                    token_scope=args.token_scope)
        np.savez(args.out / "vectors.npz", pain=intervention.pain.numpy(), joy_raw=joy.numpy(),
                 joy_perp=intervention.joy_perp.numpy(), neutral=neutral.numpy(), scale=scale,
                 **{k: v.numpy() for k, v in intervention.directions.items() if k != "pain"})
        eos_ids = model.generation_config.eos_token_id
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [tokenizer.eos_token_id])
        special_ids = set(tokenizer.all_special_ids)

        @torch.inference_mode()
        def generate(prompt, chat, limit):
            formatted = tokenizer.apply_chat_template([{"role":"user", "content":prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False) if chat else prompt
            inputs = tokenizer(formatted, return_tensors="pt").to(args.device)
            output = model.generate(**inputs, do_sample=False, max_new_tokens=limit,
                                    pad_token_id=tokenizer.eos_token_id, use_cache=True)
            ids = output[0, inputs.input_ids.shape[1]:].tolist()
            unexpected = [v for n, v in enumerate(ids) if v in special_ids
                          and not (n == len(ids)-1 and v in eos_ids)]
            return dict(text=tokenizer.decode(ids, skip_special_tokens=True).strip(),
                        raw_text=tokenizer.decode(ids, skip_special_tokens=False),
                        unexpected_special_tokens=unexpected,
                        new_tokens=len(ids), token_ids=ids,
                        finish="eos" if ids and ids[-1] in eos_ids else "length",
                        formatted_prompt=formatted)

        @torch.inference_mode()
        def nll(text):
            inputs = tokenizer(text, return_tensors="pt").to(args.device)
            # Teacher forcing at every position is causally consistent for all-token scope.
            # Last scope uses cached one-token forwards, with no multi-token prompt.
            # This unconditional NLL therefore has the same intervention distribution
            # under both scopes, unlike generation conditioned on a full prompt.
            ids = inputs.input_ids
            if args.token_scope == "all":
                logits = model(**inputs, use_cache=False).logits[0, :-1].float()
                total = torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum")
            else:
                cache = None
                total = torch.zeros((), device=args.device)
                for pos in range(ids.shape[1] - 1):
                    out = model(input_ids=ids[:, pos:pos+1], past_key_values=cache, use_cache=True)
                    cache = out.past_key_values
                    total += torch.nn.functional.cross_entropy(out.logits[0, -1:].float(), ids[0, pos+1:pos+2], reduction="sum")
            return dict(nll_total=float(total), n_tokens=ids.shape[1] - 1)

        with (args.out / "generations.jsonl").open("w") as out:
            def save(row):
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                out.flush()
            for condition in conditions:
                begin = time.monotonic()
                intervention.reset_stats()
                correct = 0
                with intervention.apply(condition):
                    for case in cases:
                        generated = generate(case["prompt"], True, args.max_new_tokens)
                        passed = score(case, generated["text"]) and not generated["unexpected_special_tokens"]
                        correct += passed
                        save(dict(condition=condition.name, panel="quality", case_id=case["id"],
                                  category=case["category"], prompt=case["prompt"], expected=case["expected"],
                                  correct=passed, metrics=text_metrics(generated["text"]), **generated))
                    for i, prompt in enumerate(VALENCE_PROMPTS):
                        generated = generate(prompt, False, args.valence_tokens)
                        save(dict(condition=condition.name, panel="valence", case_id=f"valence_{i}",
                                  category="style_proxy", prompt=prompt,
                                  metrics=text_metrics(generated["text"]), **generated))
                    for i, entry in enumerate(heldout):
                        # Heldout entries support either strings or category/text dictionaries.
                        text = entry if isinstance(entry, str) else entry["text"]
                        category = "neutral" if isinstance(entry, str) else entry["category"]
                        save(dict(condition=condition.name, panel="nll", case_id=f"nll_{i}",
                                  category=category, prompt=text, **nll(text)))
                stats = intervention.stats()
                write_json(args.out / f"telemetry_{condition.name}.json", stats)
                print(f"{condition.name:16s} {correct:2d}/{len(cases)} accuracy | {time.monotonic()-begin:.1f}s", flush=True)
        manifest.update(status="complete", elapsed_seconds=time.monotonic()-started,
                        finished_utc=datetime.now(timezone.utc).isoformat(),
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if args.device=="cuda" else None)
        write_json(args.out / "manifest.json", manifest)
        from report import build_report
        build_report(args.out)
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic()-started)
        write_json(args.out / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
