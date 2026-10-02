#!/usr/bin/env python3
"""Try a custom suppression/joy setting with the saved pilot directions."""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="Completed run containing manifest.json and vectors.npz")
    ap.add_argument("--hf-home", type=Path)
    ap.add_argument("--suppression", type=float, default=1.0)
    ap.add_argument("--joy-dose", type=float, default=1.0)
    ap.add_argument("--centered", action="store_true")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--raw", action="store_true", help="Continue the prompt as a raw prefix instead of chat")
    args = ap.parse_args()
    if args.max_new_tokens < 1:
        ap.error("max-new-tokens must be positive")
    if args.hf_home:
        os.environ["HF_HOME"] = str(args.hf_home.resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from intervention import Intervention, Condition
    condition = Condition("custom", args.suppression, args.joy_dose, args.centered)
    manifest = json.loads((args.run / "manifest.json").read_text())
    if manifest["status"] != "complete":
        ap.error("Select a completed run")
    torch.set_num_threads(8)
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model"], revision=manifest["revision"], device_map=args.device,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        attn_implementation="sdpa", local_files_only=True, trust_remote_code=False,
    ).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"], revision=manifest["revision"],
                                              local_files_only=True, trust_remote_code=False)
    with np.load(args.run / "vectors.npz", allow_pickle=False) as vectors:
        intervention = Intervention(model, manifest["layer"], vectors["pain"], vectors["joy_raw"],
                                    vectors["neutral"], float(vectors["scale"]), {}, manifest["token_scope"])
    prompt = args.prompt if args.raw else tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    with torch.inference_mode(), intervention.apply(condition):
        output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                pad_token_id=tokenizer.eos_token_id, use_cache=True)
    print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
