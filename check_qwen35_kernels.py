"""Manually compare local Qwen3.5/3.8 GPU kernels with Torch reference functions.

Uses small synthetic tensors, never model weights. Run with the separate 27B
interpreter and put compiler caches on the selected data drive before execution.
"""
import argparse
import inspect
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicitly allow the small GPU check")
    parser.add_argument("--runtime-root", type=Path, required=True, help="Directory containing lab/")
    parser.add_argument("--output", type=Path, required=True, help="Write JSON parity receipt")
    args = parser.parse_args()
    if not args.run:
        parser.error("GPU work requires --run; this script is otherwise inert")
    sys.path.insert(0, str(args.runtime_root.resolve()))
    import torch
    from lab.runtime import _configure_qwen35_kernels

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the installed local kernels")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261002)
    torch.cuda.manual_seed_all(20261002)
    module, backend = _configure_qwen35_kernels(True)
    reference = {name: inspect.unwrap(fn) for name, fn in module._opium_bench_original_kernels.items()}
    dtype, device = torch.bfloat16, "cuda"
    checks = []

    def compare(label, actual, expected, *, relative_limit=.035, absolute_limit=.03):
        if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError(f"{label}: shape mismatch or nonfinite values")
        delta = actual.float() - expected.float()
        relative = float(delta.norm() / expected.float().norm().clamp_min(1e-8))
        maximum = float(delta.abs().max())
        passed = relative <= relative_limit and maximum <= absolute_limit
        checks.append(dict(label=label, shape=list(actual.shape), relative_l2=relative,
                           maximum_absolute=maximum, relative_limit=relative_limit,
                           absolute_limit=absolute_limit, passed=passed))
        if not passed:
            raise AssertionError(f"{label}: relative L2 {relative:.6g}, max absolute {maximum:.6g}")

    def rand(*shape, scale=1., kind=dtype):
        return torch.randn(shape, device=device, dtype=kind) * scale

    receipt = dict(status="running", backend=backend, torch=torch.__version__,
                   gpu=torch.cuda.get_device_name(), dtype="bfloat16", checks=checks,
                   note="Kernel-level numerical check, not whole-model or task equivalence; tolerances fixed before execution.")
    try:
        with torch.inference_mode():
            for length, heads, has_initial in ((17, 4, False), (65, 48, True)):
                q, k, v = (rand(1, length, heads, 128) for _ in range(3))
                g = -torch.nn.functional.softplus(rand(1, length, heads, kind=torch.float32)) * .25
                beta = rand(1, length, heads).sigmoid()
                state = rand(1, heads, 128, 128, scale=.03, kind=torch.float32) if has_initial else None
                kwargs = dict(g=g, beta=beta, initial_state=state, output_final_state=True,
                              use_qk_l2norm_in_kernel=True, cu_seqlens=None)
                ref_out, ref_state = reference["torch_chunk_gated_delta_rule"](q, k, v, **kwargs)
                out, fast_state = module.torch_chunk_gated_delta_rule(q, k, v, **kwargs)
                tag = f"prefill_T{length}_H{heads}"
                compare(tag + "_output", out, ref_out)
                compare(tag + "_state", fast_state, ref_state)
                for step in range(4):
                    q, k, v = (rand(1, 1, heads, 128) for _ in range(3))
                    g = -torch.nn.functional.softplus(rand(1, 1, heads, kind=torch.float32)) * .25
                    beta = rand(1, 1, heads).sigmoid()
                    kwargs = dict(g=g, beta=beta, output_final_state=True,
                                  use_qk_l2norm_in_kernel=True, cu_seqlens=None)
                    ref_out, ref_state = reference["torch_recurrent_gated_delta_rule"](
                        q, k, v, initial_state=ref_state, **kwargs)
                    out, fast_state = module.torch_recurrent_gated_delta_rule(
                        q, k, v, initial_state=fast_state, **kwargs)
                    compare(f"{tag}_decode{step + 1}_output", out, ref_out)
                    compare(f"{tag}_decode{step + 1}_state", fast_state, ref_state)

            for activation in (None, "silu"):
                x, weights = rand(1, 128, 17, scale=.25), rand(128, 4, scale=.25)
                bias = rand(128, scale=.1)
                expected = reference["causal_conv1d_fn"](x, weights, bias, activation=activation)
                actual = module.causal_conv1d_fn(x, weights, bias, activation=activation)
                tag = f"conv_{activation}"
                compare(tag + "_prefill", actual, expected, relative_limit=.015, absolute_limit=.004)
                ref_state, fast_state = x[:, :, -4:].contiguous().clone(), x[:, :, -4:].contiguous().clone()
                for step in range(4):
                    x = rand(1, 128, 1, scale=.25)
                    expected = reference["causal_conv1d_update"](x, ref_state, weights, bias, activation)
                    actual = module.causal_conv1d_update(x, fast_state, weights, bias, activation)
                    compare(f"{tag}_decode{step + 1}", actual, expected, relative_limit=.015, absolute_limit=.004)
                    compare(f"{tag}_state{step + 1}", fast_state, ref_state, relative_limit=0., absolute_limit=0.)
        torch.cuda.synchronize()
        receipt["status"] = "passed"
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(status=receipt["status"], checks=len(checks), output=str(args.output))))


if __name__ == "__main__":
    main()
