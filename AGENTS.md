# Repository instructions

## Continuous integration

- Do not create, modify, invoke, recommend, or rely on GitHub Actions unless the user explicitly requests an exception. Use local, manually invoked validation and release tooling.

## Research and implementation

- Qwen3-4B on the RTX 4090 is the default reference. Qwen3.8-27B at 4-bit precision is an optional target on both the RTX 4090 and RTX 5090, after actual memory and calibration validation.
- Prefer full GPU residency. Do not silently enable CPU or disk offloading.
- Use the reviewed `LynnColeArt/ai-hotbox` implementation as a reference, particularly `impossible_states`; retain upstream attribution. Do not import credentials or operational secrets from reference checkouts.
- Keep model weights, runtime environments, temporary downloads, and large caches off C: on the current machine. The WSL disk is backed by C:, so check host-volume free space as well as Linux filesystem capacity before large writes. Do not expand WSL automatically.
- Preserve immutable historical runs and source snapshots. Distinguish activation changes, behavioral effects, and claims about subjective experience.
- The full lab in `LAB_PLAN.html` is planned work. Do not describe its new protocols or controls as implemented until they are built and validated.
