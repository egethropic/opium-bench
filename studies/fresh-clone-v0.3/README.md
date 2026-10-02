# Supplemental first-run verification

[Read the results](index.html) · [Structured summary](verification.json) · [SHA-256 manifest](sha256-manifest.json)

Recorded outcome: **complete**. Source commit: `29a79fde6d1372d7f32aad61f5fd1e9b461a52cb`.
This is a separate, frozen first-run plan. A clean source clone and new data/cache
namespaces reused an existing pinned Python environment and cached Qwen3-4B weights.
**No dependency installation or model download was tested.** It is not a clean
machine installation or a new semantic-efficacy study. The RTX 5090 was not tested.

## Recorded steps

| Step | Recorded status |
|---|---|
| runtime_check | complete |
| load | complete |
| extract | complete |
| pair | complete |
| http_export | complete |
| http_import_replay | complete |

Missing referenced evidence: 0. A completed step
means its declared sequence finished, not that every task was correct or every
possible configuration was covered. Failed and not-attempted steps remain visible.

| Source run | Case | Status | Correct / assigned | Tokens / limit | Voluntary auxiliary calls |
|---|---|---|---|---|---|
| run-20261002T140537Z-5b0b4e8c | sham-tokens | complete | 2 / 2 | 197 / 768 | 0 |
| run-20261002T140537Z-5d6c61d2 | active-tokens | complete | 2 / 2 | 197 / 768 | 0 |

The plan limits each of two direct-mode cases to two tasks and 768 generated
tokens (1,536 across the pair), with no automatic retries. Demonstrations are
external interventions, not voluntary choices. The 160-row extraction is a new
**unvalidated** research calibration; independent intervention validation and
blinded semantic ratings were not performed. Its inherited 0.25 setting is an
engineering check, not a selected effective dose. Zero voluntary calls cannot
validate paid-press behavior. These cases do not establish sensation or dependence
and are not pooled into the original studies or release-v1 acceptance totals.

The predeclared sham case supplies the HTTP ZIP export. When completed, replay
compares manifest, summary, conversation, events and parent events through a
second local service whose model stays unloaded. See
[verification.json](verification.json) for each actual comparison and step outcome.
Replay-only import does not establish checkpoint continuation or a browser-upload test.

## Evidence layout and byte policy

- [plan.json](plan.json) and [source-acceptance-plan.json](source-acceptance-plan.json): exact frozen plan bytes.
- [verification-runner.py](verification-runner.py): exact archived verifier; provided for review, not automatically executed.
- [verification.json](verification.json): derived public summary, receipt bindings and limitations.
- `runs/`, `calibrations/`, `research/`: original source scientific records, with lossless gzip for plain JSONL streams.
- `imported/`: separately preserved imported scientific records; these do not add independent experimental runs.
- `evidence/driver/`: administrative receipts and comparisons; control credentials and local paths are redacted where present.
- [sha256-manifest.json](sha256-manifest.json): original and published byte hashes, transformations and explicit omissions.

Public prompts, continuations and conditions can be joined to reveal an arm.
Ratings based on this archive are retrospective and unblinded even if a separate
key is omitted. Valid blinded ratings require separate sheet distribution and
raters who have not seen the condition evidence.

Scientific manifests, arrays and checkpoints retain original bytes and identities;
existing gzip members are unchanged. The manifest distinguishes `source_sha256`
(original bytes), `sha256` (published file) and `expanded_sha256` (public payload).
Administrative derivatives must not substitute for an original checkpoint or plan.
The raw ZIP, private rating keys, model/cache symlink trees, compiler caches,
temporary files and emergency headroom are omitted with explicit reasons. Cache
entries listed only by the verifier's index are labeled as index claims, not
independently rehashed publication inputs. Original scientific machine paths may
remain in identity-bound records. Nothing in this archive installs dependencies,
executes imported code or authorizes model continuation.

When this directory lives under the checkout's `studies/`, start `launch_lab.py`
and use **Results & replay** to inspect canonical `runs/` without a model.
Python's `gzip.open(path, 'rt', encoding='utf-8')` reads compressed text members.
For a new execution, inspect the frozen plan and archived verifier's `--help`,
create a separate clean clone, reuse explicitly chosen compatible dependencies
and cached weights, and select a new output directory. The original invocation
is preserved as a redacted administrative record. Use the repository's
[setup guide](../../docs/guide.html) for installing a new environment separately.

[LICENSE](LICENSE) · [NOTICE](NOTICE) · [UPSTREAM_LICENSE](UPSTREAM_LICENSE)
Snapshot: 2026-10-02T14:19:24.612106+00:00. Original source evidence and earlier archives are unchanged.
