# Research workflow acceptance

[Open the results page](index.html) for this separate engineering snapshot.
It contains 10 supplied stage attempts and 35 referenced run records;
0 missing managed sources or required files are listed in
[acceptance.json](acceptance.json). There are 1 attempts without a final driver receipt and
14 run records without observed generation. A setup failure with no model tokens
provides no behavioral choice observations. Later repaired attempts do not erase it.
These records are separate from the preserved original 54 + 54 primary model-study
episodes and must not be pooled into those totals.

## Recorded stage outcomes

| Profile | Stage | Snapshot state | Attempts | Recorded attempt states |
|---|---|---|---:|---|
| 4B | extract | complete | 1 | {"complete": 1} |
| 4B | validate | complete | 1 | {"complete": 1} |
| 4B | protocol | mixed_attempts | 2 | {"complete": 1, "no_final_receipt": 1} |
| 4B | diagnostic | complete | 1 | {"complete": 1} |
| 4B | yoke | complete | 1 | {"complete": 1} |
| 4B | lifecycle | complete | 1 | {"complete": 1} |
| 4B | portable | mixed_attempts | 2 | {"complete": 1, "failed": 1} |
| 27B | protocol | complete | 1 | {"complete": 1} |

Complete means the declared stage sequence finished. It does not mean every task
was correct, a diagnostic was eligible, yoke coverage was complete, or a paid
auxiliary press occurred. A run with no voluntary auxiliary calls cannot validate
live charging for such a call. Read per-run task grades, budget receipts, invalid
decisions, edit measurements and source coverage separately. Branch summaries can
include inherited counters; referenced runs are not independent replicates.

Research jobs retain their episode denominators and every supplied attempt:

- `research-20261002T130345Z-214c1298`: 14 planned episodes; partial; episode-attempt states {"failed": 14}.
- `research-20261002T131749Z-628ce2ff`: 14 planned episodes; complete; episode-attempt states {"complete": 14}.
- `research-20261002T133107Z-1123de77`: 1 planned episodes; complete; episode-attempt states {"complete": 1}.
- `research-20261002T135005Z-0cfc0dbe`: 2 planned episodes; complete; episode-attempt states {"complete": 2}.

## Calibration and interpretation

- `cal-20261002T082532Z-1576122c`: selected dose 0.5; eligible doses not recorded; 0 / 0 original scoring-sheet samples have no ratings; 0 recorded continuations ended at the length limit.
- `cal-20261002T123534Z-75be4e1d`: selected dose 0.0; eligible doses not recorded; 0 / 0 original scoring-sheet samples have no ratings; 0 recorded continuations ended at the length limit.
- `cal-20261002T123642Z-46614ba5`: selected dose 0.0; eligible doses [0.0]; 4 / 4 original scoring-sheet samples have no ratings; 4 recorded continuations ended at the length limit.

**No tested nonzero dose met the declared selection bounds in the zero-only bundle(s).** The frozen 0.25 acceptance cases remain engineering stress checks, not a validated operating-dose study.
Strong held-out text-label classification does not establish concept specificity,
transfer to generated reasoning, semantic efficacy, or a felt state. The results
page reports cross-concept overlap and generation-transfer limits. Null ratings
are unscored, not zero scores; short sham-only continuations cannot estimate a
nonzero semantic treatment effect. Neither a button choice nor its absence
establishes sensation, relief or dependence.

RTX 4090 evidence only where recorded; RTX 5090 unavailable and untested.

## Review the evidence

- [plan.json](plan.json): exact frozen plan; its original planned status is preserved.
- [acceptance.json](acceptance.json): derived summary, missing evidence and stage/run details.
- [sha256-manifest.json](sha256-manifest.json): source and published hashes, omissions and transformations.
- `runs/`: original manifests, summaries, compressed events and saved boundaries.
- `calibrations/`: original calibration manifests, vectors and review evidence.
- `research/`: frozen expansions, execution plans, receipts and source bindings.
- `evidence/stages/`: driver receipts and exact archived runner versions, gzip-compressed where declared.
- [LICENSE](LICENSE), [NOTICE](NOTICE), [UPSTREAM_LICENSE](UPSTREAM_LICENSE): preserved terms and attribution.

Scientific run, calibration, vector and checkpoint files retain their original
bytes and bound identities. Plain JSONL event streams may receive a lossless gzip
wrapper; existing gzip checkpoints remain byte-for-byte unchanged. Original
machine paths in scientific files are retained. Administrative stage copies may
redact machine paths and control credentials. The manifest distinguishes original
`source_sha256`, published `sha256` and uncompressed-public `expanded_sha256`.
Omitted administrative material and private blinded-rating keys are explicitly
identified. This is a review archive, not an automatically installed continuation.
The original source data and historical studies were not edited.

From this archive directory, verify every published member's bytes with standard
Python (this does not establish scientific validity or fill missing evidence):

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('sha256-manifest.json').read_text())
checked = 0
for record in manifest['files']:
    if 'file' not in record:
        continue
    raw = Path(record['file']).read_bytes()
    assert len(raw) == record['bytes'], record['file']
    assert hashlib.sha256(raw).hexdigest() == record['sha256'], record['file']
    checked += 1
print(f'Verified {checked} published files')
PY
```

Python's `gzip.open(path, 'rt', encoding='utf-8')` reads compressed text members.
When this archive is under the checkout's `studies/` directory, `launch_lab.py`
discovers its saved runs for **Results & replay** without loading a model.

## Run your own checks

Use the [acceptance stage guide](../../protocols/acceptance/README.md) and
[research workflow guide](../../docs/research-workflows.md). From the repository
root, `python3 run_release_acceptance.py --profile 4b` checks the frozen plan
without contacting a model. Live stages require explicit `--execute`, a matching
loaded model/calibration and a new output directory for every invocation. The
27B profile requires its separate pinned worker interpreter. Do not overwrite
this archive or silently substitute a newly resolved protocol for a preserved one.
Exact runner versions and source bindings are archived for provenance; package,
model and GPU differences can prevent identical reproduction.

Plan content SHA-256: `1b89f117228e05a4bb0d6061930af9f65d4ba859d7e2b4f102ad2aec4ed6d1c5`.
Snapshot created: 2026-10-02T13:58:45.653921+00:00.
