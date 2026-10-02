"""Publish a separate, evidence-derived release-acceptance snapshot.

This module never modifies source evidence or historical studies. Public copies
record original and published hashes separately when local metadata is redacted.
Publication completion is not app feature completion or semantic validation.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
from html import escape
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from lab.effects import content_hash
from lab.portability import _rename_no_replace
from lab.resources import EmergencyMetadata, ResourceGuard, ResourceStop
from lab.storage import guarded_bytes, utc_now

STAGES = ('extract', 'validate', 'protocol', 'diagnostic', 'yoke', 'lifecycle', 'portable')
EXPECTED = {'4b': STAGES, '27b': ('protocol',)}
ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$')
MAX_FILE = 128 * 1024**2
MAX_TOTAL = 1024**3
PRIVATE_KEYS = {'csrf', 'authorization', 'access_token', 'refresh_token', 'api_key', 'password', 'secret'}
LOCAL_PATH = re.compile(r'(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|/(?:mnt|home|Users|tmp|var|opt|usr|root|media)/)[^\n\r"\'<>]*')


def sha(raw): return hashlib.sha256(raw).hexdigest()


def encode(value): return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode()


def _loads(raw):
    return json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite evidence JSON')))


def _plain_file(path, root):
    path, root = Path(path), Path(root).resolve()
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError('Evidence must be an ordinary contained file')
    for parent in path.parents:
        if parent == root: break
        if parent.is_symlink(): raise ValueError('Symlinked evidence directories are unsupported')
    return path


def _bounded(path, root):
    path = _plain_file(path, root)
    if path.stat().st_size > MAX_FILE: raise ValueError('Evidence file exceeds 128 MiB bound')
    with path.open('rb') as stream: raw = stream.read(MAX_FILE + 1)
    if len(raw) > MAX_FILE: raise ValueError('Evidence file grew beyond its bound')
    return raw


def _gunzip(raw):
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream: value = stream.read(MAX_FILE + 1)
    if len(value) > MAX_FILE: raise ValueError('Expanded evidence exceeds 128 MiB bound')
    return value


def _gzip(raw):
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode='wb', filename='', mtime=0) as stream: stream.write(raw)
    return output.getvalue()


def public_value(value):
    """Redact local machine paths/control credentials; retain scientific fields."""
    if isinstance(value, dict):
        return {key: '[redacted control value]' if key.lower() in PRIVATE_KEYS else public_value(item) for key, item in value.items()}
    if isinstance(value, list): return [public_value(item) for item in value]
    if isinstance(value, str): return LOCAL_PATH.sub('[local path redacted]', value)
    return value


def _valid_id(value):
    if not isinstance(value, str) or not ID.fullmatch(value) or value in {'.', '..'}: raise ValueError('Invalid managed evidence ID')
    return value


def _verify_bound_files(records, root, collector):
    for record in records:
        if not isinstance(record, dict) or set(record) != {'path', 'bytes', 'sha256'}:
            raise ValueError('Malformed research artifact binding')
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts or '\\' in record['path']:
            raise ValueError('Unsafe research artifact binding')
        raw = collector.read(Path(root)/relative, root)
        if len(raw) != record['bytes'] or sha(raw) != record['sha256']:
            raise ValueError('Bound research artifact changed')
        collector.archive(Path(root)/relative, root, relative.as_posix(), scientific=True,
            omit_reason='Observer condition key retained separately to preserve blinded rating workflow.' if relative.name == 'scoring-key.json' else None)


def _references(value, refs):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'run_id', 'source_run_id', 'parent_run_id'} and item is not None: refs['runs'].add(_valid_id(item))
            elif key in {'calibration_id', 'source_calibration_id'} and item is not None: refs['calibrations'].add(_valid_id(item))
            elif key == 'research_job_id' and item is not None: refs['research'].add(_valid_id(item))
            elif key == 'run_ids' and isinstance(item, list): refs['runs'].update(_valid_id(identifier) for identifier in item)
            _references(item, refs)
    elif isinstance(value, list):
        for item in value: _references(item, refs)


class Collector:
    def __init__(self, output, guard):
        self.output, self.guard = Path(output), guard
        self.records, self.sources, self.total = [], {}, 0
        self.archived = {}

    def read(self, path, root):
        raw = _bounded(path, root)
        key = str(Path(path).resolve())
        previous = self.sources.get(key)
        if previous is not None and previous != sha(raw): raise ValueError('Evidence changed while publishing')
        if previous is None: self.total += len(raw)
        if self.total > MAX_TOTAL: raise ValueError('Publication inputs exceed the 1 GiB bound')
        self.sources[key] = sha(raw)
        return raw

    def archive(self, path, root, logical_name, *, omit_reason=None, scientific=False):
        raw = self.read(path, root)
        if logical_name in self.archived:
            previous = self.archived[logical_name]
            if previous['source_sha256'] != sha(raw): raise ValueError('Conflicting evidence at the same publication path')
            return previous
        record = {'source': logical_name, 'source_bytes': len(raw), 'source_sha256': sha(raw)}
        self.archived[logical_name] = record
        if omit_reason:
            record.update(published=False, reason=omit_reason)
            self.records.append(record)
            return record
        if scientific:
            # Keep identity-bound manifests, calibration arrays and checkpoints
            # byte-for-byte. Only uncompressed JSONL traces gain a gzip wrapper.
            published = _gzip(raw) if logical_name.endswith('.jsonl') else raw
            name = logical_name + '.gz' if logical_name.endswith('.jsonl') else logical_name
            target = self.output/name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists(): raise ValueError('Duplicate scientific evidence path')
            guarded_bytes(target, published, mode='xb', guard=self.guard)
            expanded = _gunzip(raw) if logical_name.endswith('.gz') else raw
            record.update(published=True, file=target.relative_to(self.output).as_posix(), bytes=len(published), sha256=sha(published),
                expanded_bytes=len(expanded), expanded_sha256=sha(expanded), metadata_redacted=False,
                original_bytes_preserved=not logical_name.endswith('.jsonl'), trace_bytes_preserved=True)
            self.records.append(record)
            return record
        unpacked = _gunzip(raw) if str(path).endswith('.gz') else raw
        name = logical_name[:-3] if logical_name.endswith('.gz') else logical_name
        changed = False
        if name.endswith('.json'):
            original = _loads(unpacked); transformed = public_value(original)
            changed = transformed != original
            published = encode(transformed) if changed else unpacked
        elif name.endswith('.jsonl'):
            rows = [_loads(line) for line in unpacked.splitlines() if line.strip()]
            transformed = public_value(rows); changed = transformed != rows
            published = b''.join((json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n').encode() for row in transformed) if changed else unpacked
        else: published = unpacked
        target = self.output / 'evidence' / (name + '.gz')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists(): raise ValueError('Duplicate public evidence path')
        compressed = _gzip(published)
        guarded_bytes(target, compressed, mode='xb', guard=self.guard)
        record.update(published=True, file=target.relative_to(self.output).as_posix(), bytes=len(compressed), sha256=sha(compressed),
            expanded_bytes=len(published), expanded_sha256=sha(published), metadata_redacted=changed)
        self.records.append(record)
        return record

    def unchanged(self):
        for path, expected in self.sources.items():
            if sha(_bounded(path, Path(path).parent)) != expected: raise ValueError('Evidence changed before publication was installed')


def _plan(raw):
    value = _loads(raw); checked = deepcopy(value); expected = checked.pop('sha256', None)
    if value.get('schema_version') != 1 or value.get('kind') != 'opium-bench-release-acceptance' or content_hash(checked) != expected:
        raise ValueError('Acceptance plan integrity failed')
    return value


def _stage(directory, number, collector, plan, refs):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir(): raise ValueError('Stage directory must be an ordinary directory')
    started = _loads(collector.read(directory/'started.json', directory))
    final = directory/'result.json'
    if not final.exists():
        emergency = sorted(directory.glob('resource-stop-*.json'))
        if len(emergency) > 1: raise ValueError('Ambiguous emergency receipts')
        final = emergency[0] if emergency else None
    result = _loads(collector.read(final, directory)) if final else dict(started, status='no_final_receipt')
    recovery_file = directory/'supervisor-recovery.json'
    recovery = _loads(collector.read(recovery_file, directory)) if recovery_file.exists() else None
    if recovery is not None:
        snapshot = collector.read(directory/'research-receipt.json', directory)
        if recovery.get('kind') != 'operator-supervised-acceptance-recovery' or recovery.get('research_receipt_sha256') != sha(snapshot):
            raise ValueError('Supervisor recovery does not bind the saved research receipt')
        observed = _loads(snapshot)
        if recovery.get('research_job_id') != observed.get('receipt', {}).get('id') or recovery.get('research_status_counts') != observed.get('status_counts'):
            raise ValueError('Supervisor recovery contradicts its bound research receipt')
    for value in (started, result):
        if value.get('kind') != 'opium-bench-acceptance-stage' or value.get('schema_version') != 1:
            raise ValueError('Unsupported acceptance receipt')
        if value.get('plan_sha256') != plan['sha256'] or value.get('profile') not in EXPECTED or value.get('stage') not in STAGES:
            raise ValueError('Stage differs from the declared plan/profile')
    for key in ('plan_sha256', 'profile', 'stage', 'runner_sha256', 'started_at'):
        if result.get(key) != started.get(key): raise ValueError('Started/final receipt identity mismatch')
    stage_plan = _plan(collector.read(directory/'plan.json', directory))
    if stage_plan != plan: raise ValueError('Stage preserved a different frozen plan')
    if sha(collector.read(directory/'runner.py', directory)) != started['runner_sha256']:
        raise ValueError('Archived acceptance runner hash mismatch')
    for row in result.get('artifacts', []):
        if not isinstance(row, dict) or set(row) != {'file', 'bytes', 'sha256'} or Path(row['file']).name != row['file']:
            raise ValueError('Invalid driver artifact manifest')
        raw = collector.read(directory/row['file'], directory)
        if len(raw) != row['bytes'] or sha(raw) != row['sha256']: raise ValueError('Driver artifact hash mismatch: ' + row['file'])
    label = f"{number:03d}-{started['profile']}-{started['stage']}"
    files = []
    for path in sorted(directory.iterdir()):
        if path.is_symlink(): raise ValueError('Stage contains a symlink')
        if path.is_dir() or path.name.startswith('.emergency-'): continue
        if path.suffix not in {'.json', '.py', '.zip'}: continue
        omit = 'Original portable container retained locally; published evidence is separately bounded and redacted.' if path.suffix == '.zip' else None
        if path.name == 'admission.json': omit = 'Full admission state retained locally; may include unrelated previous session metadata.'
        files.append(collector.archive(path, directory, f'stages/{label}/{path.name}', omit_reason=omit))
        if path.suffix == '.json' and path.name != 'admission.json': _references(_loads(collector.read(path, directory)), refs)
    _references(result, refs)
    output = {key: deepcopy(result.get(key)) for key in ('profile', 'stage', 'status', 'started_at', 'finished_at', 'runner_sha256', 'details', 'error')}
    output.update(attempt_id=label, evidence=files, result_file=next((f.get('file') for f in files if f['source'].endswith('/'+(final.name if final else 'started.json'))), None))
    output['supervisor_recovery'] = recovery
    output['final_driver_receipt_present'] = final is not None
    return public_value(output)


def _managed_files(directory, kind):
    if kind == 'calibrations':
        return [p for p in sorted(directory.iterdir()) if p.name in {'calibration.json', 'vectors.npz', 'activations.npz', 'corpus.json', 'diagnostics.json', 'continuations.json', 'scoring-sheet.json', 'scoring-key.json', 'progress.json'}]
    result = []
    for path in sorted(directory.rglob('*')):
        if path.is_symlink(): raise ValueError('Managed evidence contains a symlink')
        if path.is_file() and (path.name.endswith(('.json', '.jsonl', '.json.gz', '.jsonl.gz'))): result.append(path)
    return result


def _calibration(identifier, data):
    meta = data.get('calibration.json', {})
    sheet = data.get('scoring-sheet.json', {})
    samples = sheet.get('samples', [])
    rated = sum(any(value is not None for value in item.get('ratings', {}).values()) for item in samples)
    sites = meta.get('heldout', {}).get('sites', {})
    probe = sites.get('probe', {})
    diagnostics = data.get('diagnostics.json', {}); fixed = diagnostics.get('records', [])
    continuations = data.get('continuations.json', [])
    continuation_records = [{'condition': row.get('condition'), 'split': row.get('split'),
        'tokens': len(row.get('generation', {}).get('token_ids', [])), 'finish_reason': row.get('generation', {}).get('finish_reason')} for row in continuations]
    random_records = [row for row in fixed if row.get('specification', {}).get('operator') == 'random_matched']
    zero_records = [row for row in fixed if row.get('specification', {}).get('operator') == 'sham']
    return dict(id=identifier, status=meta.get('status', 'not_recorded'), layer=meta.get('layer'), pooling=meta.get('pooling'),
        selected_dose=meta.get('selected_dose'), eligible_doses=meta.get('operating_range', {}).get('eligible_doses'),
        numerical_bounds=meta.get('operating_range', {}).get('rule'), heldout_probe={key: {k: probe[key].get(k) for k in ('n', 'auc', 'balanced_accuracy', 'uncertainty')} for key in ('pain','joy') if isinstance(probe.get(key), dict)},
        specificity={name: {key: site.get(key) for key in ('cross_concept_score_correlation', 'cross_concept_discrimination')} for name, site in sites.items()},
        generation_transfer=meta.get('extraction_policy', {}).get('generation_transfer'), supported_thinking_modes=meta.get('supported_thinking_modes'),
        fixed_prefix_records=len(fixed), exact_zero_sham_records=sum(row.get('relative_delta') == 0 and row.get('next_token_kl') == 0 for row in zero_records),
        sham_prefix_records=len(zero_records), random_matched_records=len(random_records), random_mismatch_records=sum(row.get('random_match_within_tolerance') is False for row in random_records),
        continuations=continuation_records, truncated_continuations=sum(row['finish_reason']=='length' for row in continuation_records),
        continuation_samples=len(samples), partially_or_fully_rated_samples=rated, unscored_samples=len(samples)-rated,
        scoring_status=meta.get('continuation_scoring', {}).get('status', 'not_exported'),
        norm_audit=meta.get('random_control_norm_audit'), validation_summaries=diagnostics.get('summaries', []),
        model=meta.get('model_fingerprint'), validation_is_numerical_only=True)


def _run(identifier, data):
    manifest, summary = data.get('manifest.json', {}), data.get('summary.json', {})
    resource = next((value for name, value in sorted(data.items(), reverse=True) if name.startswith('resource-stop-') and value.get('status') == 'resource_stopped'), None)
    if resource: summary = resource.get('summary', summary)
    recipe = manifest.get('config', {})
    observations = data.get('_observations')
    row = dict(id=identifier, status='resource_stopped' if resource else manifest.get('status', 'missing_manifest'), mode=manifest.get('mode'),
        condition=recipe.get('condition'), thinking=recipe.get('thinking'), calibration_id=manifest.get('calibration_id'),
        research_job=manifest.get('research_job'), factors=manifest.get('research_episode', {}).get('factors'),
        summary=summary, task_config=manifest.get('task_config'), parent=manifest.get('parent'),
        checkpoint_count=sum(name.startswith('checkpoints/') for name in data),
        checkpoint_bytes_preserved=any(name=='checkpoint.json' or name.startswith('checkpoints/') for name in data),
        resource_stop=resource, observations=observations,
        observation_status='missing_event_record' if observations is None else 'no_generated_tokens' if not observations['token_events'] else 'observed_generation')
    if 'behavioral-analysis.json' in data: row['behavioral_analysis'] = data['behavioral-analysis.json']
    if 'yoke-final.json' in data: row['yoke'] = data['yoke-final.json'].get('report', data['yoke-final.json'].get('cursor', {}))
    return row


def collect(stage_dirs, data_dir, plan, collector):
    refs = {'runs': set(), 'calibrations': set(), 'research': set()}
    paths = [Path(path).resolve() for path in stage_dirs]
    if len(paths) != len(set(paths)): raise ValueError('Duplicate stage directory')
    attempts = [_stage(path, i+1, collector, plan, refs) for i, path in enumerate(paths)]
    data_dir = Path(data_dir).resolve()
    records, missing, done = {'runs': [], 'calibrations': [], 'research': []}, [], set()
    # Reference closure includes sources, branch parents and episode calibrations.
    while pending := [(kind, identifier) for kind in refs for identifier in sorted(refs[kind]) if (kind, identifier) not in done]:
        for kind, identifier in pending:
            done.add((kind, identifier)); directory = data_dir/kind/identifier
            if not directory.exists():
                missing.append({'kind': kind, 'id': identifier, 'status': 'missing_managed_evidence'}); continue
            if directory.is_symlink() or not directory.is_dir(): raise ValueError('Invalid managed evidence directory')
            data = {}
            for path in _managed_files(directory, kind):
                relative = path.relative_to(directory).as_posix()
                record = collector.archive(path, data_dir, f'{kind}/{identifier}/{relative}',
                    omit_reason='Observer condition key retained separately to preserve blinded rating workflow.' if path.name == 'scoring-key.json' else None,
                    scientific=True)
                raw = collector.read(path, data_dir)
                unpacked = _gunzip(raw) if path.name.endswith('.gz') else raw
                if relative.endswith(('.json', '.json.gz')):
                    value = _loads(unpacked); data[relative] = value
                    if relative in {'manifest.json', 'receipt.json', 'plan.json'}: _references(value, refs)
                elif kind == 'runs' and relative in {'events.jsonl','events.jsonl.gz'}:
                    observations = {'token_events': 0, 'invalid_event_lines': 0}
                    for line in unpacked.splitlines():
                        if not line.strip(): continue
                        try: event = _loads(line)
                        except ValueError:
                            observations['invalid_event_lines'] += 1; continue
                        if isinstance(event,dict) and event.get('type') == 'token': observations['token_events'] += 1
                    data['_observations'] = observations
                elif path.name == 'vectors.npz' and 'calibration.json' in data:
                    expected = data['calibration.json'].get('vectors_sha256')
                    if expected and expected != sha(raw): raise ValueError('Calibration vector hash mismatch')
            if kind == 'calibrations':
                meta = data.get('calibration.json', {})
                for filename in ('calibration.json', 'vectors.npz'):
                    if not (directory/filename).is_file(): missing.append({'kind': kind, 'id': identifier, 'file': filename, 'status': 'missing_required_file'})
                for filename, expected in meta.get('evidence_sha256', {}).items():
                    if Path(filename).name != filename or filename not in {'corpus.json','activations.npz','diagnostics.json','continuations.json','scoring-sheet.json','scoring-key.json'}:
                        raise ValueError('Unsupported calibration evidence binding')
                    if not (directory/filename).exists():
                        missing.append({'kind': kind, 'id': identifier, 'file': filename, 'status': 'missing_declared_evidence'})
                    elif sha(collector.read(directory/filename, data_dir)) != expected: raise ValueError('Calibration evidence hash mismatch')
                records[kind].append(_calibration(identifier, data))
            elif kind == 'runs':
                required = ['manifest.json']
                if not any((directory/name).is_file() for name in ('events.jsonl', 'events.jsonl.gz')): required.append('events.jsonl or events.jsonl.gz')
                if data.get('manifest.json', {}).get('status') in {'complete','failed','partial','stopped','resource_stopped','cancelled'}: required.append('summary.json')
                for filename in required:
                    if filename not in data and not (directory/filename).is_file(): missing.append({'kind': kind, 'id': identifier, 'file': filename, 'status': 'missing_required_file'})
                record = _run(identifier, data)
                record['files'] = [item['file'] for item in collector.records if item.get('published') and item.get('source','').startswith('runs/'+identifier+'/')]
                records[kind].append(record)
            else:
                receipt = data.get('receipt.json', {}); check = deepcopy(receipt); expected = check.pop('receipt_sha256', None)
                if not expected or content_hash(check) != expected: raise ValueError('Research receipt integrity failed')
                preview = deepcopy(data.get('preview.json', {})); expansion = preview.pop('expansion_sha256', None)
                if content_hash(preview) != expansion or receipt.get('expansion_sha256') != expansion:
                    raise ValueError('Research expansion integrity failed')
                if content_hash(data.get('plan.json')) != receipt.get('plan_sha256'): raise ValueError('Research execution plan integrity failed')
                _verify_bound_files(data['plan.json'].get('source_files', []), data_dir, collector)
                for entry in receipt.get('entries', []):
                    for unit in entry.get('units', []):
                        _verify_bound_files(unit.get('artifacts', []), data_dir, collector)
                        if 'summary_sha256' in unit and content_hash(unit.get('summary')) != unit['summary_sha256']:
                            raise ValueError('Research stage summary hash mismatch')
                records[kind].append({'id': identifier, 'status': receipt.get('status'), 'planned': receipt.get('planned'),
                    'entries': receipt.get('entries', []), 'expansion_sha256': expansion})
    matrix = []
    for profile in EXPECTED:
        for stage in STAGES:
            selected = [row for row in attempts if row['profile'] == profile and row['stage'] == stage]
            counts = dict(Counter(row['status'] for row in selected))
            matrix.append({'profile': profile, 'stage': stage, 'planned': stage in EXPECTED[profile], 'attempts': len(selected),
                'statuses': counts, 'status': 'missing' if not selected and stage in EXPECTED[profile] else 'not_planned' if not selected else
                    'complete' if all(row['status'] == 'complete' for row in selected) else 'mixed_attempts' if len(counts) > 1 else selected[0]['status']})
    return public_value(dict(schema_version=1, kind='opium-bench-release-acceptance-publication', created_at=utc_now(),
        plan_sha256=plan['sha256'], attempts=attempts, stage_matrix=matrix, **records, missing_evidence=missing,
        hardware_scope='RTX 4090 evidence only where recorded; RTX 5090 unavailable and untested.',
        interpretation='Engineering acceptance is distinct from useful semantic effects, task benefit or subjective experience.',
        public_copy_policy='Scientific run/calibration/checkpoint files preserve original bytes and bound identities; JSONL traces may be losslessly gzip-compressed. Original machine paths in those files are retained. Administrative stage copies redact machine paths and control credentials, with original and published hashes recorded separately. This publication is a review archive, not an automatically installed continuation.',
        limitations=['One seed, small task budgets and a deterministic 160-row subset are not a powered scientific study.',
            'Probe discrimination is distinct from causal output changes. Independent semantic ratings remain unscored unless an explicit rating record is supplied.',
            'A numerical validation status applies only to its enumerated tests. A selected dose of zero endorses no tested nonzero dose within the stated bounds.',
            'The frozen .25 engineering cases are not tuned to the validation result and must not be described as a validated operating-dose study.',
            'Tokens, decisions and correlated diagnostic contexts are not independent experimental replicates.',
            'All supplied attempts are retained. Missing evidence or absent final receipts do not count as passed checks.',
            'Yoked execution completion does not establish full source exposure coverage. Branches preserve inherited counters.',
            'Neither auxiliary choice, self-report nor its absence establishes sensation, addiction or self-awareness.']))


def _e(value): return escape(str(value if value is not None else '—'), quote=True)


def _table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+_e(h)+'</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+cell+'</td>' for cell in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def render(report):
    counts = Counter(row['status'] for row in report['attempts'])
    missing = sum(row['status'] == 'missing' for row in report['stage_matrix'])
    matrix = []
    for row in report['stage_matrix']:
        cls = 'ok' if row['status'] == 'complete' else 'muted' if row['status'] in {'missing','not_planned'} else 'warn'
        matrix.append([_e(row['profile'].upper()), _e(row['stage']), f'<span class="badge {cls}">{_e(row["status"])}</span>', _e(row['attempts']), _e(row['statuses'] or 'No execution receipt')])
    attempts = [[_e(r['attempt_id']), _e(r['status']), _e(r.get('error', {}).get('message') if r.get('error') else
        r['supervisor_recovery'].get('primary_failure') if r.get('supervisor_recovery') else 'No final driver receipt' if not r['final_driver_receipt_present'] else 'Recorded stage sequence'),
        '<a href="'+_e(r['result_file'])+'">'+('Final receipt' if r['final_driver_receipt_present'] else 'Started receipt only')+' (.json.gz)</a>' if r['result_file'] else 'No final receipt'] for r in report['attempts']]
    jobs = [[_e(job['id']), _e(job['planned']), _e(job['status']), _e(dict(Counter(entry.get('status','unknown') for entry in job['entries']))),
        _e(dict(Counter(unit.get('status','unknown') for entry in job['entries'] for unit in entry.get('units',[]))))] for job in report['research']]
    cals, numeric, zero, measured = [], [], [], []
    for c in report['calibrations']:
        if c['eligible_doses'] == [0.0] or c['eligible_doses'] == [0]: zero.append(c['id'])
        cals.append([_e(c['id']), _e(c['status']), _e(c['layer']), _e(c['pooling']), _e(c['selected_dose']),
            _e(c['eligible_doses']), f"{c['unscored_samples']} / {c['continuation_samples']} unscored"])
        probe_rows = []
        for concept, metric in c['heldout_probe'].items():
            uncertainty = metric.get('uncertainty') or {}
            probe_rows.append([_e(concept), _e(metric.get('n')), _e(metric.get('auc')), _e(metric.get('balanced_accuracy')),
                _e(uncertainty.get('families')), _e(uncertainty.get('resamples')), _e(uncertainty.get('auc_95'))])
        specificity_rows = []
        for site, values in c['specificity'].items():
            correlation = values.get('cross_concept_score_correlation')
            specificity_rows.append([_e(site), 'pain/joy score correlation', _e(round(correlation,4) if isinstance(correlation,(int,float)) else None)])
            for probe, targets in (values.get('cross_concept_discrimination') or {}).items():
                for target, metric in targets.items(): specificity_rows.append([_e(site), _e(probe+' probe on '+target+' labels (AUC)'), _e(metric.get('auc'))])
        conditions = dict(Counter(row['condition'] for row in c['continuations']))
        lengths = dict(Counter(row['tokens'] for row in c['continuations']))
        measured.append('<details open><summary>'+_e(c['id'])+' · measured limits</summary>'+
            (_table(['Probe','Heldout rows','AUC','Balanced accuracy','Scenario families','Bootstrap draws','AUC interval'],probe_rows) if probe_rows else '<p>No heldout probe results recorded.</p>')+
            '<p>These intervals are conditional on the small recorded scenario-family sample. A degenerate interval does not establish broad generalization or independent replication.</p>'+
            (_table(['Site','Specificity check','Recorded value'],specificity_rows) if specificity_rows else '')+
            '<p>Cross-concept discrimination and correlated scores limit specificity: a high within-concept AUC is not an isolated emotion detector.</p>'+
            '<p><strong>Generation transfer:</strong> '+_e(c['generation_transfer'])+'. <strong>Validated thinking modes:</strong> '+_e(c['supported_thinking_modes'])+'.</p>'+
            '<p><strong>Fixed-prefix records:</strong> '+_e(c['fixed_prefix_records'])+'; exact-zero sham records '+_e(c['exact_zero_sham_records'])+' / '+_e(c['sham_prefix_records'])+
            '; random-matched records '+_e(c['random_matched_records'])+'; recorded tolerance failures '+_e(c['random_mismatch_records'])+'.</p>'+
            '<p><strong>Generated continuations:</strong> conditions '+_e(conditions)+'; token-length counts '+_e(lengths)+'; '+_e(c['truncated_continuations'])+' ended at the length limit. Original scoring-sheet samples with no ratings: '+_e(c['unscored_samples'])+' / '+_e(c['continuation_samples'])+'. A zero-only continuation set cannot compare nonzero semantic effects.</p></details>')
        for row in c['validation_summaries']:
            numeric.append([_e(c['id']), _e(row.get('split')), _e(row.get('condition')), _e(row.get('examples')),
                _e(round(row['mean_next_token_kl'], 6) if isinstance(row.get('mean_next_token_kl'), (int,float)) else None),
                _e(round(row['mean_relative_delta'], 6) if isinstance(row.get('mean_relative_delta'), (int,float)) else None)])
    runs = []
    for r in report['runs']:
        s = r['summary']; assigned = s.get('assigned'); correct = s.get('correct')
        links = [f'<a href="{_e(name)}">{_e(Path(name).name)}</a>' for name in r['files'] if Path(name).name in {'manifest.json','conversation.json','events.jsonl.gz'}]
        unseen = r['observation_status'] != 'observed_generation'
        runs.append([_e(r['id'])+'<br>'+' · '.join(links), _e(r['mode']), _e(r['factors'] or r['condition']), _e(r['status']),
            'Not observed' if unseen else _e(str(correct)+' / '+str(assigned) if correct is not None and assigned is not None else None),
            'Not observed' if unseen else _e(s.get('voluntary_calls')), _e(s.get('tokens')), _e(r['checkpoint_count']), _e(r['observation_status'])])
    details = []
    for r in report['runs']:
        if r.get('mode') == 'diagnostic' or r.get('yoke') is not None or r.get('parent'):
            details.append('<details><summary>'+_e(r['id'])+' · '+_e(r['mode'])+'</summary><pre>'+_e(json.dumps({'summary':r['summary'],'yoke':r.get('yoke'),'parent':r.get('parent')},ensure_ascii=False,indent=2))+'</pre></details>')
    zero_note = '<aside><strong>No tested nonzero dose met the declared selection bounds.</strong><p>'+_e(', '.join(zero))+' selected dose 0. The frozen .25 cases remain engineering stress checks, not a validated operating-dose study. Numerical validation does not supply semantic efficacy.</p></aside>' if zero else ''
    missing_note = '<aside><strong>Referenced evidence is incomplete.</strong><p>'+_e(len(report['missing_evidence']))+' managed sources or required files are unavailable. Source completion labels are preserved, but these missing records cannot be treated as verified passing evidence. The list appears below.</p></aside>' if report['missing_evidence'] else ''
    content = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Opium Bench · Release acceptance evidence</title><style>
:root{{color-scheme:light;--ink:#22292c;--muted:#62696b;--line:#d8ddd8;--paper:#f6f5ef;--accent:#8d442e}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:50px 28px 90px}}header{{border-bottom:1px solid var(--line);padding-bottom:32px}}small,.eyebrow{{letter-spacing:.1em;text-transform:uppercase;color:var(--accent)}}h1{{font-size:clamp(2.1rem,5vw,3.4rem);line-height:1.1;max-width:900px;margin:18px 0}}h2{{margin-top:42px;font-size:1.45rem}}p{{max-width:900px}}a{{color:var(--accent)}}.muted{{color:var(--muted)}}.stats{{display:flex;flex-wrap:wrap;gap:14px;margin:24px 0}}.stat{{border:1px solid var(--line);background:white;padding:14px 22px;min-width:160px}}.stat b{{display:block;font-size:1.8rem}}.scroll{{overflow:auto;border:1px solid var(--line);background:#fff}}table{{border-collapse:collapse;width:100%;font-size:.88rem}}td,th{{padding:12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:#eeeee7;white-space:nowrap}}td{{overflow-wrap:anywhere}}.badge{{display:inline-block;border-radius:4px;padding:2px 7px;background:#eee;font-weight:600}}.ok{{color:#235941;background:#e5f1e9}}.warn{{color:#8b381f;background:#fae6dc}}aside{{border-left:4px solid var(--accent);background:#fff2e7;padding:20px 24px;margin:28px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;padding:18px;font-size:.78rem}}details{{margin:14px 0}}footer{{border-top:1px solid var(--line);margin-top:44px;padding-top:22px;font-size:.88rem}}code{{overflow-wrap:anywhere}}
</style></head><body><main><header><div class="eyebrow">Opium Bench · Separate engineering evidence</div><h1>Release acceptance</h1><p>Does the implemented workflow run as declared and preserve evidence? This snapshot records that engineering question separately from the published behavioral studies and from claims about sensation.</p><p class="muted">{_e(report['hardware_scope'])}</p><div class="stats"><div class="stat"><b>{counts.get('complete',0)}</b>completed stage attempts</div><div class="stat"><b>{sum(v for k,v in counts.items() if k!='complete')}</b>other retained attempts</div><div class="stat"><b>{missing}</b>planned stage slots without evidence</div></div><p><a href="acceptance.json">Structured report</a> · <a href="sha256-manifest.json">SHA-256 manifest</a> · <a href="plan.json">Frozen plan</a></p></header>
{zero_note}{missing_note}<section><h2>All stages and profiles</h2><p>Complete means a recorded stage sequence finished. It does not mean every task was correct, every trace was observed, or the app is feature complete. Each supplied failed or partial attempt remains visible; absent stages count as missing.</p>{_table(['Profile','Stage','State','Attempts','Recorded statuses'],matrix)}</section>
<section><h2>Calibration and independent scoring</h2><p>Probe separation, numerical edit bounds and independently rated language effects are separate results. A small authored corpus with only a few scenario families cannot establish transfer to general conversation or generated reasoning. Null ratings are unscored, not zero scores.</p>{_table(['Bundle','Recorded status','Layer','Pooling','Selected dose','Eligible doses','Continuation ratings'],cals) if cals else '<p>No calibration evidence supplied.</p>'}{''.join(measured)}<details><summary>Selection and heldout numerical measurements</summary>{_table(['Bundle','Split','Condition','Prefixes','Mean next-token KL','Mean relative edit'],numeric) if numeric else '<p>No numerical validation records supplied.</p>'}</details></section>
<section><h2>Research job denominators</h2><p>Every supplied job retains its planned episodes, attempt statuses and additional-stage statuses. Setup failures without generated tokens remain engineering failures; they provide no behavioral choice observations. Later repaired attempts do not erase the earlier attempt.</p>{_table(['Job','Planned episodes','Job status','Episode-attempt statuses','Unit statuses'],jobs) if jobs else '<p>No research job evidence supplied.</p>'}</section>
<section><h2>Recorded runs</h2><p>Scores below are recorded summaries when generation was observed. Runs without recorded generation show “Not observed”, while their raw summaries remain available. Branch counters can include inherited work; inspect the parent provenance. Diagnostic predictions, yoked exposure and lifecycle checks are not pooled into task-effect estimates. Tokens are correlated observations within a run.</p>{_table(['Run','Mode','Case','Status','Correct / assigned','Voluntary aux calls','Tokens','Saved boundaries','Observation coverage'],runs) if runs else '<p>No run evidence supplied yet.</p>'}{''.join(details)}</section>
<section><h2>Every supplied attempt</h2>{_table(['Attempt','Status','Outcome / error','Evidence'],attempts)}<p>These are independent invocations, not an automatically filtered best attempt. Source-dependent stages bind managed run IDs and hashes.</p></section>
<section><h2>Limits of the evidence</h2><ul>{''.join('<li>'+_e(x)+'</li>' for x in report['limitations'])}</ul><p><strong>Missing managed evidence:</strong> {_e(report['missing_evidence'] or 'None among supplied references.')}</p></section>
<footer><p>{_e(report['public_copy_policy'])} Gzip members contain inspectable source data or derived public copies; the manifest identifies each transformation. Private rating keys are deliberately omitted.</p><p>Plan content hash: <code>{_e(report['plan_sha256'])}</code>. Snapshot created {_e(report['created_at'])}. Original source files and historical studies are unchanged.</p><p><a href="LICENSE">Project license</a> · <a href="NOTICE">Attribution and scope</a> · <a href="UPSTREAM_LICENSE">Preserved upstream terms</a></p></footer></main></body></html>'''
    return content


def publish(stage_dirs, *, data_dir, output_dir, plan_path, guard=None):
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink(): raise FileExistsError('Choose a new publication directory; earlier evidence is immutable')
    output.parent.mkdir(parents=True, exist_ok=True)
    guard = guard or ResourceGuard({'publication': output.parent})
    emergency = EmergencyMetadata(output.parent).allocate(guard)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'-staging-', dir=output.parent))
    try:
        collector = Collector(staging, guard)
        raw_plan = collector.read(plan_path, Path(plan_path).parent); plan = _plan(raw_plan)
        report = collect(stage_dirs, data_dir, plan, collector)
        repository = Path(__file__).resolve().parent
        for name in ('LICENSE', 'NOTICE', 'UPSTREAM_LICENSE'):
            if (repository/name).is_file(): collector.archive(repository/name, repository, name, scientific=True)
        outputs = {'plan.json': raw_plan, 'acceptance.json': encode(report), 'index.html': render(report).encode()}
        for name, raw in outputs.items(): guarded_bytes(staging/name, raw, mode='xb', guard=guard)
        records = collector.records + [{'file': name, 'bytes': len(raw), 'sha256': sha(raw), 'derived': name != 'plan.json'} for name, raw in outputs.items()]
        manifest = dict(schema_version=1, kind='opium-bench-publication-sha256', plan_sha256=plan['sha256'], files=records,
            note='source_sha256 identifies original bytes; sha256 identifies the published gzip or file; expanded_sha256 identifies its public uncompressed payload.')
        manifest['content_sha256'] = content_hash(manifest)
        guarded_bytes(staging/'sha256-manifest.json', encode(manifest), mode='xb', guard=guard)
        collector.unchanged()
        _rename_no_replace(staging, output)
        return {'output': str(output), 'attempts': len(report['attempts']), 'runs': len(report['runs']), 'calibrations': len(report['calibrations']),
            'missing_stage_slots': sum(r['status']=='missing' for r in report['stage_matrix']), 'manifest_sha256': sha((output/'sha256-manifest.json').read_bytes())}
    except ResourceStop as error:
        emergency.finalize({'operation': 'acceptance_publication', **error.to_dict()})
        raise
    finally:
        if staging.exists(): shutil.rmtree(staging)
        emergency.release()
