"""Immutable versioned recipe assets for the local experiment designer."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .effects import validate_preset
from .recipes_v2 import resolve_recipe
from .tool_definitions import validate_tool_set
from .storage import ID, read_json, utc_now

KINDS = {'recipe', 'effect_presets', 'auxiliary_tools'}


def validate_value(kind, value):
    if kind == 'recipe':
        return resolve_recipe(value)
    if kind == 'effect_presets':
        if not isinstance(value, list) or not 1 <= len(value) <= 32:
            raise ValueError('Save 1–32 effect presets')
        result = [validate_preset(item) for item in value]
        if len({item['id'] for item in result}) != len(result):
            raise ValueError('Effect preset IDs must be unique')
        return result
    if kind == 'auxiliary_tools':
        # Tool bundles preserve referenced preset IDs. Cross-reference validation
        # happens again when the tools enter a full resolved recipe.
        return validate_tool_set(value)
    raise ValueError('Unknown designer asset kind')


def asset_path(root, kind, identifier):
    if kind not in KINDS or not isinstance(identifier, str) or not ID.fullmatch(identifier):
        raise ValueError('Use a valid asset kind and a short filename-safe ID')
    directory = Path(root) / 'presets' / kind
    path = directory / (identifier + '.json')
    if (Path(root) / 'presets').is_symlink() or directory.is_symlink() or path.is_symlink():
        raise ValueError('Preset paths cannot be symbolic links')
    return path


def save(root, kind, identifier, value, guard=None):
    path = asset_path(root, kind, identifier)
    resolved = validate_value(kind, value)
    canonical = json.dumps(dict(kind=kind, value=resolved), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    record = dict(schema_version=1, kind=kind, id=identifier, created_at=utc_now(), value=resolved,
                  content_sha256=hashlib.sha256(canonical).hexdigest())
    raw = (json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
    if len(raw) > 2*1024**2:
        raise ValueError('A saved preset asset must be at most 2 MiB')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError('This asset ID already exists. Save the revision under a new ID.')
    created = False
    def write_record():
        nonlocal created
        with path.open('xb') as stream:
            created = True
            stream.write(raw)
    try:
        if guard:
            with guard.write(path.parent, len(raw)):
                write_record()
        else:
            write_record()
    except FileExistsError:
        raise ValueError('This asset ID already exists. Save the revision under a new ID.') from None
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise
    return deepcopy(record)


def load(root, kind, identifier):
    path = asset_path(root, kind, identifier)
    if not path.is_file(): raise FileNotFoundError('Designer asset not found')
    if path.stat().st_size > 2*1024**2: raise ValueError('Designer asset is too large')
    record = read_json(path)
    if not isinstance(record, dict) or record.get('schema_version') != 1 or record.get('kind') != kind or record.get('id') != identifier:
        raise ValueError('Unsupported or mismatched designer asset')
    value = validate_value(kind, record.get('value'))
    canonical = json.dumps(dict(kind=kind, value=value), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != record.get('content_sha256'):
        raise ValueError('Designer asset integrity check failed')
    return record


def catalog(root):
    rows = []
    for kind in sorted(KINDS):
        directory = Path(root) / 'presets' / kind
        if directory.is_symlink() or not directory.is_dir(): continue
        for path in sorted(directory.glob('*.json')):
            try:
                record = load(root, kind, path.stem)
                rows.append({key: record[key] for key in ('schema_version','kind','id','created_at','content_sha256')})
            except (ValueError, OSError): continue
    return rows
