"""Research calibration without model execution or imports of the GPU stack.

The schema-1 historical pilot deliberately remains in runtime.py. Schema 2
separates intervention fitting (train), readout fitting (probe), model selection
(selection), and final reporting (heldout). No output is an emotion instrument.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 2
SPLITS = ("train", "probe", "selection", "heldout")
CONCEPTS = ("pain", "joy")
SLICE_FIELDS = ("context", "lexical", "subject", "style", "difficulty")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return float(value)


def validate_config(config, layer_count=None):
    """Resolve the opt-in research preset; reject silently ignored options."""
    if not isinstance(config, dict):
        raise ValueError("calibration config must be an object")
    allowed = {"schema_version", "preset", "name", "layers", "downstream_layer", "poolings", "probe_methods", "ridge_alphas", "seed", "bootstrap_samples", "max_input_tokens", "doses", "corpus", "continuation_tokens", "validation_pairs_per_concept", "max_kl", "max_relative_delta"}
    if set(config) - allowed:
        raise ValueError(f"unknown research calibration options: {sorted(set(config) - allowed)}")
    if type(config.get("schema_version", 2)) is not int or config.get("schema_version", 2) != 2 or config.get("preset", "research") != "research":
        raise ValueError("research calibration requires schema_version 2 and preset research")
    name = config.get("name", "Research concept calibration")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100:
        raise ValueError("name must contain 1–100 characters")
    high = 4095 if layer_count is None else _integer(layer_count, "layer_count", 2, 4096) - 2
    defaults = [0] if layer_count is None else sorted({min(high, int(layer_count * f)) for f in (.35, .55, .7)})
    layers = config.get("layers", defaults)
    if not isinstance(layers, list) or not 1 <= len(layers) <= 12:
        raise ValueError("layers must contain 1–12 indices")
    layers = sorted(set(_integer(x, "layer", 0, high) for x in layers))
    downstream = config.get("downstream_layer", (layer_count - 1) if layer_count else max(layers) + 1)
    downstream = _integer(downstream, "downstream_layer", max(layers) + 1, 4096 if layer_count is None else layer_count - 1)
    poolings = config.get("poolings", ["final", "mean"])
    methods = config.get("probe_methods", ["mean", "ridge"])
    for values, accepted, name_ in ((poolings, {"final", "mean", "span"}, "poolings"), (methods, {"mean", "ridge"}, "probe_methods")):
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or v not in accepted for v in values) or len(set(values)) != len(values):
            raise ValueError(f"invalid {name_}")
    alphas = config.get("ridge_alphas", [1., 10.])
    if not isinstance(alphas, list) or not 1 <= len(alphas) <= 8:
        raise ValueError("ridge_alphas must contain 1–8 values")
    alphas = sorted(set(_number(x, "ridge alpha", 1e-6, 1e6) for x in alphas))
    doses = config.get("doses", [0., .25, .5, 1.])
    if not isinstance(doses, list) or not 1 <= len(doses) <= 12:
        raise ValueError("doses must contain 1–12 values")
    result = {"schema_version": 2, "preset": "research", "name": name.strip(), "layers": layers,
              "downstream_layer": downstream, "poolings": sorted(poolings), "probe_methods": sorted(methods),
              "ridge_alphas": alphas, "seed": _integer(config.get("seed", 1729), "seed", 0, 2**32-1),
              "bootstrap_samples": _integer(config.get("bootstrap_samples", 300), "bootstrap_samples", 20, 5000),
              "max_input_tokens": _integer(config.get("max_input_tokens", 512), "max_input_tokens", 16, 4096),
              "continuation_tokens": _integer(config.get("continuation_tokens", 64), "continuation_tokens", 8, 512),
              "validation_pairs_per_concept": _integer(config.get("validation_pairs_per_concept", 4), "validation_pairs_per_concept", 1, 1000),
              "doses": sorted(set(_number(x, "dose", 0., 4.) for x in doses) | {0.}),
              "max_kl": _number(config.get("max_kl", .5), "max_kl", 0., 100.),
              "max_relative_delta": _number(config.get("max_relative_delta", .3), "max_relative_delta", 0., 10.)}
    if "corpus" in config:
        result["corpus"] = validate_corpus(config["corpus"])
    return result


def validate_corpus(document):
    """Require paired examples and disjoint scenario AND template families.

    A pair is matched within one concept, split and context. The family bootstrap
    below keeps every paraphrase and both labels of each family together.
    """
    if not isinstance(document, dict) or set(document) - {"schema_version", "version", "provenance", "rows", "limitations"}:
        raise ValueError("research corpus must be a versioned object with known fields")
    if type(document.get("schema_version")) is not int or document.get("schema_version") != 2 or not isinstance(document.get("version"), str) or not document["version"].strip():
        raise ValueError("research corpus requires schema_version 2 and version")
    if not isinstance(document.get("provenance"), str) or not document["provenance"].strip():
        raise ValueError("corpus provenance is required")
    rows = document.get("rows")
    if not isinstance(rows, list) or not 16 <= len(rows) <= 20000:
        raise ValueError("corpus must contain 16–20000 rows")
    ids, texts, families, templates, pairs = set(), set(), {}, {}, {}
    counts = {(s, c, label): 0 for s in SPLITS for c in CONCEPTS for label in (0, 1)}
    required = {"id", "family", "template_family", "split", "concept", "label", "pair_id", "text", *SLICE_FIELDS}
    for row in rows:
        if not isinstance(row, dict) or set(row) - (required | {"span"}) or not required <= set(row):
            raise ValueError("corpus row has missing or unknown fields")
        for field in required - {"label"}:
            if not isinstance(row[field], str) or not row[field].strip() or len(row[field]) > (16000 if field == "text" else 200):
                raise ValueError(f"invalid corpus {field}")
        if type(row["label"]) is not int or (row["split"], row["concept"], row["label"]) not in counts:
            raise ValueError("invalid corpus split/concept/binary label")
        textkey = " ".join(row["text"].casefold().split())
        if row["id"] in ids or textkey in texts:
            raise ValueError("duplicate corpus ID or text")
        ids.add(row["id"]); texts.add(textkey)
        for field, lookup in (("family", families), ("template_family", templates)):
            prior = lookup.setdefault(row[field], row["split"])
            if prior != row["split"]:
                raise ValueError(f"cross-split {field} leakage")
        pairs.setdefault(row["pair_id"], []).append(row)
        counts[row["split"], row["concept"], row["label"]] += 1
        if "span" in row:
            span = row["span"]
            if not isinstance(span, list) or len(span) != 2 or any(type(i) is not int for i in span) or not 0 <= span[0] < span[1] <= len(row["text"]):
                raise ValueError("span must be a nonempty [start,end] character range")
    for pair in pairs.values():
        if len(pair) != 2 or {r["label"] for r in pair} != {0, 1}:
            raise ValueError("every pair needs exactly one positive and negative")
        shared = ("split", "concept", "family", "template_family", *SLICE_FIELDS)
        if any(pair[0][key] != pair[1][key] for key in shared):
            raise ValueError("paired labels must share all matching metadata")
    if min(counts.values()) < 2:
        raise ValueError("every split needs two paired cases for each concept")
    return deepcopy(document)


def load_research_corpus(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[1] / "corpora" / "research-v2.json"
    return validate_corpus(json.loads(path.read_text(encoding="utf-8")))


def corpus_summary(document):
    data = validate_corpus(document)
    return {"version": data["version"], "sha256": digest(data), "splits": {
        split: {"sha256": digest([r for r in data["rows"] if r["split"] == split]),
                "rows": sum(r["split"] == split for r in data["rows"]),
                "families": len({r["family"] for r in data["rows"] if r["split"] == split}),
                "pairs": {c: sum(r["split"] == split and r["concept"] == c and r["label"] == 1 for r in data["rows"]) for c in CONCEPTS}}
        for split in SPLITS}}


def pool_hidden(hidden, attention_mask, policy="final", span_mask=None):
    """Pool B×T×D numpy OR torch tensors; ignore left/right padding exactly.

    For span pooling the caller maps the corpus character span to token offsets,
    supplies a B×T mask, and records that policy in the extraction manifest.
    Empty examples/spans are errors, never fallbacks to an unrelated token.
    """
    torch_backend = type(hidden).__module__.startswith("torch")
    if torch_backend:
        import torch
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.as_tensor(attention_mask, device=hidden.device)
        if hidden.ndim != 3 or tuple(attention_mask.shape) != tuple(hidden.shape[:2]):
            raise ValueError("hidden/mask shapes must be B×T×D and B×T")
        mask = attention_mask.bool()
        if not bool(mask.any(dim=1).all()):
            raise ValueError("cannot pool an empty example")
        if policy == "span":
            if span_mask is None or tuple(span_mask.shape) != tuple(mask.shape):
                raise ValueError("span pooling requires a matching span mask")
            mask = mask & torch.as_tensor(span_mask, device=hidden.device).bool()
            if not bool(mask.any(dim=1).all()):
                raise ValueError("cannot pool an empty span")
        if policy in {"mean", "span"}:
            values = hidden.float() if hidden.dtype in (torch.float16, torch.bfloat16) else hidden
            return torch.where(mask[..., None], values, torch.zeros_like(values)).sum(1) / mask.sum(1, keepdim=True)
        if policy == "final":
            positions = torch.arange(mask.shape[1], device=hidden.device).expand_as(mask).masked_fill(~mask, -1).max(1).values
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
    else:
        hidden, mask = np.asarray(hidden), np.asarray(attention_mask, dtype=bool)
        if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
            raise ValueError("hidden/mask shapes must be B×T×D and B×T")
        if not mask.any(1).all():
            raise ValueError("cannot pool an empty example")
        if policy == "span":
            if span_mask is None or np.shape(span_mask) != mask.shape:
                raise ValueError("span pooling requires a matching span mask")
            mask = mask & np.asarray(span_mask, dtype=bool)
            if not mask.any(1).all():
                raise ValueError("cannot pool an empty span")
        if policy in {"mean", "span"}:
            values = hidden.astype(np.float32) if hidden.dtype == np.float16 else hidden
            return np.where(mask[..., None], values, 0).sum(1) / mask.sum(1, keepdims=True)
        if policy == "final":
            positions = np.where(mask, np.arange(mask.shape[1]), -1).max(1)
            return hidden[np.arange(hidden.shape[0]), positions]
    raise ValueError("pooling must be final, mean or span")


def _matrix(value, length=None):
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or (length is not None and len(x) != length) or not x.size or not np.isfinite(x).all():
        raise ValueError("activations must be finite N×D matrices matching corpus rows")
    return x


def unit(vector, label="direction"):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or norm < 1e-10:
        raise ValueError(f"degenerate {label}")
    return vector / norm


def fit_readout(x, labels, method="ridge", alpha=1.):
    """Affine score fit only on independent probe data; threshold fixed at 0.

    Ridge solves the dual n×n system when d>n. Centering and standardization
    never see selection/heldout activations. Coefficients are exported in raw
    activation coordinates, so inference need not re-standardize separately.
    """
    x = _matrix(x)
    labels = np.asarray(labels)
    if labels.shape != (len(x),) or not set(labels.tolist()) <= {0, 1} or len(set(labels.tolist())) != 2:
        raise ValueError("readout requires both binary labels")
    center, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-8, scale, 1.)
    z = (x - center) / scale
    y = labels.astype(float) * 2 - 1
    if method == "ridge":
        alpha = _number(alpha, "ridge alpha", 1e-6, 1e6)
        target = y - y.mean()
        if z.shape[1] > len(z):
            standardized = z.T @ np.linalg.solve(z @ z.T + alpha * np.eye(len(z)), target)
        else:
            standardized = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ target)
        intercept = float(y.mean())
    elif method == "mean":
        standardized = z[labels == 1].mean(0) - z[labels == 0].mean(0)
        # A constant activation readout is valid null evidence, not a crash.
        norm = np.linalg.norm(standardized)
        if norm > 1e-10:
            standardized /= norm
        intercept = -float((z[labels == 1].mean(0) + z[labels == 0].mean(0)) @ standardized / 2)
    else:
        raise ValueError("readout method must be mean or ridge")
    weight = standardized / scale
    bias = intercept - float(center @ weight)
    return {"weight": weight, "bias": bias, "center": center, "scale": scale,
            "method": method, "alpha": alpha if method == "ridge" else None}


def readout_score(readout, x):
    return _matrix(x) @ np.asarray(readout["weight"]) + float(readout["bias"])


def binary_metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1 or not np.isfinite(scores).all() or not set(labels.tolist()) <= {0, 1}:
        raise ValueError("invalid binary observations")
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not len(positive) or not len(negative):
        return {"n": len(labels), "positive": len(positive), "negative": len(negative), "auc": None, "balanced_accuracy": None}
    auc = float(((positive[:, None] > negative).mean() + .5 * (positive[:, None] == negative).mean()))
    return {"n": len(labels), "positive": len(positive), "negative": len(negative), "auc": auc,
            "balanced_accuracy": float(((positive > 0).mean() + (negative <= 0).mean()) / 2),
            "mean_positive_score": float(positive.mean()), "mean_negative_score": float(negative.mean())}


def family_interval(rows, scores, seed=1729, samples=300, checkpoint=lambda: None):
    """Percentile family-cluster bootstrap, not an independence/power claim."""
    labels = np.asarray([r["label"] for r in rows])
    families = sorted({r["family"] for r in rows})
    if len(families) < 2:
        return {"unit": "scenario_family", "families": len(families), "auc_95": None, "balanced_accuracy_95": None}
    groups = [np.asarray([i for i, r in enumerate(rows) if r["family"] == f]) for f in families]
    rng = np.random.default_rng(seed)
    values = []
    for iteration in range(samples):
        if iteration % 32 == 0:
            checkpoint()
        ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        result = binary_metrics(labels[ix], np.asarray(scores)[ix])
        if result["auc"] is not None:
            values.append([result["auc"], result["balanced_accuracy"]])
    interval = np.quantile(values, [.025, .975], axis=0).T.tolist() if values else [None, None]
    return {"unit": "scenario_family", "families": len(families), "resamples": samples, "seed": seed,
            "auc_95": interval[0], "balanced_accuracy_95": interval[1]}


def report_readout(rows, scores, seed=1729, samples=300, checkpoint=lambda: None):
    labels = [r["label"] for r in rows]
    result = binary_metrics(labels, scores)
    result["uncertainty"] = family_interval(rows, scores, seed, samples, checkpoint)
    result["slices"] = {field: {value: binary_metrics([r["label"] for r in rows if r[field] == value],
                           [s for r, s in zip(rows, scores) if r[field] == value])
                        for value in sorted({r[field] for r in rows})} for field in SLICE_FIELDS}
    return result


def _correlation(a, b):
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def fit_calibration(activations, corpus, config, checkpoint=lambda: None, emit=lambda event: None):
    """Fit and select without ever reading heldout rows' activations/labels.

    Input matrices are keyed by '<block>:<pooling>'. Caller extracts all requested
    candidates and downstream block once. Returned arrays are npz-compatible;
    metadata is JSON-compatible. Call evaluate_fitted separately after locking.
    """
    corpus = validate_corpus(corpus)
    config = validate_config(config)
    if "corpus" in config and digest(config["corpus"]) != digest(corpus):
        raise ValueError("config corpus differs from the supplied extraction corpus")
    rows = corpus["rows"]
    required = [f"{layer}:{pooling}" for layer in config["layers"] + [config["downstream_layer"]] for pooling in config["poolings"]]
    if set(activations) != set(required):
        raise ValueError("activation sites must exactly match declared layers/poolings")
    # Validation checks shape/finiteness only; heldout values never enter fitting.
    matrices = {key: _matrix(value, len(rows)) for key, value in activations.items()}
    if len({x.shape[1] for x in matrices.values()}) != 1:
        raise ValueError("candidate activation dimensions differ")
    candidates, readouts = [], {}
    for key in sorted(required, key=lambda k: (int(k.split(":")[0]), k.split(":")[1])):
        for method in config["probe_methods"]:
            for alpha in (config["ridge_alphas"] if method == "ridge" else [None]):
                checkpoint()
                candidate_id = f"{key}:{method}:{alpha}"
                readouts[candidate_id] = {}
                scores = []
                reports = {}
                for concept in CONCEPTS:
                    fit_ix = [i for i, r in enumerate(rows) if r["split"] == "probe" and r["concept"] == concept]
                    select_ix = [i for i, r in enumerate(rows) if r["split"] == "selection" and r["concept"] == concept]
                    readout = fit_readout(matrices[key][fit_ix], [rows[i]["label"] for i in fit_ix], method, alpha or 1.)
                    readouts[candidate_id][concept] = readout
                    report = binary_metrics([rows[i]["label"] for i in select_ix], readout_score(readout, matrices[key][select_ix]))
                    reports[concept] = report
                    scores.append(report["auc"])
                candidates.append({"id": candidate_id, "site": key, "layer": int(key.split(":")[0]), "pooling": key.split(":")[1], "method": method, "alpha": alpha,
                                   "score": sum(scores)/len(scores), "selection": reports})
                emit({"type": "calibration_progress", "stage": "research_fit_candidate", "completed": len(candidates),
                      "total": len(required) * sum(len(config["ridge_alphas"]) if m == "ridge" else 1 for m in config["probe_methods"]),
                      "candidate_id": candidate_id})
    # Deterministic order: highest selection AUC, earliest block, alphabetical
    # pooling/method, smallest alpha. No heldout data participate in tie-breaking.
    rank = lambda c: (-c["score"], c["layer"], c["pooling"], c["method"], c["alpha"] or 0.)
    selected = sorted([c for c in candidates if c["layer"] in config["layers"]], key=rank)[0]
    downstream = sorted([c for c in candidates if c["layer"] == config["downstream_layer"]], key=rank)[0]
    vectors, interventions = {}, {}
    x = matrices[selected["site"]]
    neutral = x[[i for i, r in enumerate(rows) if r["split"] == "train" and r["label"] == 0]].mean(0)
    for concept in CONCEPTS:
        positive = x[[i for i, r in enumerate(rows) if r["split"] == "train" and r["concept"] == concept and r["label"] == 1]]
        negative = x[[i for i, r in enumerate(rows) if r["split"] == "train" and r["concept"] == concept and r["label"] == 0]]
        raw = positive.mean(0) - negative.mean(0)
        vectors[f"{concept}_contrast"] = raw
        interventions[concept] = unit(raw, f"{concept} intervention")
    vectors["pain"] = interventions["pain"]
    vectors["joy_raw"] = interventions["joy"]
    orthogonal = interventions["joy"] - (interventions["joy"] @ interventions["pain"]) * interventions["pain"]
    orthogonal_available = np.linalg.norm(orthogonal) > 1e-8
    # Save a zero orthogonal vector with an explicit unsupported flag if concepts
    # collapse; never invent an arbitrary substitute direction.
    vectors["joy"] = unit(orthogonal) if orthogonal_available else np.zeros_like(orthogonal)
    negative_train = x[[i for i, r in enumerate(rows) if r["split"] == "train" and r["label"] == 0]]
    scale = float(np.linalg.norm(negative_train, axis=1).mean()) / 4
    if scale < 1e-10:
        raise ValueError("degenerate training reference scale")
    vectors.update(neutral=neutral, scale=np.asarray(scale), random=unit(np.random.default_rng(config["seed"] + 1).normal(size=x.shape[1])))
    for prefix, candidate in (("probe", selected), ("downstream", downstream)):
        for concept, readout in readouts[candidate["id"]].items():
            vectors[f"{prefix}_{concept}_weight"] = readout["weight"]
            vectors[f"{prefix}_{concept}_bias"] = np.asarray(readout["bias"])
            vectors[f"{prefix}_{concept}_fit_center"] = readout["center"]
            vectors[f"{prefix}_{concept}_fit_scale"] = readout["scale"]
        # Legacy projection keys remain available but are clearly separate from
        # the affine independent readouts used by schema-2 reports.
        probe_ix = [i for i, r in enumerate(rows) if r["split"] == "probe" and r["label"] == 0]
        probe_x = matrices[candidate["site"]][probe_ix]
        vectors[f"{prefix}_center"] = probe_x.mean(0)
        vectors[f"{prefix}_scale"] = np.asarray(max(float(np.linalg.norm(probe_x, axis=1).mean()) / 4, 1e-8))
        for concept in CONCEPTS:
            weight = readouts[candidate["id"]][concept]["weight"]
            vectors[f"{prefix}_{concept}"] = unit(weight) if np.linalg.norm(weight) > 1e-10 else np.zeros_like(weight)
    # Independent seeded label shuffle, preserving pairing and all fit boundaries.
    # Each pair's orientation is flipped independently; chance isn't assumed.
    shuffle = {}
    for j, concept in enumerate(CONCEPTS):
        rng = np.random.default_rng(config["seed"] + 100 + j)
        ix = [i for i, r in enumerate(rows) if r["split"] == "probe" and r["concept"] == concept]
        flips = {p: int(rng.integers(2)) for p in sorted({rows[i]["pair_id"] for i in ix})}
        labels = [rows[i]["label"] ^ flips[rows[i]["pair_id"]] for i in ix]
        sh = fit_readout(matrices[selected["site"]][ix], labels, selected["method"], selected["alpha"] or 1.)
        shuffle[concept] = {"weight": sh["weight"], "bias": sh["bias"]}
        vectors[f"shuffled_probe_{concept}_weight"] = sh["weight"]
        vectors[f"shuffled_probe_{concept}_bias"] = np.asarray(sh["bias"])
        train_ix = [i for i, r in enumerate(rows) if r["split"] == "train" and r["concept"] == concept]
        flips = {p: int(rng.integers(2)) for p in sorted({rows[i]["pair_id"] for i in train_ix})}
        labels = np.asarray([rows[i]["label"] ^ flips[rows[i]["pair_id"]] for i in train_ix])
        z = x[train_ix]
        raw = z[labels == 1].mean(0) - z[labels == 0].mean(0)
        vectors[f"shuffled_{concept}_contrast"] = raw
        direction = unit(raw) if np.linalg.norm(raw) > 1e-10 else np.zeros_like(raw)
        vectors[f"shuffled_{concept}"] = direction
        midpoint = (z[labels == 1].mean(0) + z[labels == 0].mean(0)) / 2
        vectors[f"shuffled_{concept}_bias"] = np.asarray(-float(midpoint @ direction))
    public_config = {k: v for k, v in config.items() if k != "corpus"}
    metadata = {"schema_version": 2, "kind": "research_split_calibration", "config": public_config,
                "layer": selected["layer"], "downstream_layer": downstream["layer"], "pooling": selected["pooling"],
                "selected": selected, "downstream_selected": downstream, "candidate_layers": candidates,
                "corpus": corpus_summary(corpus), "orthogonal_joy_available": bool(orthogonal_available),
                "direction_cosine_before_orthogonalization": float(interventions["joy"] @ interventions["pain"]),
                "selection_rule": "Maximum mean selection AUC; ties: earliest block, alphabetical pooling/method, smallest ridge alpha",
                "status": "unvalidated", "validated_tests": [],
                "interpretation": "Concept-associated text readouts and interventions, not measurements of subjective emotion",
                "fit_split": "train", "readout_split": "probe", "selection_split": "selection", "evaluation_split": "heldout",
                "shuffle_seeds": {c: config["seed"] + 100 + i for i, c in enumerate(CONCEPTS)}}
    metadata["compatibility_sha256"] = digest({"config": public_config, "corpus": metadata["corpus"], "selected": selected, "downstream": downstream})
    return {"metadata": metadata, "vectors": {k: np.asarray(v, dtype=np.float32) for k, v in vectors.items()}}


def evaluate_fitted(fitted, activations, corpus, split="heldout", checkpoint=lambda: None):
    """Evaluate locked readouts; never mutate the package or select a dose/site."""
    corpus = validate_corpus(corpus)
    if split not in {"selection", "heldout"}:
        raise ValueError("evaluation split must be selection or heldout")
    metadata, vectors, rows = fitted["metadata"], fitted["vectors"], corpus["rows"]
    if metadata["corpus"]["sha256"] != digest(corpus):
        raise ValueError("evaluation corpus differs from the locked calibration corpus")
    result = {"split": split, "corpus_sha256": digest(corpus), "sites": {}}
    for prefix, candidate in (("probe", metadata["selected"]), ("downstream", metadata["downstream_selected"])):
        x = _matrix(activations[candidate["site"]], len(rows))
        scores = {c: x @ vectors[f"{prefix}_{c}_weight"] + float(vectors[f"{prefix}_{c}_bias"]) for c in CONCEPTS}
        site = {}
        for j, concept in enumerate(CONCEPTS):
            ix = [i for i, r in enumerate(rows) if r["split"] == split and r["concept"] == concept]
            site[concept] = report_readout([rows[i] for i in ix], scores[concept][ix], metadata["config"]["seed"] + j, metadata["config"]["bootstrap_samples"], checkpoint)
        all_ix = [i for i, r in enumerate(rows) if r["split"] == split]
        site["cross_concept_score_correlation"] = _correlation(scores["pain"][all_ix], scores["joy"][all_ix])
        site["cross_concept_discrimination"] = {c: {other: binary_metrics([rows[i]["label"] for i in all_ix if rows[i]["concept"] == other], scores[c][[i for i in all_ix if rows[i]["concept"] == other]]) for other in CONCEPTS if other != c} for c in CONCEPTS}
        result["sites"][prefix] = site
    result["shuffled_controls"] = {}
    result["shuffled_direction_controls"] = {}
    x = _matrix(activations[metadata["selected"]["site"]], len(rows))
    for j, concept in enumerate(CONCEPTS):
        ix = [i for i, r in enumerate(rows) if r["split"] == split and r["concept"] == concept]
        scores = x[ix] @ vectors[f"shuffled_probe_{concept}_weight"] + float(vectors[f"shuffled_probe_{concept}_bias"])
        result["shuffled_controls"][concept] = report_readout([rows[i] for i in ix], scores, metadata["config"]["seed"] + 10 + j, metadata["config"]["bootstrap_samples"], checkpoint)
        direction_scores = x[ix] @ vectors[f"shuffled_{concept}"] + float(vectors[f"shuffled_{concept}_bias"])
        result["shuffled_direction_controls"][concept] = report_readout([rows[i] for i in ix], direction_scores, metadata["config"]["seed"] + 20 + j, metadata["config"]["bootstrap_samples"], checkpoint)
    return result
