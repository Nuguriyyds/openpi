"""Create interactive validation/test curves for a temporal completion report.

The evaluator report intentionally stores aggregate metrics only.  This small
companion reuses the report's exact manifest, feature cache, checkpoint, and
config to recover row-level sigmoid scores, then writes a self-contained HTML
viewer.  Each curve is one subtask episode, with the samples ordered by their
local current frame.

Example::

    uv run scripts/visualize_temporal_completion.py \
        --report /mnt/data/models/wyt/evaluations/temporal_completion_reports/history_seed42.json \
        --output /mnt/data/models/wyt/evaluations/temporal_completion_reports/history_seed42_curves.html

The first run also writes a small ``.predictions.npz`` sidecar next to the
HTML.  Later runs reuse it when it belongs to the same report/checkpoint;
remove that sidecar to force fresh inference.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from openpi.training import temporal_completion_data as temporal_data
from openpi.training import temporal_completion_features as temporal_features
from openpi.training import temporal_completion_metrics as temporal_metrics
from scripts import evaluate_temporal_completion as evaluator

SPLIT_CHOICES = ("val", "test")
PREDICTION_SCHEMA_VERSION = 1


def _read_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"evaluation report not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("evaluation report must be a JSON object")
    if not isinstance(value.get("bindings"), dict) or not isinstance(value.get("oracle_prompt"), dict):
        raise ValueError("evaluation report lacks bindings/oracle_prompt")
    if not isinstance(value.get("threshold_selection"), dict):
        raise ValueError("evaluation report lacks threshold_selection")
    for split in SPLIT_CHOICES:
        if not isinstance(value["oracle_prompt"].get(split), dict):
            raise ValueError(f"evaluation report lacks oracle_prompt.{split}")
    return value


def _selected_episodes(keys: Sequence[str], max_episodes: int) -> set[str]:
    if max_episodes < 0:
        raise ValueError("--max-episodes must be non-negative (0 means all)")
    if max_episodes == 0 or len(keys) <= max_episodes:
        return set(keys)
    # Evenly spread the preview over the split instead of selecting only the
    # first few episode IDs (which are usually dominated by task 0).
    positions = np.linspace(0, len(keys) - 1, num=max_episodes, dtype=np.int64)
    return {keys[int(position)] for position in positions}


def _prediction_metadata(report_path: Path, report: Mapping[str, Any]) -> dict[str, str | int]:
    bindings = report["bindings"]
    if not isinstance(bindings, Mapping):
        raise ValueError("report bindings must be an object")
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "report_path": str(report_path.resolve()),
        "config_name": str(bindings["config_name"]),
        "checkpoint_path": str(bindings["checkpoint_path"]),
        "feature_cache_path": str(bindings["feature_cache_path"]),
    }


def _load_prediction_sidecar(
    path: Path,
    *,
    metadata: Mapping[str, str | int],
) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as arrays:
            if "metadata_json" not in arrays.files:
                return None
            stored = json.loads(str(np.asarray(arrays["metadata_json"]).item()))
            if stored != dict(metadata):
                return None
            required = {
                "metadata_json",
                "split",
                "episode_key",
                "trajectory_id",
                "subtask_episode_id",
                "task_index",
                "current_frame",
                "boundary_frame",
                "label",
                "sample_kind",
                "score",
            }
            if set(arrays.files) != required:
                return None
            return {name: np.asarray(arrays[name]) for name in required if name != "metadata_json"}
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _rows_to_payload(
    rows: Sequence[temporal_data.TemporalSampleRow],
    scores: np.ndarray,
    *,
    split: str,
) -> dict[str, np.ndarray]:
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(rows) != score_array.size:
        raise ValueError(f"row/score length mismatch for {split}: {len(rows)} vs {score_array.size}")
    if not np.isfinite(score_array).all():
        raise ValueError(f"non-finite scores in {split} predictions")
    episode_keys = np.asarray(
        [f"{row.trajectory_id}/subtask-{row.subtask_episode_id}/task-{row.task_index}" for row in rows]
    )
    return {
        "split": np.asarray([split] * len(rows)),
        "episode_key": episode_keys,
        "trajectory_id": np.asarray([row.trajectory_id for row in rows]),
        "subtask_episode_id": np.asarray([row.subtask_episode_id for row in rows], dtype=np.int64),
        "task_index": np.asarray([row.task_index for row in rows], dtype=np.int8),
        "current_frame": np.asarray([row.source_frame_indices[-1] for row in rows], dtype=np.int64),
        "boundary_frame": np.asarray([row.boundary_tick for row in rows], dtype=np.int64),
        "label": np.asarray([row.label for row in rows], dtype=np.int8),
        "sample_kind": np.asarray([row.sample_kind for row in rows]),
        "score": score_array,
    }


def _merge_payloads(payloads: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not payloads:
        raise ValueError("no prediction payloads were produced")
    names = tuple(payloads[0])
    if any(tuple(payload) != names for payload in payloads[1:]):
        raise ValueError("prediction payload schemas differ")
    return {name: np.concatenate([payload[name] for payload in payloads]) for name in names}


def _save_prediction_sidecar(path: Path, metadata: Mapping[str, str | int], payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata_json=np.asarray(json.dumps(dict(metadata), sort_keys=True)), **payload)


def _compute_predictions(
    report: Mapping[str, Any],
    *,
    splits: Sequence[str],
    batch_size: int,
) -> dict[str, np.ndarray]:
    bindings = report["bindings"]
    config_name = str(bindings["config_name"])

    # Keep the same loading and scoring path as the evaluator so the plotted
    # probabilities are numerically identical to the report metrics.
    from openpi.training import config as training_config  # noqa: PLC0415

    config = training_config.get_config(config_name)
    manifest_path = Path(str(bindings["manifest_path"])).resolve()
    cache_path = Path(str(bindings["feature_cache_path"])).resolve()
    checkpoint_path = Path(str(bindings["checkpoint_path"])).resolve()
    manifest = evaluator._load_sealed_manifest(manifest_path)  # noqa: SLF001
    cache = temporal_features.load_temporal_feature_cache(
        cache_path,
        manifest=manifest,
        expected_checkpoint_path=config.completion.temporal_source_checkpoint_path,
        expected_model_config_name=config.completion.temporal_source_model_config_name,
    )
    model = evaluator._load_temporal_model(  # noqa: SLF001
        config, checkpoint_path, feature_dim=cache.metadata.feature_dim
    )
    all_payloads: list[dict[str, np.ndarray]] = []
    for split in splits:
        rows, history = evaluator._split_cache(cache, split)  # noqa: SLF001
        logits = evaluator._predict_temporal_logits(  # noqa: SLF001
            model,
            history,
            batch_size=batch_size,
            input_mode=config.completion.temporal_input_mode,
        )
        scores = temporal_metrics.stable_sigmoid(logits)
        all_payloads.append(_rows_to_payload(rows, scores, split=split))
    return _merge_payloads(all_payloads)


def _group_payload(
    payload: Mapping[str, np.ndarray], *, max_episodes: int, task_index: int | None
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(payload["episode_key"].astype(str).tolist()):
        if task_index is not None and int(payload["task_index"][index]) != task_index:
            continue
        groups[key].append(index)
    keys = sorted(groups)
    selected = _selected_episodes(keys, max_episodes)
    episodes: list[dict[str, Any]] = []
    for key in keys:
        if key not in selected:
            continue
        indices = sorted(groups[key], key=lambda index: int(payload["current_frame"][index]))
        first = indices[0]
        episodes.append(
            {
                "key": key,
                "trajectory_id": str(payload["trajectory_id"][first]),
                "subtask_episode_id": int(payload["subtask_episode_id"][first]),
                "task_index": int(payload["task_index"][first]),
                "boundary_frame": int(payload["boundary_frame"][first]),
                "frames": [int(payload["current_frame"][index]) for index in indices],
                "labels": [int(payload["label"][index]) for index in indices],
                "scores": [float(payload["score"][index]) for index in indices],
                "kinds": [str(payload["sample_kind"][index]) for index in indices],
            }
        )
    return episodes


def _metric_cards(report: Mapping[str, Any]) -> dict[str, dict[str, float | None]]:
    cards: dict[str, dict[str, float | None]] = {}
    oracle = report["oracle_prompt"]
    for split in SPLIT_CHOICES:
        overall = oracle[split]["overall"]
        keys = (
            "natural/auprc",
            "natural/roc_auc",
            "hard_local/auprc",
            "paired_hard/ordering_accuracy",
            "paired_hard/margin_median",
        )
        cards[split] = {key: overall.get(key) for key in keys}
    return cards


def _html_document(
    report: Mapping[str, Any],
    payload: Mapping[str, np.ndarray],
    *,
    max_episodes: int,
    task_index: int | None,
) -> str:
    threshold = float(report["threshold_selection"]["threshold"])
    if not 0.0 <= threshold <= float(np.nextafter(1.0, np.inf)):
        raise ValueError(f"invalid report threshold: {threshold}")
    splits: dict[str, list[dict[str, Any]]] = {}
    for split in SPLIT_CHOICES:
        mask = payload["split"].astype(str) == split
        split_payload = {name: values[mask] for name, values in payload.items()}
        splits[split] = _group_payload(split_payload, max_episodes=max_episodes, task_index=task_index)
    data = {
        "title": str(report["bindings"]["config_name"]),
        "threshold": threshold,
        "splits": splits,
        "cards": _metric_cards(report),
        "report_path": str(report["bindings"].get("checkpoint_path", "")),
    }
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    title = html.escape(str(report["bindings"]["config_name"]))
    template = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Temporal completion curves</title>
<style>
:root { color-scheme: dark; --bg:#0f172a; --panel:#111827; --grid:#334155; --text:#dbeafe;
        --target:#61e6b5; --score:#ffab45; --threshold:#b7a2ff; }
body { margin:0; padding:24px; background:var(--bg); color:var(--text); font:14px/1.4 system-ui,sans-serif; }
h1 { margin:0 0 4px; font-size:20px; } .sub { color:#94a3b8; margin-bottom:18px; }
.controls { display:flex; flex-wrap:wrap; gap:10px; align-items:center; background:var(--panel); padding:12px;
            border-radius:10px; margin-bottom:12px; } label { color:#94a3b8; }
select { background:#1e293b; color:var(--text); border:1px solid #475569; border-radius:5px; padding:6px; }
.cards { display:flex; flex-wrap:wrap; gap:8px; margin:0 0 12px; }
.card { background:var(--panel); border-radius:8px; padding:8px 12px; min-width:125px; }
.card span { display:block; color:#94a3b8; font-size:11px; } .card b { font-size:17px; }
.panel { background:var(--panel); border-radius:10px; padding:14px; }
#episodeInfo { color:#94a3b8; margin-bottom:8px; }
svg { width:100%; height:auto; min-height:360px; display:block; }
.legend { display:flex; gap:18px; color:#cbd5e1; margin-top:8px; } .legend i { width:28px; height:3px; display:inline-block; vertical-align:middle; margin-right:5px; }
.target { background:var(--target); } .score { background:var(--score); } .threshold { background:var(--threshold); }
.empty { color:#fca5a5; padding:40px; text-align:center; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<div class="sub">Temporal completion scores by subtask episode · checkpoint __CHECKPOINT__</div>
<div class="controls">
  <label>Split <select id="split"></select></label>
  <label>Task <select id="task"><option value="all">all</option><option value="0">task 0</option><option value="1">task 1</option><option value="2">task 2</option><option value="3">task 3</option></select></label>
  <label>Episode <select id="episode"></select></label>
</div>
<div id="cards" class="cards"></div>
<div class="panel">
  <div id="episodeInfo"></div>
  <div id="chart"></div>
  <div class="legend"><span><i class="target"></i>Target (0/1)</span><span><i class="score"></i>Predicted score (sigmoid)</span><span><i class="threshold"></i>Validation threshold</span></div>
</div>
<script id="payload" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const splitEl = document.getElementById('split');
const taskEl = document.getElementById('task');
const episodeEl = document.getElementById('episode');
const cardsEl = document.getElementById('cards');
const infoEl = document.getElementById('episodeInfo');
const chartEl = document.getElementById('chart');
for (const split of ['val','test']) {
  const option = document.createElement('option'); option.value = split; option.textContent = split;
  splitEl.appendChild(option);
}
function selectedEpisodes() {
  const task = taskEl.value;
  return (DATA.splits[splitEl.value] || []).filter(item => task === 'all' || String(item.task_index) === task);
}
function fmt(value) { return value == null ? '—' : Number(value).toFixed(4); }
function updateCards() {
  const cards = DATA.cards[splitEl.value] || {};
  const labels = {'natural/auprc':'Natural AUPRC','natural/roc_auc':'ROC-AUC','hard_local/auprc':'Hard AUPRC','paired/ordering_accuracy':'Pair order','paired/hard_margin_median':'Hard margin median'};
  cardsEl.innerHTML = Object.entries(labels).map(([key,label]) => `<div class="card"><span>${label}</span><b>${fmt(cards[key])}</b></div>`).join('');
}
function updateEpisodes() {
  const episodes = selectedEpisodes(); episodeEl.innerHTML = '';
  episodes.forEach((item, index) => { const option = document.createElement('option'); option.value = String(index); option.textContent = `${item.key} (E=${item.boundary_frame})`; episodeEl.appendChild(option); });
  updateCards(); draw();
}
function linePoints(values, x, y) { return values.map((value,index) => `${x(index)},${y(value)}`).join(' '); }
function draw() {
  const item = selectedEpisodes()[Number(episodeEl.value) || 0];
  if (!item) { infoEl.textContent = 'No episodes in this selection.'; chartEl.innerHTML = '<div class="empty">No plotted rows.</div>'; return; }
  infoEl.textContent = `task ${item.task_index} · subtask episode ${item.subtask_episode_id} · boundary E=${item.boundary_frame} · ${item.frames.length} sampled rows`;
  const W=1100,H=480,L=64,R=22,T=20,B=52, iw=W-L-R, ih=H-T-B;
  const minX=Math.min(...item.frames), maxX=Math.max(...item.frames); const span=Math.max(1,maxX-minX);
  const x=i => L+(item.frames[i]-minX)/span*iw; const y=v => T+(1-Math.max(0,Math.min(1,v)))*ih;
  const grid=[0,0.25,0.5,0.75,1];
  const gridSvg=grid.map(v=>`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${L-10}" y="${y(v)+4}" text-anchor="end" fill="#94a3b8">${v.toFixed(2)}</text>`).join('');
  const scorePts=linePoints(item.scores,x,y);
  const targetPts=linePoints(item.labels,x,y);
  const markerSvg=item.frames.map((frame,i)=>`<circle cx="${x(i)}" cy="${y(item.scores[i])}" r="3.5" fill="var(--score)"/><circle cx="${x(i)}" cy="${y(item.labels[i])}" r="3" fill="var(--target)"/>`).join('');
  const thresholdY=y(DATA.threshold);
  chartEl.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="completion score curve"><rect x="0" y="0" width="${W}" height="${H}" fill="var(--panel)"/><g>${gridSvg}</g><line x1="${L}" x2="${L}" y1="${T}" y2="${H-B}" stroke="#94a3b8"/><line x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}" stroke="#94a3b8"/><line x1="${L}" x2="${W-R}" y1="${thresholdY}" y2="${thresholdY}" stroke="var(--threshold)" stroke-dasharray="7 6"/><polyline points="${targetPts}" fill="none" stroke="var(--target)" stroke-width="3"/><polyline points="${scorePts}" fill="none" stroke="var(--score)" stroke-width="2.5"/>${markerSvg}<text x="${L}" y="${H-18}" fill="#94a3b8">frame ${minX}</text><text x="${W-R}" y="${H-18}" text-anchor="end" fill="#94a3b8">frame ${maxX}</text><text x="${W-R}" y="${thresholdY-7}" text-anchor="end" fill="var(--threshold)">threshold ${DATA.threshold.toFixed(4)}</text></svg>`;
}
splitEl.addEventListener('change', updateEpisodes); taskEl.addEventListener('change', updateEpisodes); episodeEl.addEventListener('change', draw);
updateEpisodes();
</script>
</body>
</html>
"""
    return (
        template.replace("__TITLE__", title)
        .replace("__CHECKPOINT__", html.escape(str(report["bindings"].get("checkpoint_step", "?"))))
        .replace("__DATA__", encoded)
    )


def visualize(args: argparse.Namespace) -> Path:
    report_path = args.report.resolve()
    report = _read_report(report_path)
    splits = tuple(SPLIT_CHOICES if args.split == "both" else (args.split,))
    output = args.output.resolve()
    prediction_path = (
        args.predictions_cache.resolve()
        if args.predictions_cache is not None
        else output.with_suffix(".predictions.npz")
    )
    metadata = _prediction_metadata(report_path, report)
    payload = _load_prediction_sidecar(prediction_path, metadata=metadata)
    if payload is None:
        batch_size = int(args.batch_size)
        if batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        payload = _compute_predictions(report, splits=splits, batch_size=batch_size)
        _save_prediction_sidecar(prediction_path, metadata, payload)
        print(f"Saved prediction sidecar: {prediction_path}")
    else:
        available = set(payload["split"].astype(str).tolist())
        if not set(splits).issubset(available):
            payload = _compute_predictions(report, splits=splits, batch_size=int(args.batch_size))
            _save_prediction_sidecar(prediction_path, metadata, payload)
        print(f"Reused prediction sidecar: {prediction_path}")

    document = _html_document(report, payload, max_episodes=args.max_episodes, task_index=args.task_index)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="JSON written by evaluate_temporal_completion.py")
    parser.add_argument("--output", type=Path, required=True, help="HTML output path")
    parser.add_argument("--split", choices=("val", "test", "both"), default="both")
    parser.add_argument("--task-index", type=int, choices=range(4), help="Only include one subtask index")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Maximum episodes per split (0 means all; evenly spaced preview when limited)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Prefix-head inference batch size")
    parser.add_argument("--predictions-cache", type=Path, help="Optional path for the reusable score sidecar")
    return parser


def main() -> None:
    output = visualize(_parser().parse_args())
    print(f"Wrote temporal completion curves: {output}")


if __name__ == "__main__":
    main()
