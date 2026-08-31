"""Visualize held-out breakfast progress/done predictions by subtask segment."""

# ruff: noqa: I001 -- the trainer configures XLA before importing JAX.

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from collections.abc import Sequence
import dataclasses
import html
import json
from pathlib import Path
import shutil

import av
import cv2
import numpy as np

from openpi.training import breakfast_done_data


DEFAULT_TOKEN_CACHE = Path("/home/geek/share3/vla_done/v2/qwen_done_v2/qwen_style_done_tokens_v2_shards")
DEFAULT_ANNOTATION_ROOT = Path(
    "/home/geek/share3/breakfest_data/rule_split_330-2_action_state_delay05_cut2dist10_overlap10/split"
)
DEFAULT_CHECKPOINT = Path(
    "/home/geek/share3/vla_done/v2/progress_done_head_h768_seed42_20260831/checkpoints/step_001400"
)
DEFAULT_OUTPUT = Path(
    "/home/geek/share3/vla_done/v2/progress_done_head_h768_seed42_20260831/openloop_error_cases_step1400"
)
CAMERA_KEYS = (
    "observation.image.top",
    "observation.image.left_wrist",
    "observation.image.right_wrist",
)


@dataclasses.dataclass(frozen=True)
class Point:
    frame: int
    seconds: float
    done_target: int
    done_score: float
    progress_target: float | None
    progress_score: float


@dataclasses.dataclass(frozen=True)
class Segment:
    episode: int
    task_index: int
    task_id: str
    start_frame: int
    end_frame: int | None
    points: tuple[Point, ...]

    @property
    def key(self) -> str:
        return f"ep{self.episode:06d}_task{self.task_index}"


@dataclasses.dataclass(frozen=True)
class SegmentIssues:
    reasons: tuple[str, ...]
    done_fp: int
    done_fn: int
    max_abs_error: float | None
    max_rollback: float
    focus_index: int


def _load_predictions(
    checkpoint: Path,
    token_cache: Path,
    annotation_root: Path,
) -> tuple[
    breakfast_done_data.BreakfastDoneDataset,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    try:
        from scripts import train_token_done_head as done_training
        from scripts import train_token_progress_head as progress_training
    except ModuleNotFoundError:
        import train_token_done_head as done_training
        import train_token_progress_head as progress_training

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    import openpi.models.model as model_api
    from openpi.models import completion as completion_model
    from openpi.models import progress_done

    cache = done_training.load_token_cache(token_cache)
    progress_targets, progress_valid = progress_training.build_progress_targets(
        token_cache, cache.metadata, annotation_root
    )
    dataset = breakfast_done_data.load_breakfast_done_dataset(
        Path(str(cache.metadata["dataset_root"])), annotation_root
    )
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    head_metadata = metadata["head"]
    config = completion_model.CompletionHeadConfig(
        enabled=True,
        variant="token_query_attention",
        temporal_steps=int(head_metadata["temporal_steps"]),
        hidden_dim=int(head_metadata["hidden_dim"]),
        query_count=int(head_metadata["query_count"]),
        attention_heads=int(head_metadata["attention_heads"]),
        temporal_layers=int(head_metadata["temporal_layers"]),
        dropout_rate=float(head_metadata["dropout_rate"]),
    )
    head = progress_done.TokenQueryProgressDoneHead(
        input_dim=int(cache.metadata["feature_dim"]), config=config, rngs=nnx.Rngs(0)
    )
    graphdef, state = nnx.split(head)
    restored = model_api.restore_params(checkpoint / "params", dtype=jnp.float32)
    if set(restored) != {"completion_head"}:
        raise ValueError("head checkpoint must contain only completion_head")
    state.replace_by_pure_dict(restored["completion_head"])

    @jax.jit
    def predict(tokens, masks):
        done_logits, progress_logits = nnx.merge(graphdef, state)(tokens, masks, train=False)
        return jax.nn.sigmoid(done_logits), jax.nn.sigmoid(progress_logits)

    val_indices = cache.indices("val")
    done_scores: list[np.ndarray] = []
    progress_scores: list[np.ndarray] = []
    for start in range(0, len(val_indices), 16):
        indices = val_indices[start : start + 16]
        tokens, masks = cache.histories(indices)
        batch_done, batch_progress = predict(tokens, masks)
        done_scores.append(np.asarray(batch_done))
        progress_scores.append(np.asarray(batch_progress))
    return (
        dataset,
        val_indices,
        np.concatenate(done_scores),
        np.concatenate(progress_scores),
        progress_targets[val_indices],
        progress_valid[val_indices],
    )


def _build_segments(
    dataset: breakfast_done_data.BreakfastDoneDataset,
    val_indices: np.ndarray,
    done_scores: np.ndarray,
    progress_scores: np.ndarray,
    progress_targets: np.ndarray,
    progress_valid: np.ndarray,
    *,
    fps: int,
) -> list[Segment]:
    episode_by_id = {episode.index: episode for episode in dataset.episodes}
    task_indices = {task_id: index for index, task_id in enumerate(breakfast_done_data.SUB_TASK_IDS)}
    grouped: dict[tuple[int, int], list[Point]] = defaultdict(list)
    for position, row in enumerate(val_indices):
        sample = dataset.samples[int(row)]
        task_index = task_indices[sample.current_sub_task]
        episode = episode_by_id[sample.episode_index]
        start_frame = episode.stage_starts[task_index]
        grouped[(sample.episode_index, task_index)].append(
            Point(
                frame=sample.query_frame,
                seconds=(sample.query_frame - start_frame) / fps,
                done_target=sample.label,
                done_score=float(done_scores[position]),
                progress_target=float(progress_targets[position]) if progress_valid[position] else None,
                progress_score=float(progress_scores[position]),
            )
        )

    segments: list[Segment] = []
    for (episode_id, task_index), points in sorted(grouped.items()):
        episode = episode_by_id[episode_id]
        end_frame = (
            episode.stage_starts[task_index + 1]
            if task_index + 1 < len(episode.stage_starts)
            else episode.terminal_frame
        )
        segments.append(
            Segment(
                episode=episode_id,
                task_index=task_index,
                task_id=breakfast_done_data.SUB_TASK_IDS[task_index],
                start_frame=episode.stage_starts[task_index],
                end_frame=end_frame,
                points=tuple(sorted(points, key=lambda point: point.frame)),
            )
        )
    return segments


def _segment_issues(
    segment: Segment,
    *,
    max_progress_error: float,
    rollback_threshold: float,
) -> SegmentIssues | None:
    done_predictions = [point.done_score >= 0.5 for point in segment.points]
    pairs = tuple(zip(done_predictions, segment.points, strict=True))
    done_fp = sum(prediction and not point.done_target for prediction, point in pairs)
    done_fn = sum(not prediction and point.done_target for prediction, point in pairs)
    errors = [
        (index, abs(point.progress_score - point.progress_target))
        for index, point in enumerate(segment.points)
        if point.progress_target is not None
    ]
    max_error_index, largest_error = max(errors, key=lambda item: item[1]) if errors else (0, None)
    rollbacks = [
        (index, segment.points[index - 1].progress_score - segment.points[index].progress_score)
        for index in range(1, len(segment.points))
    ]
    rollback_index, largest_rollback = max(rollbacks, key=lambda item: item[1]) if rollbacks else (0, 0.0)
    reasons = []
    focus_candidates: list[tuple[float, int]] = []
    if done_fp:
        reasons.append(f"Done FP x{done_fp}")
        first_fp = next(
            index for index, (prediction, point) in enumerate(pairs) if prediction and not point.done_target
        )
        focus_candidates.append((1.0, first_fp))
    if done_fn:
        reasons.append(f"Done FN x{done_fn}")
        first_fn = next(
            index for index, (prediction, point) in enumerate(pairs) if not prediction and point.done_target
        )
        focus_candidates.append((1.0, first_fn))
    if largest_error is not None and largest_error >= max_progress_error:
        reasons.append(f"Progress |error| {largest_error:.3f}")
        focus_candidates.append((largest_error / max_progress_error, max_error_index))
    if largest_rollback >= rollback_threshold:
        reasons.append(f"Progress rollback {largest_rollback:.3f}")
        focus_candidates.append((largest_rollback / rollback_threshold, rollback_index))
    if not reasons:
        return None
    focus_index = max(focus_candidates, key=lambda item: item[0])[1]
    return SegmentIssues(
        reasons=tuple(reasons),
        done_fp=done_fp,
        done_fn=done_fn,
        max_abs_error=largest_error,
        max_rollback=largest_rollback,
        focus_index=focus_index,
    )


def _video_path(dataset_root: Path, episode: int, camera: str, *, chunks_size: int) -> Path:
    return dataset_root / "videos" / f"chunk-{episode // chunks_size:03d}" / camera / f"episode_{episode:06d}.mp4"


def _read_frames(video_path: Path, frame_indices: Sequence[int], *, fps: int) -> dict[int, np.ndarray]:
    targets = set(frame_indices)
    if not targets:
        return {}
    decoded: dict[int, np.ndarray] = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        container.seek(int((min(targets) / fps) / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            decoded_index = round(float(frame.pts * stream.time_base) * fps)
            if decoded_index in targets:
                decoded[decoded_index] = frame.to_ndarray(format="bgr24")
            if decoded_index >= max(targets):
                break
    missing = targets - decoded.keys()
    if missing:
        raise RuntimeError(f"cannot decode frames {sorted(missing)} from {video_path}")
    return decoded


def _save_monitor_images(
    segment: Segment,
    output_dir: Path,
    *,
    dataset_root: Path,
    chunks_size: int,
    fps: int,
) -> list[dict[str, object]]:
    selected = segment.points
    frame_indices = [point.frame for point in selected]
    camera_frames = [
        _read_frames(
            _video_path(dataset_root, segment.episode, camera, chunks_size=chunks_size), frame_indices, fps=fps
        )
        for camera in CAMERA_KEYS
    ]
    images: list[dict[str, object]] = []
    for index, point in enumerate(selected):
        views = [frames[point.frame] for frames in camera_frames]
        height = min(320, *(view.shape[0] for view in views))
        resized = [
            cv2.resize(view, (round(view.shape[1] * height / view.shape[0]), height), interpolation=cv2.INTER_AREA)
            for view in views
        ]
        combined = np.hstack(resized)
        path = output_dir / "frames" / segment.key / f"point_{index:03d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), combined, [cv2.IMWRITE_JPEG_QUALITY, 82]):
            raise RuntimeError(f"failed to write image: {path}")
        images.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "frame": point.frame,
                "seconds": point.seconds,
            }
        )
    return images


def _html_document(pages: Sequence[dict[str, object]], *, checkpoint: Path) -> str:
    encoded = json.dumps(list(pages), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    title = html.escape(f"Progress + Done · {checkpoint.name}")
    template = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
body{margin:0;background:#f3f4f6;color:#111827;font:14px system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:18px}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:14px}button,select{padding:7px 10px;border:1px solid #d1d5db;border-radius:6px;background:white}
button{cursor:pointer}.muted{color:#6b7280}.panel{background:white;border-radius:10px;padding:14px;box-shadow:0 1px 3px #0001}
h1{font-size:19px;margin:0 0 4px}#reason{color:#b45309;margin-bottom:10px}#curve svg{width:100%;display:block}
.pointbar{display:flex;gap:10px;align-items:center;margin:10px 0}.pointbar input{flex:1}.image{width:100%;display:block;border-radius:7px}
#caption{font-size:13px;color:#374151;margin-top:6px}@media(max-width:700px){main{padding:8px}.bar{gap:5px}}
</style></head><body><main><div class="bar"><button id="prevCase">← Previous case</button><button id="nextCase">Next case →</button>
<select id="jump"></select><span id="caseCount" class="muted"></span></div><section class="panel"><h1 id="heading"></h1>
<div id="reason"></div><div id="curve"></div><div class="pointbar"><button id="prevPoint">← Point</button><input id="pointSlider" type="range" min="0" step="1"><button id="nextPoint">Point →</button><span id="pointCount" class="muted"></span></div>
<img id="image" class="image" alt="Top, left wrist, and right wrist cameras"><div id="caption"></div></section></main>
<script id="data" type="application/json">__DATA__</script><script>
const pages=JSON.parse(document.getElementById('data').textContent),jump=document.getElementById('jump'),slider=document.getElementById('pointSlider');let page=0,point=0;
pages.forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`Episode ${p.episode} · Task ${p.task_index} · ${p.reasons.join(', ')}`;jump.appendChild(o)});
function line(values,x,y){return values.map(v=>`${x(v[0])},${y(v[1])}`).join(' ')}
function drawCurve(p){const W=1100,H=350,L=52,R=18,T=28,B=42,iw=W-L-R,ih=H-T-B,maxX=Math.max(1,...p.points.map(x=>x.seconds));
 const x=v=>L+v/maxX*iw,y=v=>T+(1-Math.max(0,Math.min(1,v)))*ih,grid=[0,.25,.5,.75,1];
 const grids=grid.map(v=>`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="#e5e7eb"/><text x="${L-8}" y="${y(v)+4}" text-anchor="end" fill="#6b7280">${v.toFixed(2)}</text>`).join('');
 const progress=line(p.points.map(v=>[v.seconds,v.progress_score]),x,y),done=line(p.points.map(v=>[v.seconds,v.done_score]),x,y);
 const targets=p.points.filter(v=>v.progress_target!==null),target=targets.length?`<polyline points="${line(targets.map(v=>[v.seconds,v.progress_target]),x,y)}" fill="none" stroke="#2563eb" stroke-width="1.6" stroke-dasharray="7 5"/>`:'';
 const boundary=p.boundary_seconds===null?'':`<line x1="${x(p.boundary_seconds)}" x2="${x(p.boundary_seconds)}" y1="${T}" y2="${H-B}" stroke="#111827" stroke-dasharray="3 4"/>`;
 const current=p.points[point],cx=x(current.seconds);
 document.getElementById('curve').innerHTML=`<svg viewBox="0 0 ${W} ${H}">${grids}<line x1="${L}" x2="${W-R}" y1="${y(.5)}" y2="${y(.5)}" stroke="#9ca3af" stroke-dasharray="2 4"/>${boundary}${target}<polyline points="${progress}" fill="none" stroke="#2563eb" stroke-width="2.5"/><polyline points="${done}" fill="none" stroke="#f97316" stroke-width="2.3"/><line x1="${cx}" x2="${cx}" y1="${T}" y2="${H-B}" stroke="#16a34a" stroke-width="2"/><circle cx="${cx}" cy="${y(current.progress_score)}" r="4" fill="#2563eb"/><circle cx="${cx}" cy="${y(current.done_score)}" r="4" fill="#f97316"/><text x="${L}" y="17" fill="#2563eb">Progress prediction</text><text x="${L+145}" y="17" fill="#2563eb">-- Progress target</text><text x="${L+285}" y="17" fill="#f97316">Done probability</text><text x="${L}" y="${H-10}" fill="#6b7280">0 s</text><text x="${W-R}" y="${H-10}" text-anchor="end" fill="#6b7280">${maxX.toFixed(1)} s</text></svg>`;}
function drawPoint(){const p=pages[page],v=p.points[point],image=p.images[point];slider.value=point;document.getElementById('pointCount').textContent=`${point+1} / ${p.points.length}`;document.getElementById('image').src=image.path;
 const target=v.progress_target===null?'N/A':v.progress_target.toFixed(3),error=v.progress_target===null?'N/A':Math.abs(v.progress_score-v.progress_target).toFixed(3);
 document.getElementById('caption').textContent=`t=${v.seconds.toFixed(1)}s · frame ${v.frame} · Progress ${v.progress_score.toFixed(3)} / GT ${target} · |error| ${error} · Done ${v.done_score.toFixed(3)} / GT ${v.done_target}`;drawCurve(p)}
function drawCase(){const p=pages[page];jump.value=page;point=p.focus_index;slider.max=p.points.length-1;document.getElementById('caseCount').textContent=`${page+1} / ${pages.length}`;document.getElementById('heading').textContent=`Episode ${p.episode} · Task ${p.task_index} · ${p.task_id}`;document.getElementById('reason').textContent=p.reasons.join(' · ');drawPoint()}
document.getElementById('prevCase').onclick=()=>{page=(page-1+pages.length)%pages.length;drawCase()};document.getElementById('nextCase').onclick=()=>{page=(page+1)%pages.length;drawCase()};jump.onchange=()=>{page=Number(jump.value);drawCase()};
document.getElementById('prevPoint').onclick=()=>{point=Math.max(0,point-1);drawPoint()};document.getElementById('nextPoint').onclick=()=>{point=Math.min(pages[page].points.length-1,point+1);drawPoint()};slider.oninput=()=>{point=Number(slider.value);drawPoint()};
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON'].includes(e.target.tagName))return;if(e.key==='ArrowLeft')document.getElementById('prevPoint').click();if(e.key==='ArrowRight')document.getElementById('nextPoint').click();if(e.key==='PageUp')document.getElementById('prevCase').click();if(e.key==='PageDown')document.getElementById('nextCase').click()});drawCase();
</script></body></html>"""
    return template.replace("__TITLE__", title).replace("__DATA__", encoded)


def _embed_assets(pages: Sequence[dict[str, object]], output: Path) -> list[dict[str, object]]:
    def data_url(relative_path: str) -> str:
        path = output / relative_path
        media_type = "image/png" if path.suffix == ".png" else "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"

    embedded = []
    for page in pages:
        item = dict(page)
        item["images"] = [
            {**image, "path": data_url(str(image["path"]))}
            for image in page["images"]  # type: ignore[union-attr]
        ]
        embedded.append(item)
    return embedded


def visualize(args: argparse.Namespace) -> Path:
    if args.max_progress_error <= 0 or args.rollback_threshold <= 0:
        raise ValueError("error and rollback thresholds must be positive")
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"refusing to overwrite visualization output: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    dataset, indices, done, progress, targets, valid = _load_predictions(
        args.checkpoint.resolve(), args.token_cache.resolve(), args.annotation_root.resolve()
    )
    dataset_root = Path(args.dataset_root).resolve()
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = int(info["fps"])
    segments = _build_segments(dataset, indices, done, progress, targets, valid, fps=fps)
    selected = [
        (segment, issues)
        for segment in segments
        if (
            issues := _segment_issues(
                segment,
                max_progress_error=args.max_progress_error,
                rollback_threshold=args.rollback_threshold,
            )
        )
        is not None
    ]
    print(f"Selected {len(selected)}/{len(segments)} error segments", flush=True)
    pages = []
    for number, (segment, issues) in enumerate(selected, start=1):
        images = _save_monitor_images(
            segment, output, dataset_root=dataset_root, chunks_size=int(info["chunks_size"]), fps=fps
        )
        pages.append(
            {
                "episode": segment.episode,
                "task_index": segment.task_index,
                "task_id": segment.task_id,
                "reasons": issues.reasons,
                "done_fp": issues.done_fp,
                "done_fn": issues.done_fn,
                "max_abs_error": issues.max_abs_error,
                "max_rollback": issues.max_rollback,
                "focus_index": issues.focus_index,
                "boundary_seconds": (
                    (segment.end_frame - segment.start_frame) / fps if segment.end_frame is not None else None
                ),
                "points": [dataclasses.asdict(point) for point in segment.points],
                "images": images,
            }
        )
        print(f"Rendered {number}/{len(selected)} {segment.key}", flush=True)
    if not pages:
        raise ValueError("no segments matched the error criteria")
    index = output / "index.html"
    index.write_text(
        _html_document(_embed_assets(pages, output), checkpoint=args.checkpoint.resolve()), encoding="utf-8"
    )
    return index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--token-cache", type=Path, default=DEFAULT_TOKEN_CACHE)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/geek/share3/breakfest_data/agilex_make_breakfast_330-2"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-progress-error", type=float, default=0.20)
    parser.add_argument("--rollback-threshold", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    print(f"Wrote visualization: {visualize(_parser().parse_args())}")


if __name__ == "__main__":
    main()
