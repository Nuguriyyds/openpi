#!/usr/bin/env python3
"""Align an online progress JSONL log with a recorded Top-camera video."""

from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
from pathlib import Path

import cv2
import numpy as np


PALETTE_HEX = ("#4cc9f0", "#f72585", "#fca311", "#80ed99")
PALETTE_BGR = ((240, 201, 76), (133, 37, 247), (17, 163, 252), (153, 237, 128))


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def nearest_index(sorted_values: np.ndarray, value: float) -> int:
    position = int(np.searchsorted(sorted_values, value))
    if position <= 0:
        return 0
    if position >= len(sorted_values):
        return len(sorted_values) - 1
    before = position - 1
    return before if abs(sorted_values[before] - value) <= abs(sorted_values[position] - value) else position


def align_events(events: list[dict], frames: list[dict], fps: float) -> tuple[list[dict], list[dict], str]:
    inferences = [record.copy() for record in events if record.get("event") == "inference"]
    completions = [record.copy() for record in events if record.get("event") == "subtask_complete"]
    if not inferences:
        raise ValueError("events.jsonl does not contain inference records")
    if not frames:
        raise ValueError("top_frames.jsonl is empty")

    frame_ros = np.asarray([float(record.get("ros_timestamp_ns", 0)) for record in frames])
    inference_ros = np.asarray(
        [float(record.get("observation_top_ros_timestamp_ns", 0)) for record in inferences]
    )
    use_ros = bool(np.all(frame_ros > 0) and np.all(inference_ros > 0))

    if use_ros:
        frame_keys = frame_ros
        inference_keys = inference_ros
        alignment = "ros_timestamp_ns"
    else:
        frame_keys = np.asarray([float(record["timestamp_unix"]) for record in frames])
        inference_keys = np.asarray(
            [
                float(
                    record.get(
                        "observation_timestamp_unix",
                        datetime.fromisoformat(record["timestamp_utc"]).timestamp(),
                    )
                )
                for record in inferences
            ]
        )
        alignment = "unix_receipt_timestamp"

    if np.any(np.diff(frame_keys) < 0):
        raise ValueError(f"Frame timestamps are not monotonic for alignment basis {alignment}")

    for record, key in zip(inferences, inference_keys, strict=True):
        frame_position = nearest_index(frame_keys, float(key))
        frame = frames[frame_position]
        record["video_frame_index"] = int(frame["frame_index"])
        record["video_time_s"] = float(frame.get("video_time_s", frame["frame_index"] / fps))

    for completion in completions:
        candidates = [
            record for record in inferences if record["subtask_index"] == completion["completed_index"]
        ]
        if candidates:
            completion["video_time_s"] = candidates[-1]["video_time_s"]
            completion["inference_index"] = candidates[-1]["inference_index"]

    return inferences, completions, alignment


def summarize(events: list[dict], inferences: list[dict], completions: list[dict]) -> dict:
    start = next((record for record in events if record.get("event") == "session_start"), None)
    end = next((record for record in reversed(events) if record.get("event") == "session_end"), None)
    session_seconds = None
    if start and end:
        session_seconds = (
            datetime.fromisoformat(end["timestamp_utc"]) - datetime.fromisoformat(start["timestamp_utc"])
        ).total_seconds()

    subtasks = []
    for index in sorted({int(record["subtask_index"]) for record in inferences}):
        records = [record for record in inferences if int(record["subtask_index"]) == index]
        values = np.asarray([float(record["progress_raw"]) for record in records])
        subtasks.append(
            {
                "index": index,
                "name": records[0]["subtask_name"],
                "inference_count": len(records),
                "progress_first": float(values[0]),
                "progress_last": float(values[-1]),
                "completed": any(int(record["completed_index"]) == index for record in completions),
            }
        )
    return {
        "inference_count": len(inferences),
        "completion_count": len(completions),
        "session_seconds": session_seconds,
        "subtasks": subtasks,
    }


def write_html(
    output_path: Path,
    video_name: str,
    video_duration: float,
    inferences: list[dict],
    completions: list[dict],
    summary: dict,
    alignment: str,
) -> None:
    report = {
        "video_duration": video_duration,
        "alignment": alignment,
        "palette": list(PALETTE_HEX),
        "inferences": [
            {
                "time": record["video_time_s"],
                "progress": record["progress_clipped"],
                "progress_raw": record["progress_raw"],
                "subtask_index": record["subtask_index"],
                "subtask_name": record["subtask_name"],
                "inference_index": record["inference_index"],
            }
            for record in inferences
        ],
        "completions": [
            {
                "time": record.get("video_time_s"),
                "subtask_index": record["completed_index"],
                "subtask_name": record["completed_name"],
                "reason": record["trigger_reason"],
            }
            for record in completions
            if "video_time_s" in record
        ],
    }
    report_json = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    table_rows = "".join(
        "<tr>"
        f"<td>{item['index'] + 1}</td>"
        f"<td>{html.escape(item['name'])}</td>"
        f"<td>{item['inference_count']}</td>"
        f"<td>{item['progress_first']:.3f}</td>"
        f"<td>{item['progress_last']:.3f}</td>"
        f"<td>{'yes' if item['completed'] else 'no'}</td>"
        "</tr>"
        for item in summary["subtasks"]
    )
    session_text = "n/a" if summary["session_seconds"] is None else f"{summary['session_seconds']:.2f} s"

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Breakfast progress online report</title>
<style>
body {{ margin: 0; background: #0b1020; color: #e7edf7; font-family: system-ui, sans-serif; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
h1 {{ margin: 0 0 8px; }}
.muted {{ color: #9aa8bd; }}
.card {{ background: #121a2b; border: 1px solid #25314a; border-radius: 12px; padding: 16px; margin-top: 16px; }}
video {{ width: 100%; max-height: 65vh; background: black; border-radius: 8px; }}
canvas {{ width: 100%; height: 280px; display: block; }}
#status {{ font: 600 16px ui-monospace, monospace; margin-top: 8px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #25314a; }}
</style>
</head>
<body><main>
<h1>Breakfast progress online report</h1>
<div class="muted">{summary['inference_count']} inferences · {summary['completion_count']} completions · {session_text} · alignment: {html.escape(alignment)}</div>
<div class="card"><video id="video" controls preload="metadata" src="{html.escape(video_name)}"></video></div>
<div class="card"><canvas id="chart" width="1040" height="280"></canvas><div id="status">Loading…</div></div>
<div class="card"><table><thead><tr><th>#</th><th>Subtask</th><th>Inferences</th><th>First</th><th>Last</th><th>Completed</th></tr></thead><tbody>{table_rows}</tbody></table></div>
</main>
<script>const REPORT={report_json};
const video=document.getElementById('video'), canvas=document.getElementById('chart'), ctx=canvas.getContext('2d'), status=document.getElementById('status');
const M={{left:58,right:20,top:18,bottom:38}};
function draw() {{
  const W=canvas.width,H=canvas.height,pw=W-M.left-M.right,ph=H-M.top-M.bottom;
  const duration=Math.max(REPORT.video_duration, video.duration||0, 0.001);
  const x=t=>M.left+Math.max(0,Math.min(duration,t))/duration*pw;
  const y=p=>M.top+(1-Math.max(0,Math.min(1,p)))*ph;
  ctx.fillStyle='#0d1424';ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#34425f';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(M.left,M.top);ctx.lineTo(M.left,H-M.bottom);ctx.lineTo(W-M.right,H-M.bottom);ctx.stroke();
  ctx.setLineDash([7,6]);ctx.strokeStyle='#f1fa8c';ctx.beginPath();ctx.moveTo(M.left,y(.95));ctx.lineTo(W-M.right,y(.95));ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#9aa8bd';ctx.font='13px system-ui';
  for (const p of [0,.25,.5,.75,1]) {{ctx.fillText(p.toFixed(2),8,y(p)+4);}}
  for (const f of [0,.25,.5,.75,1]) {{const t=duration*f;ctx.fillText(t.toFixed(1)+'s',x(t)-12,H-10);}}
  for (let s=0;s<4;s++) {{
    const points=REPORT.inferences.filter(e=>e.subtask_index===s); if(!points.length) continue;
    ctx.strokeStyle=REPORT.palette[s];ctx.lineWidth=3;ctx.beginPath();
    points.forEach((e,i)=>{{if(i===0)ctx.moveTo(x(e.time),y(e.progress));else ctx.lineTo(x(e.time),y(e.progress));}});ctx.stroke();
  }}
  for (const c of REPORT.completions) {{ctx.strokeStyle=REPORT.palette[c.subtask_index];ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x(c.time),M.top);ctx.lineTo(x(c.time),H-M.bottom);ctx.stroke();}}
  const now=video.currentTime||0;ctx.strokeStyle='#ffffff';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x(now),M.top);ctx.lineTo(x(now),H-M.bottom);ctx.stroke();
  let current=null;for(const e of REPORT.inferences){{if(e.time<=now)current=e;else break;}}
  if(current){{ctx.fillStyle=REPORT.palette[current.subtask_index];ctx.beginPath();ctx.arc(x(current.time),y(current.progress),6,0,Math.PI*2);ctx.fill();status.textContent=`subtask ${{current.subtask_index+1}}/4 · ${{current.subtask_name}} · progress=${{Number(current.progress_raw).toFixed(4)}} · inference #${{current.inference_index}}`;}}
  else status.textContent='Waiting for the first inference';
}}
for(const event of ['loadedmetadata','timeupdate','seeked','play','pause']) video.addEventListener(event,draw);
setInterval(()=>{{if(!video.paused)draw();}},50);draw();
</script></body></html>"""
    output_path.write_text(document, encoding="utf-8")


def open_video_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    for codec in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()
        path.unlink(missing_ok=True)
    raise RuntimeError("Could not open an MP4 writer with avc1, H264, or mp4v")


def chart_point(time_s: float, progress: float, width: int, height: int, duration: float) -> tuple[int, int]:
    left, right, top, bottom = 62, 20, 24, 42
    x = left + int(np.clip(time_s / max(duration, 1e-6), 0, 1) * (width - left - right))
    y = top + int((1 - np.clip(progress, 0, 1)) * (height - top - bottom))
    return x, y


def build_chart_base(
    width: int,
    height: int,
    duration: float,
    inferences: list[dict],
    completions: list[dict],
) -> np.ndarray:
    panel = np.full((height, width, 3), (36, 20, 13), dtype=np.uint8)
    left, right, top, bottom = 62, 20, 24, 42
    cv2.line(panel, (left, top), (left, height - bottom), (90, 66, 52), 1)
    cv2.line(panel, (left, height - bottom), (width - right, height - bottom), (90, 66, 52), 1)
    _, threshold_y = chart_point(0, 0.95, width, height, duration)
    for x in range(left, width - right, 14):
        cv2.line(panel, (x, threshold_y), (min(x + 7, width - right), threshold_y), (140, 250, 241), 1)
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        _, y = chart_point(0, value, width, height, duration)
        cv2.putText(panel, f"{value:.2f}", (7, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 180, 165), 1, cv2.LINE_AA)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        time_s = duration * fraction
        x, _ = chart_point(time_s, 0, width, height, duration)
        cv2.putText(panel, f"{time_s:.1f}s", (x - 16, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 180, 165), 1, cv2.LINE_AA)
    for subtask_index in range(4):
        records = [record for record in inferences if int(record["subtask_index"]) == subtask_index]
        for first, second in zip(records, records[1:], strict=False):
            p1 = chart_point(first["video_time_s"], first["progress_clipped"], width, height, duration)
            p2 = chart_point(second["video_time_s"], second["progress_clipped"], width, height, duration)
            cv2.line(panel, p1, p2, PALETTE_BGR[subtask_index], 2, cv2.LINE_AA)
    for record in completions:
        if "video_time_s" not in record:
            continue
        x, _ = chart_point(record["video_time_s"], 0, width, height, duration)
        cv2.line(panel, (x, top), (x, height - bottom), PALETTE_BGR[int(record["completed_index"])], 1)
    return panel


def export_mp4(
    source_path: Path,
    output_path: Path,
    inferences: list[dict],
    completions: list[dict],
) -> None:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {source_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    panel_height = 260
    writer, codec = open_video_writer(output_path, fps, (width, height + panel_height))
    chart_base = build_chart_base(width, panel_height, duration, inferences, completions)
    event_times = np.asarray([record["video_time_s"] for record in inferences])

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            time_s = frame_index / fps
            event_index = int(np.searchsorted(event_times, time_s, side="right") - 1)
            current = inferences[event_index] if event_index >= 0 else None
            panel = chart_base.copy()
            x, _ = chart_point(time_s, 0, width, panel_height, duration)
            cv2.line(panel, (x, 24), (x, panel_height - 42), (255, 255, 255), 2)
            if current is not None:
                point = chart_point(
                    current["video_time_s"], current["progress_clipped"], width, panel_height, duration
                )
                color = PALETTE_BGR[int(current["subtask_index"])]
                cv2.circle(panel, point, 6, color, -1, cv2.LINE_AA)
                label = (
                    f"subtask {int(current['subtask_index']) + 1}/4  "
                    f"progress={float(current['progress_raw']):.4f}  infer={int(current['inference_index'])}"
                )
                cv2.putText(frame, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
            combined = np.vstack([frame, panel])
            writer.write(combined)
            frame_index += 1
            if frame_index % 300 == 0:
                print(f"Rendered {frame_index}/{frame_count} frames")
    finally:
        capture.release()
        writer.release()
    print(f"MP4 written with codec {codec}: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--export-mp4", action="store_true")
    parser.add_argument("--html-name", default="progress_report.html")
    parser.add_argument("--mp4-name", default="progress_overlay.mp4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    events_path = run_dir / "events.jsonl"
    frames_path = run_dir / "top_frames.jsonl"
    metadata_path = run_dir / "top_recording.json"
    video_path = run_dir / "top.mp4"

    events = read_jsonl(events_path)
    frames = read_jsonl(frames_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open recorded video: {video_path}")
    fps = float(metadata.get("fps") or capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(metadata.get("frame_count") or capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    duration = frame_count / fps

    inferences, completions, alignment = align_events(events, frames, fps)
    summary = summarize(events, inferences, completions)
    html_path = run_dir / args.html_name
    write_html(html_path, video_path.name, duration, inferences, completions, summary, alignment)
    print(f"HTML report: {html_path}")

    if args.export_mp4:
        export_mp4(video_path, run_dir / args.mp4_name, inferences, completions)


if __name__ == "__main__":
    main()
