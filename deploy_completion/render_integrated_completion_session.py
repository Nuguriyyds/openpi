"""Render an integrated completion deployment session as an interactive HTML report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _report_payload(session_dir: Path) -> dict[str, Any]:
    session = _read_json(session_dir / "session.json")
    events = _read_jsonl(session_dir / "events.jsonl")
    frame_index_path = session_dir / "top_camera_frames.jsonl"
    frames = _read_jsonl(frame_index_path) if frame_index_path.is_file() else []
    control_hz = float(session.get("control_hz", 30.0))

    scores = []
    switches = []
    max_step = 0
    for event in events:
        step_value = event.get("control_step")
        if isinstance(step_value, int):
            max_step = max(max_step, step_value)
        if event.get("event") == "completion_result" and isinstance(event.get("score"), (int, float)):
            scores.append(
                {
                    "step": int(event.get("control_step", 0)),
                    "score": float(event["score"]),
                    "task": event.get("task_index"),
                    "history_ready": bool(event.get("history_ready", False)),
                    "history_size": int(event.get("history_size", 0)),
                    "relative_times": event.get("relative_times", []),
                }
            )
        event_name = str(event.get("event", ""))
        if event_name in {"auto_prompt_switch", "manual_prompt_switch", "episode_complete"}:
            switches.append(
                {
                    "step": int(event.get("control_step", 0)),
                    "event": event_name,
                    "source": event.get("source", event_name.split("_", 1)[0]),
                    "old_task": event.get("old_task_index", event.get("task_index")),
                    "new_task": event.get("new_task_index"),
                    "score": event.get("score"),
                }
            )
    for frame in frames:
        if isinstance(frame.get("control_step"), int):
            max_step = max(max_step, int(frame["control_step"]))

    return {
        "session": {
            "start": session.get("session_start_wall_time"),
            "threshold": float(session.get("completion_threshold", 0.6)),
            "control_hz": control_hz,
            "completion_head": session.get("completion_head", {}),
            "vla": session.get("vla", {}),
        },
        "scores": scores,
        "switches": switches,
        "frames": frames,
        "max_step": max_step,
        "video_fps": control_hz,
        "video_available": (session_dir / "top_camera.mp4").is_file(),
    }


def _html_document(payload: dict[str, Any], title: str) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    escaped_title = html.escape(title)
    template = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#121a2e;--grid:#32405e;--text:#e2e8f0;--muted:#94a3b8;--score:#ffa94d;--threshold:#b69cff;--cursor:#f8fafc;--auto:#55e6b3;--manual:#54b5ff;--complete:#f472b6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,sans-serif}.page{max-width:1500px;margin:auto;padding:22px}h1{font-size:22px;margin:0 0 4px}.sub{color:var(--muted);margin-bottom:14px}.cards{display:flex;flex-wrap:wrap;gap:9px;margin-bottom:14px}.card{background:var(--panel);border-radius:9px;padding:9px 13px;min-width:145px}.card span{display:block;color:var(--muted);font-size:11px}.card b{font-size:17px}.panel{background:var(--panel);border-radius:12px;padding:14px;margin-bottom:14px}.video-wrap{display:flex;justify-content:center;background:#05070d;border-radius:8px;overflow:hidden}video{display:block;width:min(100%,1100px);max-height:62vh;background:#000}.empty{padding:55px;color:#fca5a5}.now{margin-top:9px;color:var(--muted)}svg{display:block;width:100%;height:auto;min-height:420px;touch-action:none}.legend{display:flex;flex-wrap:wrap;gap:18px;margin-top:6px;color:#cbd5e1}.legend i{display:inline-block;width:27px;height:3px;margin-right:6px;vertical-align:middle}.hint{color:var(--muted);font-size:12px;margin-top:8px}
</style></head><body><div class="page">
<h1>__TITLE__</h1><div class="sub" id="subtitle"></div><div class="cards" id="cards"></div>
<section class="panel"><div class="video-wrap" id="videoWrap"><video id="video" controls preload="metadata"><source src="top_camera.mp4" type="video/mp4"></video></div><div class="now" id="now">等待视频...</div></section>
<section class="panel"><div id="chart"></div><div class="legend"><span><i style="background:var(--score)"></i>完成分数</span><span><i style="background:var(--threshold)"></i>阈值</span><span><i style="background:var(--auto)"></i>自动切换</span><span><i style="background:var(--manual)"></i>人工切换</span><span><i style="background:var(--complete)"></i>Episode 完成</span><span><i style="background:var(--cursor)"></i>视频当前位置</span></div><div class="hint">点击曲线可跳转到对应视频位置；播放视频时，白色竖线会同步移动。</div></section>
</div><script id="payload" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('payload').textContent),S=D.session,scores=D.scores,frames=D.frames,switches=D.switches;
const video=document.getElementById('video'),videoWrap=document.getElementById('videoWrap'),chart=document.getElementById('chart'),now=document.getElementById('now');
const hz=S.control_hz||30,maxStep=Math.max(1,D.max_step||0),fmt=v=>v==null?'—':Number(v).toFixed(3),seconds=s=>(s/hz).toFixed(1);
document.getElementById('subtitle').textContent=`开始时间 ${S.start||'—'} · control ${hz} Hz · 横轴为真实控制时间`;
const autoN=switches.filter(x=>x.event==='auto_prompt_switch').length,manualN=switches.filter(x=>x.event==='manual_prompt_switch').length;
document.getElementById('cards').innerHTML=[["推理次数",scores.length],["阈值",fmt(S.threshold)],["自动切换",autoN],["人工切换",manualN],["记录时长",`${seconds(maxStep)} s`]].map(([k,v])=>`<div class="card"><span>${k}</span><b>${v}</b></div>`).join('');
if(!D.video_available){videoWrap.innerHTML='<div class="empty">本次 session 没有可播放的视频。</div>'}
const W=1320,H=470,L=68,R=24,T=28,B=62,iw=W-L-R,ih=H-T-B,x=s=>L+s/maxStep*iw,y=v=>T+(1-Math.max(0,Math.min(1,v)))*ih;
const grid=[0,.25,.5,.75,1].map(v=>`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${L-10}" y="${y(v)+4}" text-anchor="end" fill="var(--muted)">${v.toFixed(2)}</text>`).join('');
const points=scores.map(p=>`${x(p.step)},${y(p.score)}`).join(' ');
const dots=scores.map(p=>`<circle cx="${x(p.step)}" cy="${y(p.score)}" r="${p.history_ready?3.2:2.5}" fill="${p.history_ready?'var(--score)':'#64748b'}"><title>${seconds(p.step)}s · step ${p.step} · task ${p.task} · score ${fmt(p.score)} · history ${p.history_size}/3</title></circle>`).join('');
const switchLines=switches.map(s=>{const c=s.event==='auto_prompt_switch'?'var(--auto)':s.event==='manual_prompt_switch'?'var(--manual)':'var(--complete)',label=s.event==='episode_complete'?'complete':`${s.source}: ${s.old_task}→${s.new_task}`;return `<line x1="${x(s.step)}" x2="${x(s.step)}" y1="${T}" y2="${T+ih}" stroke="${c}" stroke-width="2"><title>${seconds(s.step)}s · ${label} · score ${fmt(s.score)}</title></line><text x="${x(s.step)+4}" y="${T+14}" fill="${c}" font-size="11">${label}</text>`}).join('');
const thresholdY=y(S.threshold),ticks=[0,.25,.5,.75,1].map(q=>{const st=Math.round(maxStep*q);return `<text x="${x(st)}" y="${H-23}" text-anchor="middle" fill="var(--muted)">${seconds(st)}s</text>`}).join('');
chart.innerHTML=`<svg id="scoreSvg" viewBox="0 0 ${W} ${H}" aria-label="completion score timeline"><rect width="${W}" height="${H}" fill="var(--panel)"/>${grid}<line x1="${L}" x2="${L}" y1="${T}" y2="${T+ih}" stroke="#64748b"/><line x1="${L}" x2="${W-R}" y1="${T+ih}" y2="${T+ih}" stroke="#64748b"/><line x1="${L}" x2="${W-R}" y1="${thresholdY}" y2="${thresholdY}" stroke="var(--threshold)" stroke-dasharray="8 6"/><text x="${W-R}" y="${thresholdY-7}" text-anchor="end" fill="var(--threshold)">threshold ${fmt(S.threshold)}</text><polyline points="${points}" fill="none" stroke="var(--score)" stroke-width="2.5" stroke-linejoin="round"/>${dots}${switchLines}<line id="cursor" x1="${L}" x2="${L}" y1="${T}" y2="${T+ih}" stroke="var(--cursor)" stroke-width="2"/>${ticks}<text x="${L}" y="${H-5}" fill="var(--muted)">运行时间</text></svg>`;
const svg=document.getElementById('scoreSvg'),cursor=document.getElementById('cursor');
function nearest(items,value,key){let lo=0,hi=items.length-1;if(hi<0)return null;while(lo<hi){const m=Math.floor((lo+hi)/2);if(items[m][key]<value)lo=m+1;else hi=m}if(lo>0&&Math.abs(items[lo-1][key]-value)<Math.abs(items[lo][key]-value))lo--;return items[lo]}
function videoStep(){if(!frames.length)return Math.round(video.currentTime*hz);const frame=nearest(frames,Math.round(video.currentTime*D.video_fps),'video_frame_index');return frame?frame.control_step:0}
function updateCursor(){const step=Math.max(0,Math.min(maxStep,videoStep())),cx=x(step);cursor.setAttribute('x1',cx);cursor.setAttribute('x2',cx);const p=nearest(scores,step,'step');now.textContent=`视频 ${video.currentTime.toFixed(1)}s · 控制时间 ${seconds(step)}s · step ${step} · task ${p?.task??'—'} · score ${fmt(p?.score)} · history ${p?.history_size??0}/3`}
video.addEventListener('timeupdate',updateCursor);video.addEventListener('seeked',updateCursor);video.addEventListener('loadedmetadata',updateCursor);
svg.addEventListener('click',e=>{if(!frames.length||!D.video_available)return;const r=svg.getBoundingClientRect(),px=(e.clientX-r.left)/r.width*W,step=Math.round(Math.max(0,Math.min(1,(px-L)/iw))*maxStep),f=nearest(frames,step,'control_step');if(f){video.currentTime=f.video_frame_index/D.video_fps;video.play().catch(()=>{})}});
</script></body></html>"""
    return template.replace("__TITLE__", escaped_title).replace("__DATA__", encoded)


def render_session_html(session_dir: Path, output: Path | None = None) -> Path:
    session_dir = session_dir.resolve()
    output = (output or session_dir / "report.html").resolve()
    payload = _report_payload(session_dir)
    output.write_text(_html_document(payload, f"Completion deployment · {session_dir.name}"), encoding="utf-8")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(f"Wrote completion deployment HTML: {render_session_html(args.session_dir, args.output)}")


if __name__ == "__main__":
    main()
