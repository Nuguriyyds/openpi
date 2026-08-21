"""Build a self-contained HTML viewer for a semi-closed full-trajectory report."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import html
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"semi-closed report not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("episodes"), list):
        raise ValueError("semi-closed report must contain an episodes list")
    required = {"mode", "threshold", "threshold_source", "summary", "episodes"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"semi-closed report is missing fields: {sorted(missing)}")
    threshold = float(value["threshold"])
    if not np.isfinite(threshold) or not 0.0 <= threshold <= float(np.nextafter(1.0, np.inf)):
        raise ValueError("semi-closed report threshold is outside the supported range")
    return value


def _select_episodes(episodes: list[Mapping[str, Any]], max_episodes: int) -> list[Mapping[str, Any]]:
    if max_episodes < 0:
        raise ValueError("--max-episodes must be non-negative (0 means all)")
    if max_episodes == 0 or len(episodes) <= max_episodes:
        return episodes
    positions = np.linspace(0, len(episodes) - 1, num=max_episodes, dtype=np.int64)
    return [episodes[int(position)] for position in positions]


def _html_document(report: Mapping[str, Any], *, max_episodes: int) -> str:
    episodes = _select_episodes(list(report["episodes"]), max_episodes)
    data = {
        "mode": report["mode"],
        "threshold": float(report["threshold"]),
        "threshold_source": report["threshold_source"],
        "config_name": report.get("config_name", ""),
        "checkpoint": report.get("checkpoint", ""),
        "summary": report["summary"],
        "episodes": episodes,
    }
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    title = html.escape(f"Semi-closed temporal completion · {report.get('mode', '')}")
    template = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#0f172a;--panel:#111827;--grid:#334155;--text:#dbeafe;--target:#61e6b5;--score:#ffab45;--threshold:#b7a2ff;--ref:#60a5fa;--trigger:#fb7185}
body{margin:0;padding:24px;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,sans-serif}h1{margin:0 0 4px;font-size:20px}.sub{color:#94a3b8;margin-bottom:18px}
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;background:var(--panel);padding:12px;border-radius:10px;margin-bottom:12px}label{color:#94a3b8}select{background:#1e293b;color:var(--text);border:1px solid #475569;border-radius:5px;padding:6px}
.cards{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 12px}.card{background:var(--panel);border-radius:8px;padding:8px 12px;min-width:125px}.card span{display:block;color:#94a3b8;font-size:11px}.card b{font-size:17px}
.panel{background:var(--panel);border-radius:10px;padding:14px}#info{color:#94a3b8;margin-bottom:8px}svg{width:100%;height:auto;min-height:510px;display:block}.legend{display:flex;flex-wrap:wrap;gap:16px;color:#cbd5e1;margin-top:8px}.legend i{width:28px;height:3px;display:inline-block;vertical-align:middle;margin-right:5px}.target{background:var(--target)}.score{background:var(--score)}.threshold{background:var(--threshold)}.ref{background:var(--ref)}.trigger{background:var(--trigger)}.empty{color:#fca5a5;padding:40px;text-align:center}.status{font-weight:600}
</style></head><body>
<h1>__TITLE__</h1><div class="sub">30 fps complete trajectory · 2 Hz prefix/head only · threshold __THRESHOLD__ (__SOURCE__) · checkpoint __CHECKPOINT__</div>
<div class="controls"><label>Filter <select id="filter"><option value="all">all</option><option value="all_correct">all correct</option><option value="has_failure">has failure</option><option value="early">early</option><option value="late">late</option><option value="missed">missed</option><option value="task0">task 0 failure</option><option value="task1">task 1 failure</option><option value="task2">task 2 failure</option><option value="task3">task 3 failure</option></select></label><label>Episode <select id="episode"></select></label></div>
<div id="cards" class="cards"></div><div class="panel"><div id="info"></div><div id="chart"></div><div class="legend"><span><i class="target"></i>Target for active task</span><span><i class="score"></i>Predicted score</span><span><i class="threshold"></i>Threshold</span><span><i class="ref"></i>GT reference tick</span><span><i class="trigger"></i>Actual trigger</span></div></div>
<script id="payload" type="application/json">__DATA__</script><script>
const DATA=JSON.parse(document.getElementById('payload').textContent), filterEl=document.getElementById('filter'), episodeEl=document.getElementById('episode'), infoEl=document.getElementById('info'), chartEl=document.getElementById('chart'), cardsEl=document.getElementById('cards');
const fmt=v=>v==null?'—':(typeof v==='number'?v.toFixed(3):String(v));
function boundaries(item){return item.boundary_results||[]}
function matches(item){const f=filterEl.value,b=boundaries(item);if(f==='all')return true;if(f==='all_correct')return !!item.all_correct;if(f==='has_failure')return !item.all_correct;if(f==='early')return b.some(x=>x.classification==='early');if(f==='late')return b.some(x=>x.classification==='late');if(f==='missed')return b.some(x=>x.classification==='missed');if(f.startsWith('task')){const k=Number(f.slice(4));return b.some(x=>x.task_index===k&&!['correct','unavailable'].includes(x.classification))}return true}
function selected(){return DATA.episodes.filter(matches)}
function updateCards(){const s=DATA.summary||{},b=s.boundaries||{};const entries=[['all correct',s.all_correct_rate],['done rate',s.done_rate],['correct boundaries',b.correct_rate],['early',b.early],['late',b.late],['missed',b.missed],['prompt mismatch',s.prompt_mismatch_rate],['warmup ticks',s.history_not_ready_rate]];cardsEl.innerHTML=entries.map(([k,v])=>`<div class="card"><span>${k}</span><b>${fmt(v)}</b></div>`).join('')}
function updateEpisodes(){const items=selected();episodeEl.innerHTML='';items.forEach((x,i)=>{const o=document.createElement('option');o.value=String(i);o.textContent=`episode ${x.full_episode_id} · ${x.all_correct?'correct':'failure'}`;episodeEl.appendChild(o)});draw()}
function xScale(frame,min,max,L,iw){return L+(frame-min)/Math.max(1,max-min)*iw}function yScale(v,T,ih){return T+(1-Math.max(0,Math.min(1,v)))*ih}
function runs(values){const out=[];if(!values.length)return out;let start=0;for(let i=1;i<=values.length;i++){if(i===values.length||values[i]!==values[start]){out.push([start,i-1,values[start]]);start=i}}return out}
function draw(){const items=selected(),item=items[Number(episodeEl.value)||0];if(!item){infoEl.textContent='No episodes in this filter.';chartEl.innerHTML='<div class="empty">No episodes.</div>';return}const ticks=item.ticks||[],frames=ticks.map(x=>x.frame_index),scores=ticks.map(x=>x.score),labels=ticks.map(x=>x.target),minX=0,maxX=Math.max(1,item.full_length-1),W=1150,H=560,L=66,R=25,T=20,B=96,iw=W-L-R,ih=H-T-B;const x=i=>xScale(frames[i],minX,maxX,L,iw),y=v=>yScale(v,T,ih),grid=[0,.25,.5,.75,1],gridSvg=grid.map(v=>`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${L-10}" y="${y(v)+4}" text-anchor="end" fill="#94a3b8">${v.toFixed(2)}</text>`).join('');const points=(vals)=>vals.map((v,i)=>v==null?'':`${x(i)},${y(v)}`).filter(Boolean).join(' ');const markers=ticks.map((t,i)=>t.score==null?'':`<circle cx="${x(i)}" cy="${y(t.score)}" r="3" fill="var(--score)"/>`).join('');const refs=(item.reference_ticks||[]).map((v,i)=>`<line x1="${xScale(v,minX,maxX,L,iw)}" x2="${xScale(v,minX,maxX,L,iw)}" y1="${T}" y2="${T+ih}" stroke="var(--ref)" stroke-dasharray="4 5"/><text x="${xScale(v,minX,maxX,L,iw)+3}" y="${T+14+i*14}" fill="var(--ref)" font-size="11">GT t${i}=${v}</text>`).join('');const triggers=(item.predicted_ticks||[]).map((v,i)=>v==null?'':`<line x1="${xScale(v,minX,maxX,L,iw)}" x2="${xScale(v,minX,maxX,L,iw)}" y1="${T}" y2="${T+ih}" stroke="var(--trigger)" stroke-width="2"/><text x="${xScale(v,minX,maxX,L,iw)+3}" y="${T+14+i*14}" fill="var(--trigger)" font-size="11">trigger t${i}=${v}</text>`).join('');const active=runs(ticks.map(t=>t.active_task_index)),oracle=runs(ticks.map(t=>t.oracle_task_index));const taskColor=['#38bdf8','#a78bfa','#fbbf24','#fb7185'];const bar=(segments,y0,label,withPrompt)=>segments.map(([a,b,k])=>`<rect x="${x(a)}" y="${y0}" width="${Math.max(1,x(b)-x(a))}" height="10" fill="${taskColor[k]||'#94a3b8'}">${withPrompt?`<title>${label} task ${k}: ${ticks[a].active_prompt||''}</title>`:''}</rect><text x="${x(a)+2}" y="${y0+9}" font-size="9" fill="#0f172a">${label}${k}</text>`).join('');const status=boundaries(item).map(b=>`task ${b.task_index}: <span class="status">${b.classification}</span>`).join(' · ');infoEl.innerHTML=`episode ${item.full_episode_id} · length ${item.full_length} · done=${item.done} · ${status}<br>prompt mismatches ${item.prompt_mismatch_count}/${ticks.length} · history-not-ready ${item.history_not_ready_count}/${ticks.length}`;const thresholdY=y(DATA.threshold);chartEl.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="semi-closed temporal completion curve"><rect width="${W}" height="${H}" fill="var(--panel)"/><g>${gridSvg}</g><line x1="${L}" x2="${L}" y1="${T}" y2="${T+ih}" stroke="#94a3b8"/><line x1="${L}" x2="${W-R}" y1="${T+ih}" y2="${T+ih}" stroke="#94a3b8"/>${refs}${triggers}<line x1="${L}" x2="${W-R}" y1="${thresholdY}" y2="${thresholdY}" stroke="var(--threshold)" stroke-dasharray="7 6"/><polyline points="${points(labels)}" fill="none" stroke="var(--target)" stroke-width="3"/><polyline points="${points(scores)}" fill="none" stroke="var(--score)" stroke-width="2.5"/>${markers}<text x="${L}" y="${H-62}" fill="#94a3b8">active task</text>${bar(active,H-58,'A',true)}<text x="${L}" y="${H-32}" fill="#94a3b8">oracle task</text>${bar(oracle,H-28,'O',false)}<text x="${W-R}" y="${H-62}" text-anchor="end" fill="var(--threshold)">threshold ${DATA.threshold.toFixed(4)}</text><text x="${L}" y="${H-8}" fill="#94a3b8">frame 0</text><text x="${W-R}" y="${H-8}" text-anchor="end" fill="#94a3b8">frame ${maxX}</text></svg>`}
filterEl.addEventListener('change',updateEpisodes);episodeEl.addEventListener('change',draw);updateCards();updateEpisodes();
</script></body></html>"""
    return (
        template.replace("__TITLE__", title)
        .replace("__THRESHOLD__", f"{float(report['threshold']):.4f}")
        .replace("__SOURCE__", html.escape(str(report["threshold_source"])))
        .replace("__CHECKPOINT__", html.escape(str(report.get("checkpoint", ""))))
        .replace("__DATA__", encoded)
    )


def visualize(args: argparse.Namespace) -> Path:
    report = _read_report(args.report.resolve())
    document = _html_document(report, max_episodes=args.max_episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    return args.output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=0, help="0 means all test episodes")
    return parser


def main() -> None:
    output = visualize(_parser().parse_args())
    print(f"Wrote semi-closed temporal completion HTML: {output}")


if __name__ == "__main__":
    main()
