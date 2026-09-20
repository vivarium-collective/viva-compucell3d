"""Generate committed study artifacts for one CompuCell3D demonstration scenario.

Runs the REAL CompuCell3D engine (cc3d, via the pixi py3.10 env) and writes, into
``workspace/studies/<slug>/``:

  viz/scene.html      self-contained animated 2-D lattice viewer (play/scrub)
  charts/NN_*.png     static figures (population, composition, morphometry) + .meta.json
  metrics.csv         per-snapshot scalar metrics
  run.log             runtime + summary

The workbench SPA renders these committed artifacts; it never runs cc3d itself
(exactly the split pbg-chaste uses with its Docker engine). Scenario configs and
the run loop are reused from ``demo/demo_report.py`` so there is one source of truth.

Usage:  pixi run python scripts/gen_study_artifacts.py <scenario_id>
        scenario_id in {sorting, chemotaxis, growth, invasion}
"""
from __future__ import annotations

import csv
import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))         # pbg_compucell3d imports from the worktree tree
sys.path.insert(0, str(ROOT / "demo"))
import demo_report as demo  # noqa: E402  (CONFIGS, run_simulation)

# scenario id -> (study slug, study-relative capability tag)
SLUGS = {
    "sorting": "cell-sorting",
    "chemotaxis": "chemotaxis",
    "growth": "growth-division",
    "invasion": "spheroid-invasion",
}

TYPE_COLORS = ["#eef2f7", "#22c55e", "#f97316", "#6366f1"]  # medium, TypeA, TypeB, extra


# ── animated scene ──────────────────────────────────────────────────────────

def _b64_bytes(flat) -> str:
    import base64
    return base64.b64encode(bytes(flat)).decode("ascii")


def render_scene_html(cfg: dict, snapshots: list[dict], runtime: float) -> str:
    dim_x = cfg["config"]["dim_x"]
    dim_y = cfg["config"]["dim_y"]
    has_conc = snapshots[-1].get("conc_field") is not None

    # Global concentration max for stable quantisation/colouring across frames.
    cmax = 0.0
    if has_conc:
        for s in snapshots:
            cf = s.get("conc_field")
            if cf is not None:
                cmax = max(cmax, max(max(row) for row in cf))
    cmax = cmax or 1.0

    # Pack each frame as base64 byte arrays instead of nested-JSON int lists —
    # ~4-8x smaller, no fidelity loss for the type lattice (values 0-3); the
    # scalar field is quantised to 0-255 for display only.
    frames = []
    for s in snapshots:
        tf = s["type_field"]
        type_bytes = [tf[y][x] & 0xFF for y in range(dim_y) for x in range(dim_x)]
        entry = {"mcs": s["mcs"], "t": _b64_bytes(type_bytes)}
        if has_conc and s.get("conc_field") is not None:
            cf = s["conc_field"]
            conc_bytes = [min(255, int(round(255 * cf[y][x] / cmax)))
                          for y in range(dim_y) for x in range(dim_x)]
            entry["c"] = _b64_bytes(conc_bytes)
        frames.append(entry)
    data = json.dumps({
        "frames": frames,
        "dim": [dim_x, dim_y],
        "has_conc": has_conc,
        "colors": TYPE_COLORS,
    })

    title = cfg["title"]
    subtitle = cfg["subtitle"]
    conc_ui = (
        '<label class="tog"><input type="checkbox" id="conc" checked> concentration field</label>'
        if has_conc else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — CompuCell3D scene</title>
<style>
  :root {{
    --bg:#ffffff; --fg:#0f172a; --muted:#64748b; --panel:#f1f5f9; --border:#e2e8f0; --accent:#6366f1;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#0b1120; --fg:#e2e8f0; --muted:#94a3b8; --panel:#111a2e; --border:#1e293b; --accent:#818cf8;
    }}
  }}
  :root[data-theme="dark"] {{ --bg:#0b1120; --fg:#e2e8f0; --muted:#94a3b8; --panel:#111a2e; --border:#1e293b; --accent:#818cf8; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:16px; background:var(--bg); color:var(--fg);
          font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  h1 {{ font-size:17px; margin:0 0 2px; }}
  p.sub {{ color:var(--muted); margin:0 0 14px; }}
  .stage {{ display:flex; justify-content:center; }}
  canvas {{ image-rendering:pixelated; width:min(100%,460px); height:auto; aspect-ratio:1;
            border:1px solid var(--border); border-radius:10px; background:var(--panel); }}
  .controls {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin:14px 0 4px; }}
  button {{ background:var(--accent); color:#fff; border:0; border-radius:8px; padding:7px 16px;
            font-weight:600; cursor:pointer; }}
  input[type=range] {{ flex:1; min-width:140px; accent-color:var(--accent); }}
  .mcs {{ font-variant-numeric:tabular-nums; color:var(--muted); min-width:96px; }}
  .legend {{ display:flex; gap:14px; flex-wrap:wrap; color:var(--muted); font-size:12px; margin-top:8px; }}
  .legend i {{ display:inline-block; width:11px; height:11px; border-radius:2px; margin-right:5px; vertical-align:-1px; }}
  .tog {{ color:var(--muted); font-size:12px; user-select:none; }}
</style></head>
<body>
  <h1>{title}</h1>
  <p class="sub">{subtitle}</p>
  <div class="stage"><canvas id="cv"></canvas></div>
  <div class="controls">
    <button id="play">Play</button>
    <input type="range" id="slider" min="0" value="0" step="1">
    <span class="mcs" id="mcs">MCS 0</span>
    {conc_ui}
  </div>
  <div class="legend">
    <span><i style="background:{TYPE_COLORS[1]}"></i>Cell type A</span>
    <span><i style="background:{TYPE_COLORS[2]}"></i>Cell type B</span>
    <span><i style="background:{TYPE_COLORS[0]};border:1px solid var(--border)"></i>Medium</span>
  </div>
<script>
const D = {data};
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const [W,H] = D.dim; cv.width = W; cv.height = H;
const slider = document.getElementById('slider'); slider.max = D.frames.length - 1;
const mcsEl = document.getElementById('mcs'), playBtn = document.getElementById('play');
const concCb = document.getElementById('conc');

function hex(c){{const n=parseInt(c.slice(1),16);return [n>>16&255,n>>8&255,n&255];}}
const PAL = D.colors.map(hex);
function unb64(s){{ const bin = atob(s); const a = new Uint8Array(bin.length);
  for(let k=0;k<bin.length;k++) a[k]=bin.charCodeAt(k); return a; }}
const CACHE = {{}};
function frame(i){{
  if (!CACHE[i]){{ const f=D.frames[i]; CACHE[i]={{mcs:f.mcs, t:unb64(f.t), c:f.c?unb64(f.c):null}}; }}
  return CACHE[i];
}}

function draw(i){{
  const f = frame(i), tf = f.t, cf = f.c;
  const showConc = cf && concCb && concCb.checked;
  const img = ctx.createImageData(W,H);
  for(let p=0;p<W*H;p++){{
    const t = tf[p]; let [r,g,b] = PAL[t] || PAL[0];
    if (t===0 && showConc){{
      const v = cf[p]/255;
      r = 238-(238-99)*v; g = 242-(242-102)*v; b = 247-(247-241)*v;  // blend medium->accent
    }}
    const o = p*4; img.data[o]=r; img.data[o+1]=g; img.data[o+2]=b; img.data[o+3]=255;
  }}
  ctx.putImageData(img,0,0);
  mcsEl.textContent = 'MCS ' + f.mcs.toLocaleString();
  slider.value = i;
}}
let cur=0, timer=null;
slider.oninput = e => {{ cur=+e.target.value; draw(cur); }};
if (concCb) concCb.onchange = () => draw(cur);
function stop(){{ clearInterval(timer); timer=null; playBtn.textContent='Play'; }}
playBtn.onclick = () => {{
  if (timer){{ stop(); return; }}
  playBtn.textContent='Pause';
  timer = setInterval(() => {{ cur=(cur+1)%D.frames.length; draw(cur); if(cur===D.frames.length-1){{stop();}} }}, 220);
}};
draw(0);
</script>
</body></html>
"""


# ── static charts (committed PNG + meta.json) ───────────────────────────────

def _save(fig, path: Path, title: str, caption: str, interpretation: str):
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    path.with_suffix(".meta.json").write_text(json.dumps(
        {"title": title, "caption": caption, "interpretation": interpretation}, indent=2))


def render_charts(cfg: dict, snapshots: list[dict], outdir: Path):
    mcs = [s["mcs"] for s in snapshots]
    n_cells = [s["n_cells"] for s in snapshots]
    t1 = [s["type_1_count"] for s in snapshots]
    t2 = [s["type_2_count"] for s in snapshots]
    vol = [s["avg_volume"] for s in snapshots]
    surf = [s["avg_surface"] for s in snapshots]
    title = cfg["title"]

    # 00 — population
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(mcs, n_cells, color="#6366f1", lw=2.2, marker="o", ms=3)
    ax.set_xlabel("Monte-Carlo step (MCS)"); ax.set_ylabel("cell count")
    ax.set_title(f"{title}: population", fontsize=11)
    ax.grid(alpha=0.25)
    _save(fig, outdir / "00_population.png", f"{title} — population over time",
          "Number of cells on the lattice across the simulation.",
          f"Cell count goes {n_cells[0]} → {n_cells[-1]} over {mcs[-1]:,} MCS.")

    # 01 — composition (two cell types)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(mcs, t1, color="#22c55e", lw=2.2, label="type A")
    ax.plot(mcs, t2, color="#f97316", lw=2.2, label="type B")
    ax.set_xlabel("Monte-Carlo step (MCS)"); ax.set_ylabel("cell count")
    ax.set_title(f"{title}: composition", fontsize=11); ax.legend(); ax.grid(alpha=0.25)
    _save(fig, outdir / "01_composition.png", f"{title} — cell-type composition",
          "Per-type cell counts (type A vs type B) over time.",
          f"Final composition: {t1[-1]} type-A, {t2[-1]} type-B.")

    # 02 — morphometry
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(mcs, vol, color="#0ea5e9", lw=2.2, label="mean volume (px)")
    ax.plot(mcs, surf, color="#ec4899", lw=2.2, label="mean surface (px)")
    ax.set_xlabel("Monte-Carlo step (MCS)"); ax.set_ylabel("pixels")
    ax.set_title(f"{title}: morphometry", fontsize=11); ax.legend(); ax.grid(alpha=0.25)
    _save(fig, outdir / "02_morphometry.png", f"{title} — mean cell morphometry",
          "Mean per-cell volume and surface across the population.",
          f"Mean volume ends at {vol[-1]:.1f} px; mean surface {surf[-1]:.1f} px.")


def write_metrics(snapshots: list[dict], path: Path):
    cols = ["mcs", "n_cells", "type_1_count", "type_2_count", "avg_volume", "avg_surface"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for s in snapshots:
            w.writerow({k: s[k] for k in cols})


def main(scenario_id: str):
    if scenario_id not in SLUGS:
        raise SystemExit(f"unknown scenario '{scenario_id}'; choose from {list(SLUGS)}")
    cfg = next(c for c in demo.CONFIGS if c["id"] == scenario_id)
    slug = SLUGS[scenario_id]
    outdir = ROOT / "workspace" / "studies" / slug
    (outdir / "viz").mkdir(parents=True, exist_ok=True)
    (outdir / "charts").mkdir(parents=True, exist_ok=True)

    t0 = _time.perf_counter()
    print(f"[{slug}] running cc3d: {cfg['title']} ({cfg['total_mcs']:,} MCS)...", flush=True)
    snapshots, runtime = demo.run_simulation(cfg)

    (outdir / "viz" / "scene.html").write_text(render_scene_html(cfg, snapshots, runtime))
    render_charts(cfg, snapshots, outdir / "charts")
    write_metrics(snapshots, outdir / "metrics.csv")
    (outdir / "run.log").write_text(
        f"scenario: {scenario_id}\ntitle: {cfg['title']}\n"
        f"lattice: {cfg['config']['dim_x']}x{cfg['config']['dim_y']}\n"
        f"total_mcs: {cfg['total_mcs']}\nsnapshots: {len(snapshots)}\n"
        f"cells: {snapshots[0]['n_cells']} -> {snapshots[-1]['n_cells']}\n"
        f"cc3d_runtime_s: {runtime:.1f}\nwall_s: {_time.perf_counter()-t0:.1f}\n"
        f"cc3d_version: 4.6.0\n"
    )
    print(f"[{slug}] done in {_time.perf_counter()-t0:.1f}s -> {outdir}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sorting")
