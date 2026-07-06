from __future__ import annotations

import html
import json
from pathlib import Path


def write_regression_html(report_path: Path, output_path: Path | None = None) -> Path:
    data = json.loads(report_path.read_text())
    out = output_path or report_path.with_name("regression_report.html")

    max_diff = max((d["max_abs_diff"] for d in data["divergences"]), default=1.0) or 1.0

    def heat_color(diff: float, failed: bool) -> str:
        if not failed and diff < data["atol"]:
            return "#1a3d2e"
        t = min(1.0, diff / max(max_diff, data["atol"]))
        r = int(180 + 75 * t)
        g = int(60 - 40 * t)
        b = int(50 - 20 * t)
        return f"rgb({r},{g},{b})"

    nodes_svg = ""
    cell_w, cell_h, gap = 130, 52, 12
    for n in data.get("graph_layout", []):
        x = 40 + n["col"] * (cell_w + gap)
        y = 80 + n["row"] * (cell_h + gap)
        color = heat_color(n.get("max_abs_diff", 0), n.get("failed", False))
        diff = n.get("max_abs_diff", 0)
        nodes_svg += f"""
        <g class="graph-node" data-node="{html.escape(n['node'])}">
          <rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" rx="8" fill="{color}" stroke="#3a4a58" stroke-width="1"/>
          <text x="{x + 8}" y="{y + 18}" fill="#e8edf2" font-size="11" font-weight="600">{html.escape(n['node'][:16])}</text>
          <text x="{x + 8}" y="{y + 34}" fill="#b8c5d0" font-size="10">{html.escape(n['op'])}</text>
          <text x="{x + 8}" y="{y + 46}" fill="#fff" font-size="9" opacity="0.85">Δ {diff:.2e}</text>
        </g>"""

    rows = max((n["row"] for n in data.get("graph_layout", [])), default=0) + 1
    svg_h = 80 + rows * (cell_h + gap) + 40

    table_rows = ""
    for d in data.get("divergences", [])[:12]:
        status = "FAIL" if not d["passed"] else "ok"
        table_rows += f"""
        <tr class="{status}">
          <td><code>{html.escape(d['producer_node'])}</code></td>
          <td>{html.escape(d['producer_op'])}</td>
          <td><code>{html.escape(d['tensor_name'])}</code></td>
          <td>{d['max_abs_diff']:.2e}</td>
          <td>{d['relative_l2']:.2e}</td>
          <td>{status}</td>
        </tr>"""

    breaker = data.get("breaker_recommendations", [])
    breaker_html = ""
    for b in breaker:
        breaker_html += f"""
        <div class="breaker-card">
          <h4>{html.escape(b['node_name'])} <span>({html.escape(b['op_type'])})</span></h4>
          <p>{html.escape(b['reason'])}</p>
          <pre>{html.escape(json.dumps(b['suggested_rule'], indent=2))}</pre>
        </div>"""

    ff = data.get("first_failure")
    ff_html = "No divergence above tolerance."
    if ff:
        ff_html = (
            f"First failure at <code>{html.escape(ff['producer_node'])}</code> "
            f"({html.escape(ff['producer_op'])}) — max Δ {ff['max_abs_diff']:.2e}"
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Regression Report — {html.escape(Path(data['model_path']).name)}</title>
  <style>
    :root {{ --bg:#0a0e12; --panel:#121a22; --text:#e8edf2; --muted:#8b9cb0; --accent:#e85d4c; --ok:#3d9a6a; }}
    body {{ margin:0; font-family:system-ui,sans-serif; background:var(--bg); color:var(--text); line-height:1.5; }}
    main {{ max-width:1200px; margin:0 auto; padding:2rem 1.25rem 4rem; }}
    h1 {{ font-size:1.6rem; margin-bottom:0.25rem; }}
    .problem {{ background:var(--panel); border-left:4px solid var(--accent); padding:1rem 1.25rem; margin:1.5rem 0; border-radius:0 8px 8px 0; color:var(--muted); }}
    .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:0.75rem; margin:1.5rem 0; }}
    .stat {{ background:var(--panel); padding:0.9rem; border-radius:8px; }}
    .stat strong {{ display:block; font-size:1.4rem; color:var(--accent); }}
    .first-fail {{ background:#2a1818; border:1px solid #5a3030; padding:0.75rem 1rem; border-radius:8px; margin-bottom:1.5rem; }}
    svg {{ width:100%; background:var(--panel); border-radius:12px; margin:1rem 0; }}
    table {{ width:100%; border-collapse:collapse; font-size:0.88rem; }}
    th,td {{ text-align:left; padding:0.5rem; border-bottom:1px solid #2a3540; }}
    th {{ color:var(--muted); }}
    tr.FAIL td {{ color:#f0a090; }}
    tr.ok td:last-child {{ color:var(--ok); }}
    .breaker-card {{ background:var(--panel); border-radius:8px; padding:1rem; margin-bottom:0.75rem; border:1px solid #2a3540; }}
    .breaker-card h4 span {{ color:var(--muted); font-weight:400; }}
    pre {{ background:#0a0e12; padding:0.75rem; border-radius:6px; overflow-x:auto; font-size:0.8rem; }}
    .disclaimer {{ color:var(--muted); font-size:0.85rem; margin-top:2rem; }}
  </style>
</head>
<body>
  <main>
    <h1>Silent Regression Report</h1>
    <p class="meta">{html.escape(data['reference_path'])} vs {html.escape(data['candidate_path'])} · rtol={data['rtol']} atol={data['atol']}</p>
    <div class="problem">{html.escape(data['problem'])}</div>
    <div class="stats">
      <div class="stat"><span>Tensors</span><strong>{data['tensors_compared']}</strong></div>
      <div class="stat"><span>Failed</span><strong>{data['tensors_failed']}</strong></div>
      <div class="stat"><span>Breakers</span><strong>{len(breaker)}</strong></div>
    </div>
    <div class="first-fail"><strong>First divergence in topo order:</strong> {ff_html}</div>
    <h2>Graph heatmap</h2>
    <p style="color:var(--muted);font-size:0.9rem">Darker green = stable. Red = activation diverged between reference and candidate compile path.</p>
    <svg viewBox="0 0 600 {svg_h}" role="img" aria-label="Graph divergence heatmap">{nodes_svg}</svg>
    <h2>Top divergences</h2>
    <table>
      <thead><tr><th>Node</th><th>Op</th><th>Tensor</th><th>Max Δ</th><th>Rel L2</th><th>Status</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
    <h2>Recommended segment breakers (FTL-style)</h2>
    {breaker_html or '<p>No breakers needed — within tolerance.</p>'}
    <p class="disclaimer">Third-party tool inspired by Nuro's published FTL architecture. Not affiliated with Nuro, Inc.</p>
  </main>
</body>
</html>"""
    out.write_text(page)
    return out
