from __future__ import annotations

import html
import json
from pathlib import Path


def write_html_report(receipt_path: Path, output_path: Path | None = None) -> Path:
    data = json.loads(receipt_path.read_text())
    out = output_path or receipt_path.with_suffix(".html")

    segments_rows = ""
    for seg in data["segments"]:
        ops = ", ".join(seg.get("node_names", seg.get("nodes", []))[:3])
        segments_rows += f"""
        <tr>
          <td>{seg['id']}</td>
          <td><code>{seg['backend']}</code></td>
          <td>{seg['dtype']}</td>
          <td>{len(seg.get('node_names', seg.get('nodes', [])))}</td>
          <td>{seg.get('estimated_ms', 0):.3f} ms</td>
          <td>{html.escape(seg.get('reason', ''))}</td>
          <td><small>{html.escape(ops)}</small></td>
        </tr>"""

    parity_rows = ""
    for p in data.get("parity", []):
        status = "pass" if p["passed"] else "fail"
        parity_rows += f"""
        <tr class="{status}">
          <td>{p['segment_id']}</td>
          <td><code>{html.escape(p['output_name'])}</code></td>
          <td>{p['max_abs_diff']:.2e}</td>
          <td>{p['mean_abs_diff']:.2e}</td>
          <td>{status}</td>
        </tr>"""

    early_rows = ""
    for e in data.get("early_publish", []):
        early_rows += f"""
        <tr>
          <td>{e['priority']}</td>
          <td><code>{html.escape(e['output_name'])}</code></td>
          <td>seg {e['producer_segment_id']}</td>
          <td>{e['estimated_ready_ms']:.3f} ms</td>
        </tr>"""

    gpu_rows = ""
    for g in data.get("gpu_copies", []):
        gpu_rows += f"""
        <tr>
          <td>after seg {g['after_segment_id']}</td>
          <td>GPU {g['from_gpu']} → GPU {g['to_gpu']}</td>
          <td><code>{html.escape(g['tensor_name'])}</code></td>
        </tr>"""

    stitch = " → ".join(str(s) for s in data.get("stitch_order", []))

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Segment Receipt — {html.escape(Path(data['model_path']).name)}</title>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #1a2229;
      --text: #e8edf2;
      --muted: #8b9aab;
      --accent: #3d9a6a;
      --warn: #c97b3d;
      --fail: #c94d4d;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }}
    h1 {{ font-size: 1.75rem; margin-bottom: 0.25rem; }}
    .meta {{ color: var(--muted); margin-bottom: 2rem; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
    .card {{ background: var(--panel); border-radius: 10px; padding: 1rem; }}
    .card strong {{ display: block; font-size: 1.5rem; color: var(--accent); }}
    section {{ margin-bottom: 2rem; }}
    h2 {{ font-size: 1.1rem; margin-bottom: 0.75rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ text-align: left; padding: 0.5rem 0.6rem; border-bottom: 1px solid #2a3540; }}
    th {{ color: var(--muted); font-weight: 600; }}
    tr.fail td {{ color: var(--fail); }}
    tr.pass td:last-child {{ color: var(--accent); }}
    code {{ font-size: 0.85em; }}
    .disclaimer {{ color: var(--muted); font-size: 0.85rem; border-top: 1px solid #2a3540; padding-top: 1.5rem; }}
    .stitch {{ background: var(--panel); padding: 0.75rem 1rem; border-radius: 8px; font-family: monospace; }}
    .back-link {{ display:inline-block; margin-bottom:1rem; color:#cfe8d8; text-decoration:none; border:1px solid #2a3540; padding:0.4rem 0.7rem; border-radius:8px; font-size:0.9rem; }}
    .back-link:hover {{ border-color: var(--accent); color:#fff; }}
  </style>
</head>
<body>
  <main>
    <a class="back-link" href="../index.html">← Back to home</a>
    <h1>FTL Segment Receipt</h1>
    <p class="meta">Model: <code>{html.escape(data['model_path'])}</code><br>
    Rules: <code>{html.escape(data['rules_path'])}</code></p>

    <div class="cards">
      <div class="card"><span>Nodes</span><strong>{data['node_count']}</strong></div>
      <div class="card"><span>Segments</span><strong>{data['segment_count']}</strong></div>
      <div class="card"><span>Latency budget</span><strong>{data['latency_budget_ms']:.2f} ms</strong></div>
      <div class="card"><span>Parity checks</span><strong>{sum(1 for p in data.get('parity',[]) if p['passed'])}/{len(data.get('parity',[]))}</strong></div>
    </div>

    <section>
      <h2>Stitch order</h2>
      <div class="stitch">{stitch or '—'}</div>
    </section>

    <section>
      <h2>Compiler islands</h2>
      <table>
        <thead><tr><th>ID</th><th>Backend</th><th>Dtype</th><th>Nodes</th><th>Est.</th><th>Reason</th><th>Sample</th></tr></thead>
        <tbody>{segments_rows}</tbody>
      </table>
    </section>

    <section>
      <h2>Numerical parity</h2>
      <table>
        <thead><tr><th>Seg</th><th>Output</th><th>Max Δ</th><th>Mean Δ</th><th>Status</th></tr></thead>
        <tbody>{parity_rows or '<tr><td colspan="5">No parity data</td></tr>'}</tbody>
      </table>
    </section>

    <section>
      <h2>Early publish priority</h2>
      <table>
        <thead><tr><th>Priority</th><th>Output</th><th>Producer</th><th>Ready</th></tr></thead>
        <tbody>{early_rows or '<tr><td colspan="4">No marked outputs</td></tr>'}</tbody>
      </table>
    </section>

    <section>
      <h2>Multi-GPU copy points</h2>
      <table>
        <thead><tr><th>Location</th><th>Copy</th><th>Tensor</th></tr></thead>
        <tbody>{gpu_rows or '<tr><td colspan="3">Single-GPU plan</td></tr>'}</tbody>
      </table>
    </section>

    <p class="disclaimer">{html.escape(data.get('disclaimer', ''))}</p>
  </main>
</body>
</html>"""

    out.write_text(page)
    return out


def write_markdown_report(run_dir: Path) -> Path:
    """Gold artifact: human-readable report.md in run directory."""
    run_dir = Path(run_dir)
    lines: list[str] = ["# FTL Segment Receipt — Run Report", ""]

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        lines.append(f"- **Run ID:** `{manifest.get('run_id', run_dir.name)}`")
        lines.append(f"- **Model:** `{manifest.get('model', '?')}`")
        lines.append(f"- **Rules:** `{manifest.get('rules', '?')}`")
        lines.append(f"- **Candidate path:** `{manifest.get('candidate', '?')}`")
        lines.append("")

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        lines.append("## Summary")
        lines.append(f"- Doctor status: **{s.get('status', '?')}**")
        lines.append(f"- Regression: {s.get('tensors_failed', '?')}/{s.get('tensors_compared', '?')} tensors failed")
        lines.append(f"- Segments: {s.get('segment_count', '?')}")
        lines.append(f"- Parity: {s.get('parity_passed', '?')}/{s.get('parity_total', '?')} passed")
        if s.get("first_failure_node"):
            lines.append(f"- First failure node: `{s['first_failure_node']}`")
        lines.append("")

    reg_path = run_dir / "regression_report.json"
    if reg_path.exists():
        reg = json.loads(reg_path.read_text())
        lines.append("## Silent regression scan")
        lines.append(reg.get("problem", ""))
        lines.append("")
        recs = reg.get("breaker_recommendations", [])
        if recs:
            lines.append("### Recommended segment breakers")
            for r in recs[:3]:
                lines.append(f"- `{r['node_name']}` ({r['op_type']}): max Δ {r['max_abs_diff']:.2e}")
            lines.append("")

    receipt_path = run_dir / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        lines.append("## Compiler islands")
        for seg in receipt.get("segments", []):
            nodes = seg.get("node_names", [])
            lines.append(
                f"- Island {seg['id']}: {seg['backend']}/{seg['dtype']} "
                f"({len(nodes)} nodes, {seg.get('estimated_ms', 0):.3f} ms)"
            )
        lines.append("")

    lines.append("---")
    lines.append("*Third-party tool inspired by Nuro's published FTL architecture. Not affiliated with Nuro, Inc.*")

    out = run_dir / "report.md"
    out.write_text("\n".join(lines))
    return out

