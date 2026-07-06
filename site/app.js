function heatColor(diff, failed, maxDiff, atol) {
  if (!failed && diff <= atol) return "#1a3d2e";
  const t = Math.min(1, diff / Math.max(maxDiff, atol, 1e-8));
  const r = Math.round(160 + 90 * t);
  const g = Math.round(50 - 30 * t);
  const b = Math.round(45 - 15 * t);
  return `rgb(${r},${g},${b})`;
}

function renderGraph(svg, layout, maxDiff, atol) {
  const cellW = 118;
  const cellH = 46;
  const gap = 10;
  const offsetX = 20;
  const offsetY = 14;

  svg.innerHTML = layout
    .map((n) => {
      const x = offsetX + n.col * (cellW + gap);
      const y = offsetY + n.row * (cellH + gap);
      const diff = n.max_abs_diff || 0;
      const fill = heatColor(diff, n.failed, maxDiff, atol);
      const label = n.node.length > 13 ? n.node.slice(0, 12) + "…" : n.node;
      return `
        <g class="graph-node">
          <rect x="${x}" y="${y}" width="${cellW}" height="${cellH}" rx="6" fill="${fill}" stroke="#2a3a48"/>
          <text x="${x + 6}" y="${y + 15}" fill="#f0f4f8" font-size="10" font-weight="600">${label}</text>
          <text x="${x + 6}" y="${y + 29}" fill="#a8b8c8" font-size="9">${n.op}</text>
          <text x="${x + 6}" y="${y + 41}" fill="#fff" font-size="8" opacity="0.9">Δ ${diff.toExponential(1)}</text>
        </g>`;
    })
    .join("");

  const rows = Math.max(...layout.map((n) => n.row), 0) + 1;
  svg.setAttribute("viewBox", `0 0 560 ${offsetY * 2 + rows * (cellH + gap)}`);
}

async function loadJson(path) {
  const res = await fetch(path);
  return res.ok ? res.json() : null;
}

async function init() {
  const [reg, summary, doctor, manifest] = await Promise.all([
    loadJson("data/regression_report.json"),
    loadJson("data/summary.json"),
    loadJson("data/doctor.json"),
    loadJson("data/manifest.json"),
  ]);

  if (manifest) {
    const model = manifest.model?.split("/").pop() || "branch.onnx";
    document.getElementById("run-label").textContent =
      `Run ${manifest.run_id} · ${model} · ${manifest.candidate}`;
  }

  if (summary) {
    document.getElementById("scan-stats").innerHTML = `
      <div class="stat-box"><span>Failed</span><strong>${summary.tensors_failed}</strong></div>
      <div class="stat-box"><span>Compared</span><strong>${summary.tensors_compared}</strong></div>
      <div class="stat-box"><span>Segments</span><strong>${summary.segment_count}</strong></div>
    `;
    const ds = document.getElementById("doctor-status");
    ds.textContent = `doctor: ${summary.status.toUpperCase()} · parity ${summary.parity_passed}/${summary.parity_total}`;
    ds.className = `doctor-status ${summary.status}`;
  }

  if (reg) {
    const maxDiff = Math.max(...reg.divergences.map((d) => d.max_abs_diff), reg.atol);
    const ff = reg.first_failure;
    document.getElementById("first-failure").innerHTML = ff
      ? `<strong>First topo failure:</strong> <code>${ff.producer_node}</code> (${ff.producer_op}) · Δ ${ff.max_abs_diff.toExponential(2)}`
      : "All tensors within tolerance.";

    renderGraph(document.getElementById("graph-svg"), reg.graph_layout, maxDiff, reg.atol);

    const panel = document.getElementById("breaker-panel");
    const recs = reg.breaker_recommendations || [];
    if (recs.length) {
      panel.innerHTML =
        "<h4>Recommended breakers</h4>" +
        recs
          .slice(0, 3)
          .map(
            (b) => `
          <div class="breaker-card">
            <strong>${b.node_name}</strong> (${b.op_type})<br>
            <span style="color:#8fa3b8">break_before · force_fp32 · ORT</span>
          </div>`
          )
          .join("");
    }
  }

  if (doctor?.findings) {
    document.getElementById("findings-list").innerHTML = doctor.findings
      .map(
        (f) => `
      <li class="sev-${f.severity}">
        <strong>[${f.code}]</strong> ${f.message}
        <br><small style="color:#8fa3b8">→ ${f.suggestion}</small>
      </li>`
      )
      .join("");
  }
}

init();
