function heatColor(diff, failed, maxDiff, atol) {
  if (!failed && diff <= atol) return "#1a3d2e";
  const t = Math.min(1, diff / Math.max(maxDiff, atol, 1e-8));
  const r = Math.round(160 + 90 * t);
  const g = Math.round(50 - 30 * t);
  const b = Math.round(45 - 15 * t);
  return `rgb(${r},${g},${b})`;
}

function renderGraph(svg, layout, maxDiff, atol) {
  const cellW = 120;
  const cellH = 48;
  const gap = 10;
  const offsetX = 24;
  const offsetY = 16;

  svg.innerHTML = layout
    .map((n) => {
      const x = offsetX + n.col * (cellW + gap);
      const y = offsetY + n.row * (cellH + gap);
      const diff = n.max_abs_diff || 0;
      const fill = heatColor(diff, n.failed, maxDiff, atol);
      const label = n.node.length > 14 ? n.node.slice(0, 13) + "…" : n.node;
      return `
        <g class="graph-node">
          <rect x="${x}" y="${y}" width="${cellW}" height="${cellH}" rx="6" fill="${fill}" stroke="#2a3a48"/>
          <text x="${x + 7}" y="${y + 16}" fill="#f0f4f8" font-size="10" font-weight="600">${label}</text>
          <text x="${x + 7}" y="${y + 30}" fill="#a8b8c8" font-size="9">${n.op}</text>
          <text x="${x + 7}" y="${y + 42}" fill="#fff" font-size="8" opacity="0.9">Δ ${diff.toExponential(1)}</text>
        </g>`;
    })
    .join("");

  const rows = Math.max(...layout.map((n) => n.row), 0) + 1;
  svg.setAttribute("viewBox", `0 0 580 ${offsetY * 2 + rows * (cellH + gap)}`);
}

async function init() {
  const res = await fetch("data/regression_report.json");
  if (!res.ok) return;
  const data = await res.json();

  const maxDiff = Math.max(...data.divergences.map((d) => d.max_abs_diff), data.atol);

  document.getElementById("scan-stats").innerHTML = `
    <div class="stat-box"><span>Failed tensors</span><strong>${data.tensors_failed}</strong></div>
    <div class="stat-box"><span>Compared</span><strong>${data.tensors_compared}</strong></div>
    <div class="stat-box"><span>Breakers</span><strong>${data.breaker_recommendations.length}</strong></div>
  `;

  const ff = data.first_failure;
  document.getElementById("first-failure").innerHTML = ff
    ? `<strong>First topo failure:</strong> <code>${ff.producer_node}</code> (${ff.producer_op}) · max Δ ${ff.max_abs_diff.toExponential(2)} — insert FP32 segment breaker here.`
    : "All tensors within tolerance.";

  renderGraph(document.getElementById("graph-svg"), data.graph_layout, maxDiff, data.atol);

  const panel = document.getElementById("breaker-panel");
  if (data.breaker_recommendations.length) {
    panel.innerHTML =
      "<h4>Recommended segment breakers</h4>" +
      data.breaker_recommendations
        .slice(0, 2)
        .map(
          (b) => `
        <div class="breaker-card">
          <strong>${b.node_name}</strong> (${b.op_type})<br>
          <span style="color:#8fa3b8">force_fp32 + break_before + ORT backend</span>
        </div>`
        )
        .join("");
  }
}

init();
