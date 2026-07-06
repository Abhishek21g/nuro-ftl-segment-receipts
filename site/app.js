async function loadReceipt() {
  const res = await fetch("data/receipt.json");
  if (!res.ok) return;
  const data = await res.json();

  const grid = document.getElementById("island-preview");
  const metrics = document.getElementById("metrics-preview");
  if (!grid || !metrics) return;

  grid.innerHTML = data.segments
    .map(
      (seg) => `
      <div class="island-card">
        <h4>Island ${seg.id}: ${seg.backend} / ${seg.dtype}</h4>
        <p>${seg.node_names.length} nodes · ${seg.estimated_ms.toFixed(2)} ms · ${seg.reason}</p>
      </div>`
    )
    .join("");

  const parityPass = data.parity.filter((p) => p.passed).length;
  metrics.innerHTML = `
    <div class="metric"><span>Segments</span><strong>${data.segment_count}</strong></div>
    <div class="metric"><span>Latency budget</span><strong>${data.latency_budget_ms.toFixed(2)} ms</strong></div>
    <div class="metric"><span>Parity</span><strong>${parityPass}/${data.parity.length}</strong></div>
  `;

  const toolbar = document.querySelector(".preview-toolbar");
  if (toolbar) {
    toolbar.textContent = `${data.model_path.split("/").pop()} → ${data.segment_count} compiler islands`;
  }
}

loadReceipt();
