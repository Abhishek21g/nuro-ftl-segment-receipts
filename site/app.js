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
  const mode = data.parity_mode || "synthetic";
  metrics.innerHTML = `
    <div class="metric"><span>Segments</span><strong>${data.segment_count}</strong></div>
    <div class="metric"><span>Latency budget</span><strong>${data.latency_budget_ms.toFixed(2)} ms</strong></div>
    <div class="metric"><span>Parity (${mode})</span><strong>${parityPass}/${data.parity.length}</strong></div>
  `;

  const toolbar = document.querySelector(".preview-toolbar");
  if (toolbar) {
    const model = data.model_path.split("/").pop();
    toolbar.textContent = `${model} → ${data.segment_count} compiler islands · ${mode} parity`;
  }
}

function renderMergeStack(el, segments) {
  el.innerHTML = segments
    .map(
      (s) => `
      <div class="merge-chip">
        <strong>seg ${s.id}</strong> ${s.backend}/${s.dtype}
        · ${s.node_count} nodes · ${s.estimated_ms.toFixed(2)} ms
      </div>`
    )
    .join("");
}

async function loadMergeDiff() {
  const res = await fetch("data/merge_diff.json");
  if (!res.ok) return;
  const data = await res.json();
  const before = document.getElementById("merge-before");
  const after = document.getElementById("merge-after");
  const rec = document.getElementById("merge-rec");
  if (!before || !after || !rec) return;

  renderMergeStack(before, data.before.segments);
  renderMergeStack(after, data.after.segments);
  rec.textContent = data.recommendation;
}

loadReceipt();
loadMergeDiff();
