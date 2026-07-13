async function loadJson(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  return res.json();
}

function layerRow(name, op, drifted) {
  return `
    <div class="layer-row ${drifted ? "bad" : "ok"}">
      <span class="layer-name">${name}</span>
      <span class="layer-op">${op}</span>
      <span class="layer-status">${drifted ? "Drift" : "OK"}</span>
    </div>`;
}

function renderDemo(reg, summary, animate) {
  const failed = summary.tensors_failed ?? 0;
  const total = summary.tensors_compared ?? 0;
  const blocked = summary.status === "fail" || failed > 0;
  const ff = reg.first_failure;

  document.getElementById("metric-layers").textContent = String(total);
  document.getElementById("status-dot").className = `status-dot ${blocked ? "bad" : "good"}`;
  document.getElementById("demo-status").textContent = blocked
    ? "Deploy check complete — action required"
    : "Deploy check complete — cleared";

  const verdict = document.getElementById("verdict-card");
  verdict.className = `verdict-card ${blocked ? "blocked" : "approved"}`;
  document.getElementById("verdict-label").textContent = blocked
    ? "SHIP BLOCKED"
    : "CLEARED TO SHIP";
  document.getElementById("verdict-detail").textContent = blocked
    ? `Drift started at ${ff?.producer_node || "unknown layer"}. This build cannot go to fleet until fixed.`
    : "All layers within tolerance. Receipt ready to sign.";

  document.getElementById("demo-stats").innerHTML = `
    <div class="mini-stat"><span>Layers checked</span><strong>${total}</strong></div>
    <div class="mini-stat"><span>Drift detected</span><strong>${failed}</strong></div>
    <div class="mini-stat"><span>Output check</span><strong>${summary.parity_passed}/${summary.parity_total}</strong></div>
  `;

  const layers = (reg.divergences || reg.graph_layout || []).slice(0, 6);
  const container = document.getElementById("demo-layers");
  container.innerHTML = "";
  container.style.opacity = animate ? "0.4" : "1";

  const items = layers.map((d) => {
    const name = d.producer_node || d.node;
    const op = d.producer_op || d.op;
    const drifted = d.passed === false || (d.max_abs_diff || 0) > (reg.atol || 1e-5);
    return { name, op, drifted };
  });

  if (animate) {
    items.forEach((item, i) => {
      setTimeout(() => {
        container.insertAdjacentHTML("beforeend", layerRow(item.name, item.op, item.drifted));
        if (i === items.length - 1) container.style.opacity = "1";
      }, i * 120);
    });
  } else {
    container.innerHTML = items.map((item) => layerRow(item.name, item.op, item.drifted)).join("");
  }

  document.getElementById("demo-foot").textContent = blocked
    ? "In production, CI exits here. No OTA. Engineer applies fix rules and re-runs."
    : "Receipt signed. Safe to promote to staging and fleet.";
}

async function init() {
  const [reg, summary] = await Promise.all([
    loadJson("data/scenarios/branch/regression_report.json"),
    loadJson("data/scenarios/branch/summary.json"),
  ]);

  renderDemo(reg, summary, false);

  document.getElementById("run-check").addEventListener("click", () => {
    document.getElementById("demo-layers").innerHTML = "";
    renderDemo(reg, summary, true);
  });
}

init().catch(() => {
  document.getElementById("demo-panel").innerHTML =
    '<p class="demo-error">Demo data missing. Run <code>bash scripts/sync-demo.sh</code>.</p>';
});
