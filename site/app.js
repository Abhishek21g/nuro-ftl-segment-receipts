const state = {
  catalog: null,
  scenarios: {},
  mode: "standard",
  runnerId: null,
  activeNode: null,
  scanning: false,
};

async function loadJson(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed ${path}`);
  return res.json();
}

async function loadScenarioData(id, prefix) {
  const [regression, summary, doctor] = await Promise.all([
    loadJson(`${prefix}/regression_report.json`),
    loadJson(`${prefix}/summary.json`),
    loadJson(`${prefix}/doctor.json`).catch(() => ({ findings: [] })),
  ]);
  return { regression, summary, doctor };
}

function getMode() {
  return state.catalog.tolerance_modes.find((m) => m.id === state.mode) || state.catalog.tolerance_modes[1];
}

function passStats(regression, atol) {
  const divs = regression.divergences || [];
  const passed = divs.filter((d) => (d.max_abs_diff || 0) <= atol).length;
  const total = divs.length || regression.tensors_compared || 0;
  const failRate = total ? ((total - passed) / total) * 100 : 0;
  const passRate = total ? (passed / total) * 100 : 100;
  return { passed, total, failRate, passRate };
}

function computeDashboardMetrics() {
  const mode = getMode();
  const scenarios = state.catalog.scenarios;
  const barItems = scenarios.map((s) => {
    const data = state.scenarios[s.id];
    const stats = passStats(data.regression, mode.atol);
    return { label: s.title, value: stats.passRate };
  });

  const xLabels = state.catalog.tolerance_modes.map((m) => m.label);
  const lineSeries = scenarios.map((s) => {
    const data = state.scenarios[s.id];
    return {
      label: s.title.split(" ")[0],
      values: state.catalog.tolerance_modes.map((m) =>
        passStats(data.regression, m.atol).passRate
      ),
    };
  });

  const heatRows = scenarios.map((s) => s.title);
  const heatCols = state.catalog.tolerance_modes.map((m) => m.label);
  const heatMatrix = scenarios.map((s) => {
    const data = state.scenarios[s.id];
    return state.catalog.tolerance_modes.map((m) =>
      passStats(data.regression, m.atol).failRate
    );
  });

  const difficulty = scenarios
    .map((s) => {
      const data = state.scenarios[s.id];
      const avgFail =
        state.catalog.tolerance_modes.reduce(
          (acc, m) => acc + passStats(data.regression, m.atol).failRate,
          0
        ) / state.catalog.tolerance_modes.length;
      return { label: s.title, failRate: avgFail };
    })
    .sort((a, b) => b.failRate - a.failRate);

  return { barItems, lineSeries, xLabels, heatRows, heatCols, heatMatrix, difficulty, mode };
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("srh-theme", theme);
}

function initTheme() {
  const saved = localStorage.getItem("srh-theme");
  setTheme(saved === "light" ? "light" : "dark");
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    setTheme(next);
    render();
  });
}

function routeFromHash() {
  const h = location.hash.replace(/^#\/?/, "") || "dashboard";
  return h.split("?")[0];
}

function setActiveNav(route) {
  document.querySelectorAll(".pill-link").forEach((a) => {
    a.classList.toggle("active", a.dataset.route === route);
  });
}

function scenarioCardsHtml() {
  const mode = getMode();
  return state.catalog.scenarios.map((s) => {
    const data = state.scenarios[s.id];
    const stats = passStats(data.regression, mode.atol);
    return `
      <a class="scenario-card" href="#/runner?scenario=${s.id}">
        <h3>${s.title}</h3>
        <div class="scenario-meta">${s.tagline} · ${stats.total} tensor checks</div>
        <p>${s.description}</p>
        <div class="scenario-actions">
          <span class="chip fail">${Math.round(stats.failRate)}% drift</span>
          <span class="chip open">Open in runner</span>
        </div>
      </a>`;
  }).join("");
}

function renderDashboard() {
  const m = computeDashboardMetrics();
  const stats = state.catalog.stats;
  return `
    <div class="fade-in">
      <section class="hero">
        <div>
          <h1>Silent Regression <em>Hunter</em></h1>
          <p class="hero-lead">
            Independent audit bench for ONNX compile-path drift — inspired by Nuro's FTL segment breaker pattern.
            Lower drift rates mean safer deploys. Find the first failing layer before the vehicle does.
          </p>
        </div>
        <div class="stat-grid">
          <div class="stat-chip"><span>Scenarios</span><strong>${stats.scenarios}</strong></div>
          <div class="stat-chip"><span>Tensor checks</span><strong>${stats.total_tensor_checks}</strong></div>
          <div class="stat-chip"><span>Tolerance modes</span><strong>${stats.tolerance_modes}</strong></div>
          <div class="stat-chip"><span>Compile paths</span><strong>${stats.compile_paths}</strong></div>
        </div>
      </section>

      <div class="mode-bar">
        <div class="mode-tabs" role="tablist">
          ${state.catalog.tolerance_modes.map((mode) => `
            <button type="button" class="mode-tab ${mode.id === state.mode ? "active" : ""}"
              data-mode="${mode.id}" role="tab">${mode.label}</button>
          `).join("")}
        </div>
        <p class="mode-hint">${m.mode.hint}</p>
        <span class="mode-badge">Bundled demo data</span>
      </div>

      <div class="panel-grid">
        <article class="panel">
          <div class="panel-head"><h2>Scenario Pass Rate</h2></div>
          <p class="panel-sub">Share of intermediate tensors within tolerance at the selected gate. Higher is safer.</p>
          <div class="chart-wrap" id="chart-pass-rate"></div>
          <p class="panel-foot">${stats.scenarios} scenarios · ${m.mode.label} mode · atol ${m.mode.atol}</p>
        </article>
        <article class="panel">
          <div class="panel-head"><h2>Tolerance Robustness</h2></div>
          <p class="panel-sub">Pass rate as tolerance widens from Strict → Relaxed. Steep drops mean brittle numerics.</p>
          <div class="chart-wrap" id="chart-robustness"></div>
          <p class="panel-foot">All tolerance modes · ${stats.scenarios} scenarios</p>
        </article>
      </div>

      <div class="panel-grid">
        <article class="panel">
          <div class="panel-head"><h2>Drift Risk Heatmap</h2></div>
          <p class="panel-sub">Failure rate per scenario and tolerance mode. Greener is safer, redder drifts more.</p>
          <div class="chart-wrap" id="chart-heatmap"></div>
          <p class="panel-foot">FP16 activation path · bundled ONNX toy graphs</p>
        </article>
        <article class="panel">
          <div class="panel-head"><h2>Scenario Difficulty</h2></div>
          <p class="panel-sub">Ranked by average drift across tolerance modes — the traps at the top fail most often.</p>
          <div class="chart-wrap" id="chart-difficulty"></div>
          <p class="panel-foot">Average fail rate across all modes</p>
        </article>
      </div>

      <h2 class="section-title">Scenario Library</h2>
      <p class="section-lead">Inspect bundled ONNX graphs behind the dashboard, then open any scenario in Live Runner.</p>
      <div class="scenario-grid">${scenarioCardsHtml()}</div>
    </div>`;
}

function renderRunner() {
  const id = state.runnerId || state.catalog.scenarios[0].id;
  const meta = state.catalog.scenarios.find((s) => s.id === id);
  const data = state.scenarios[id];
  const mode = getMode();
  const stats = passStats(data.regression, mode.atol);
  const reg = data.regression;
  const ff = reg.first_failure;

  const rows = (reg.divergences || []).map((d) => {
    const failed = (d.max_abs_diff || 0) > mode.atol;
    return `
      <tr class="${failed ? "failed" : ""}" data-node="${d.producer_node}">
        <td><code>${d.producer_node}</code></td>
        <td>${d.producer_op}</td>
        <td>${d.max_abs_diff?.toExponential(2) ?? "—"}</td>
        <td>${failed ? "DRIFT" : "ok"}</td>
      </tr>`;
  }).join("");

  return `
    <div class="fade-in runner-layout">
      <aside class="runner-sidebar">
        <h2>Scenarios</h2>
        ${state.catalog.scenarios.map((s) => `
          <button type="button" class="scenario-pick ${s.id === id ? "active" : ""}" data-scenario="${s.id}">
            <strong>${s.title}</strong>
            <span>${s.tagline}</span>
          </button>
        `).join("")}
      </aside>
      <div class="runner-main">
        <div class="runner-toolbar">
          <h1>${meta.title}</h1>
          <div>
            <button type="button" class="btn btn-primary" id="run-scan">Run scan</button>
            <a class="btn btn-secondary" href="${meta.report_href}" target="_blank" rel="noopener">Full report</a>
          </div>
        </div>
        <div class="scan-progress"><div class="scan-progress-bar" id="scan-bar"></div></div>
        <div class="runner-panel">
          <div class="live-stats">
            <div class="live-stat"><span>Layers drifted</span><strong id="stat-failed">${stats.total - stats.passed}</strong></div>
            <div class="live-stat"><span>Layers checked</span><strong>${stats.total}</strong></div>
            <div class="live-stat"><span>First failure</span><strong>${ff?.producer_node || "—"}</strong></div>
            <div class="live-stat"><span>Segments</span><strong>${meta.segment_count}</strong></div>
          </div>
          <div class="verdict ${stats.passRate === 100 ? "pass" : "fail"}" id="runner-verdict">
            ${stats.passRate === 100 ? "PASS — safe to ship" : "FAIL — do not ship"} · ${Math.round(stats.passRate)}% layers within tolerance
          </div>
          <h3>Layer heatmap</h3>
          <div class="graph-stage">
            <svg id="runner-graph" viewBox="0 0 580 280" role="img"></svg>
          </div>
          <h3>Divergence table</h3>
          <table class="divergence-table">
            <thead><tr><th>Node</th><th>Op</th><th>Max Δ</th><th>Status</th></tr></thead>
            <tbody id="divergence-body">${rows}</tbody>
          </table>
        </div>
        <div class="runner-panel">
          <h3>Artifacts</h3>
          <div class="artifact-links">
            <a href="${meta.report_href}" target="_blank" rel="noopener">Regression report</a>
            <a href="https://github.com/Abhishek21g/nuro-ftl-segment-receipts" target="_blank" rel="noopener">GitHub</a>
            <a href="#/vision">The Vision</a>
          </div>
        </div>
      </div>
    </div>`;
}

function renderVision() {
  return `
    <div class="fade-in">
      <section class="vision-hero">
        <h1>The Vision</h1>
        <p class="hero-lead">
          Nuro built FTL internally to stitch TensorRT, ONNX Runtime, and custom kernels — but third-party compilers
          still cause <strong>silent regressions</strong>. This open bench is the pre-deploy audit FTL assumes.
        </p>
      </section>

      <blockquote class="pull-quote">
        "Third party compilers are notorious for causing silent regressions in the model's performance."
        <cite>— Nuro FTL Model Compiler Framework blog</cite>
      </blockquote>

      <div class="vision-grid">
        <article class="vision-card">
          <h2>The gap</h2>
          <p>Teams discover drift in vehicle testing or sim — not at ONNX handoff. No public OSS runs
          <code>plan → run → doctor → report</code> with per-tensor receipts and FP32 breaker YAML.</p>
        </article>
        <article class="vision-card">
          <h2>What we ship</h2>
          <ul>
            <li>Scan every intermediate activation between reference and optimized paths</li>
            <li>Pinpoint the first topo failure + segment breaker rules</li>
            <li>CI-friendly doctor exit code — fail closed before deploy</li>
          </ul>
        </article>
        <article class="vision-card">
          <h2>Who it's for</h2>
          <p>ML infra engineers handing ONNX to multi-compiler pipelines — the same persona Nuro's FTL blog writes for.</p>
        </article>
      </div>

      <h2 class="section-title">60-second local run</h2>
      <pre class="code-block">pip install nuro-ftl-segment-receipts[onnx]

segment-receipts run examples/models/branch.onnx -o out/receipts
segment-receipts doctor out/receipts/&lt;run-id&gt;
open out/receipts/&lt;run-id&gt;/regression_report.html</pre>
    </div>`;
}

function bindDashboard() {
  document.querySelectorAll(".mode-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.mode = btn.dataset.mode;
      render();
    });
  });

  const m = computeDashboardMetrics();
  SRHCharts.renderBarChart(document.getElementById("chart-pass-rate"), { items: m.barItems });
  SRHCharts.renderLineChart(document.getElementById("chart-robustness"), {
    series: m.lineSeries,
    xLabels: m.xLabels,
  });
  SRHCharts.renderHeatmap(document.getElementById("chart-heatmap"), {
    rows: m.heatRows,
    cols: m.heatCols,
    matrix: m.heatMatrix,
  });
  SRHCharts.renderDifficultyChart(document.getElementById("chart-difficulty"), { items: m.difficulty });
}

function bindRunner() {
  const id = state.runnerId || state.catalog.scenarios[0].id;
  const data = state.scenarios[id];
  const mode = getMode();
  const reg = data.regression;
  const maxDiff = Math.max(...(reg.divergences || []).map((d) => d.max_abs_diff), mode.atol);
  const svg = document.getElementById("runner-graph");

  function paintGraph(revealed) {
    SRHCharts.renderGraphSvg(svg, reg.graph_layout || [], maxDiff, mode.atol, {
      activeNode: state.activeNode,
      revealed,
    });
    svg.querySelectorAll(".graph-node").forEach((g) => {
      g.addEventListener("click", () => {
        state.activeNode = state.activeNode === g.dataset.node ? null : g.dataset.node;
        paintGraph(revealed);
        document.querySelectorAll(".divergence-table tr").forEach((tr) => {
          tr.style.outline = tr.dataset.node === state.activeNode ? "1px solid var(--accent)" : "";
        });
      });
    });
  }

  paintGraph(state.scanning ? new Set() : null);

  document.querySelectorAll(".scenario-pick").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.runnerId = btn.dataset.scenario;
      state.activeNode = null;
      location.hash = `#/runner?scenario=${state.runnerId}`;
      render();
    });
  });

  document.getElementById("run-scan").addEventListener("click", async () => {
    if (state.scanning) return;
    state.scanning = true;
    const bar = document.getElementById("scan-bar");
    const layout = reg.graph_layout || [];
    const revealed = new Set();
    bar.style.width = "0%";

    for (let i = 0; i < layout.length; i++) {
      revealed.add(layout[i].node);
      bar.style.width = `${((i + 1) / layout.length) * 100}%`;
      paintGraph(revealed);
      await new Promise((r) => setTimeout(r, 280));
    }
    state.scanning = false;
    paintGraph(null);
  });
}

function parseRunnerScenario() {
  const q = location.hash.split("?")[1] || "";
  const params = new URLSearchParams(q);
  const s = params.get("scenario");
  if (s && state.catalog.scenarios.some((x) => x.id === s)) state.runnerId = s;
}

function render() {
  const route = routeFromHash();
  setActiveNav(route === "" ? "dashboard" : route);
  const app = document.getElementById("app");

  if (route === "runner") {
    parseRunnerScenario();
    app.innerHTML = renderRunner();
    bindRunner();
    return;
  }
  if (route === "vision") {
    app.innerHTML = renderVision();
    return;
  }
  app.innerHTML = renderDashboard();
  bindDashboard();
}

async function init() {
  initTheme();
  state.catalog = await loadJson("data/scenarios.json");
  await Promise.all(
    state.catalog.scenarios.map(async (s) => {
      state.scenarios[s.id] = await loadScenarioData(s.id, s.data_prefix);
    })
  );
  window.addEventListener("hashchange", render);
  render();
}

init().catch((err) => {
  document.getElementById("app").innerHTML = `
    <p style="color:var(--accent)">Failed to load demo data. Run <code>bash scripts/sync-demo.sh</code> first.</p>
    <pre>${err.message}</pre>`;
});
