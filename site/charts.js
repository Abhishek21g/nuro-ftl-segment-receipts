/* Chart utilities — SVG-only, no dependencies */

function lerpColor(t) {
  const clamped = Math.max(0, Math.min(1, t));
  const r = Math.round(62 + (255 - 62) * clamped);
  const g = Math.round(207 - 157 * clamped);
  const b = Math.round(142 - 97 * clamped);
  return `rgb(${r},${g},${b})`;
}

function failRateColor(rate) {
  return lerpColor(rate / 100);
}

function passRateColor(rate) {
  return lerpColor(1 - rate / 100);
}

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

function renderBarChart(container, { items, valueKey = "value", labelKey = "label", height = 280 }) {
  container.innerHTML = "";
  if (!items.length) return;

  const pad = { l: 42, r: 16, t: 16, b: 52 };
  const width = Math.max(320, container.clientWidth || 480);
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const max = Math.max(...items.map((d) => d[valueKey]), 1);

  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });

  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (innerH * i) / 4;
    const val = Math.round(max - (max * i) / 4);
    svg.appendChild(svgEl("line", {
      x1: pad.l, y1: y, x2: width - pad.r, y2: y,
      stroke: "var(--chart-grid)", "stroke-width": 1,
    }));
    const label = svgEl("text", {
      x: pad.l - 8, y: y + 4, fill: "var(--muted-2)",
      "font-size": 10, "text-anchor": "end",
    });
    label.textContent = `${val}%`;
    svg.appendChild(label);
  }

  const barW = innerW / items.length - 10;
  items.forEach((item, i) => {
    const val = item[valueKey];
    const h = (val / max) * innerH;
    const x = pad.l + i * (innerW / items.length) + 5;
    const y = pad.t + innerH - h;
    const color = passRateColor(val);

    svg.appendChild(svgEl("rect", {
      x, y, width: barW, height: Math.max(h, 2), rx: 4,
      fill: color,
    }));

    const pct = svgEl("text", {
      x: x + barW / 2, y: y - 6, fill: "var(--text)",
      "font-size": 11, "font-weight": 600, "text-anchor": "middle",
    });
    pct.textContent = `${Math.round(val)}%`;
    svg.appendChild(pct);

    const lbl = svgEl("text", {
      x: x + barW / 2, y: height - pad.b + 14, fill: "var(--muted)",
      "font-size": 10, "text-anchor": "middle",
    });
    const text = item[labelKey];
    lbl.textContent = text.length > 14 ? text.slice(0, 13) + "…" : text;
    svg.appendChild(lbl);
  });

  container.appendChild(svg);
}

function renderLineChart(container, { series, xLabels, height = 280 }) {
  container.innerHTML = "";
  if (!series.length || !xLabels.length) return;

  const pad = { l: 42, r: 72, t: 16, b: 40 };
  const width = Math.max(320, container.clientWidth || 480);
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const max = 100;
  const min = 0;

  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
  const colors = ["#3ecf8e", "#6ec8ff", "#ff8f6a", "#f0b429", "#c084fc", "#ff5c3a"];

  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (innerH * i) / 4;
    svg.appendChild(svgEl("line", {
      x1: pad.l, y1: y, x2: width - pad.r, y2: y,
      stroke: "var(--chart-grid)", "stroke-width": 1,
    }));
  }

  xLabels.forEach((lbl, i) => {
    const x = pad.l + (innerW * i) / (xLabels.length - 1);
    const t = svgEl("text", {
      x, y: height - 12, fill: "var(--muted)",
      "font-size": 10, "text-anchor": "middle",
    });
    t.textContent = lbl;
    svg.appendChild(t);
  });

  series.forEach((s, si) => {
    const color = colors[si % colors.length];
    const points = s.values.map((v, i) => {
      const x = pad.l + (innerW * i) / (xLabels.length - 1);
      const y = pad.t + innerH - ((v - min) / (max - min)) * innerH;
      return `${x},${y}`;
    }).join(" ");

    svg.appendChild(svgEl("polyline", {
      fill: "none", stroke: color, "stroke-width": 2.2,
      points,
    }));

    const last = s.values[s.values.length - 1];
    const lx = pad.l + innerW;
    const ly = pad.t + innerH - ((last - min) / (max - min)) * innerH;
    const end = svgEl("text", {
      x: lx + 6, y: ly + 4, fill: color, "font-size": 10, "font-weight": 600,
    });
    end.textContent = `${s.label} ${Math.round(last)}%`;
    svg.appendChild(end);
  });

  container.appendChild(svg);
}

function renderHeatmap(container, { rows, cols, matrix, height = 280 }) {
  container.innerHTML = "";
  const pad = { l: 110, r: 16, t: 28, b: 16 };
  const cell = 44;
  const width = pad.l + cols.length * cell + pad.r;
  const h = pad.t + rows.length * cell + pad.b;
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${h}`, role: "img" });

  cols.forEach((c, ci) => {
    const t = svgEl("text", {
      x: pad.l + ci * cell + cell / 2, y: 18,
      fill: "var(--muted)", "font-size": 10, "text-anchor": "middle",
    });
    t.textContent = c;
    svg.appendChild(t);
  });

  rows.forEach((row, ri) => {
    const rl = svgEl("text", {
      x: pad.l - 8, y: pad.t + ri * cell + cell / 2 + 4,
      fill: "var(--muted)", "font-size": 10, "text-anchor": "end",
    });
    rl.textContent = row.length > 16 ? row.slice(0, 15) + "…" : row;
    svg.appendChild(rl);

    cols.forEach((_, ci) => {
      const val = matrix[ri][ci];
      const x = pad.l + ci * cell;
      const y = pad.t + ri * cell;
      svg.appendChild(svgEl("rect", {
        x: x + 2, y: y + 2, width: cell - 4, height: cell - 4, rx: 6,
        fill: failRateColor(val),
      }));
      const pct = svgEl("text", {
        x: x + cell / 2, y: y + cell / 2 + 4,
        fill: "#081018", "font-size": 11, "font-weight": 700, "text-anchor": "middle",
      });
      pct.textContent = `${Math.round(val)}%`;
      svg.appendChild(pct);
    });
  });

  container.appendChild(svg);
}

function renderDifficultyChart(container, { items, height = 280 }) {
  renderBarChart(container, {
    items: items.map((d) => ({ label: d.label, value: d.failRate })),
    height,
  });
  const svg = container.querySelector("svg");
  if (svg) {
    svg.querySelectorAll("rect").forEach((rect, i) => {
      if (items[i]) rect.setAttribute("fill", failRateColor(items[i].failRate));
    });
    svg.querySelectorAll("text").forEach((t) => {
      if (t.textContent.endsWith("%") && !t.getAttribute("text-anchor")) {
        // bar value labels — keep
      }
    });
  }
}

function heatColor(diff, failed, maxDiff, atol) {
  if (!failed && diff <= atol) return "#1a3d2e";
  const t = Math.min(1, diff / Math.max(maxDiff, atol, 1e-8));
  const r = Math.round(160 + 90 * t);
  const g = Math.round(50 - 30 * t);
  const b = Math.round(45 - 15 * t);
  return `rgb(${r},${g},${b})`;
}

function renderGraphSvg(svg, layout, maxDiff, atol, { activeNode = null, revealed = null } = {}) {
  const cellW = 118;
  const cellH = 46;
  const gap = 10;
  const offsetX = 20;
  const offsetY = 14;

  svg.innerHTML = layout.map((n) => {
    const x = offsetX + n.col * (cellW + gap);
    const y = offsetY + n.row * (cellH + gap);
    const diff = n.max_abs_diff || 0;
    const isRevealed = revealed === null || revealed.has(n.node);
    const fill = isRevealed ? heatColor(diff, n.failed, maxDiff, atol) : "#1a222d";
    const label = n.node.length > 13 ? n.node.slice(0, 12) + "…" : n.node;
    const cls = ["graph-node"];
    if (activeNode && activeNode !== n.node) cls.push("dim");
    if (activeNode === n.node) cls.push("active");
    return `
      <g class="${cls.join(" ")}" data-node="${n.node}">
        <rect x="${x}" y="${y}" width="${cellW}" height="${cellH}" rx="6"
          fill="${fill}" stroke="#2a3a48"/>
        <text x="${x + 6}" y="${y + 15}" fill="#f0f4f8" font-size="10" font-weight="600">${label}</text>
        <text x="${x + 6}" y="${y + 29}" fill="#a8b8c8" font-size="9">${n.op}</text>
        <text x="${x + 6}" y="${y + 41}" fill="#fff" font-size="8" opacity="0.9">Δ ${diff.toExponential(1)}</text>
      </g>`;
  }).join("");

  const rows = Math.max(...layout.map((n) => n.row), 0) + 1;
  svg.setAttribute("viewBox", `0 0 560 ${offsetY * 2 + rows * (cellH + gap)}`);
}

window.SRHCharts = {
  renderBarChart,
  renderLineChart,
  renderHeatmap,
  renderDifficultyChart,
  renderGraphSvg,
  passRateColor,
  failRateColor,
};
