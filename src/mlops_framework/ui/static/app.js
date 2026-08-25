// Gateflow Management Console — vanilla JS, no framework and no build step.
// The shell is composed server-side in ui/mount.py; every page calls its own
// init function; all data comes from the JSON API.

const API = "/api";

/* ------------------------------------------------------------------ */
/* Console shell — side-nav toggle and theme                           */
/* ------------------------------------------------------------------ */

const NAV_BREAKPOINT = 900; // keep in sync with the media query in app.css
const THEME_KEY = "gateflow-theme";

// Called from the shell in ui/mount.py, on every page.
function initShell() {
  const body = document.body;
  const toggle = document.getElementById("nav-toggle");
  const scrim = document.getElementById("nav-scrim");

  // One button, two behaviours: on a wide viewport it folds the menu
  // away beside the content; on a narrow one it opens a drawer over it.
  const isNarrow = () => window.innerWidth <= NAV_BREAKPOINT;

  function setExpanded() {
    const open = isNarrow()
      ? body.classList.contains("nav-open")
      : !body.classList.contains("nav-collapsed");
    toggle.setAttribute("aria-expanded", String(open));
    scrim.hidden = !body.classList.contains("nav-open");
  }

  function closeDrawer() {
    body.classList.remove("nav-open");
    setExpanded();
  }

  toggle.addEventListener("click", () => {
    body.classList.toggle(isNarrow() ? "nav-open" : "nav-collapsed");
    setExpanded();
  });
  scrim.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDrawer();
  });
  // Resizing past the breakpoint leaves the drawer state stale otherwise.
  window.addEventListener("resize", () => {
    if (!isNarrow()) closeDrawer();
    else setExpanded();
  });
  setExpanded();

  const themeBtn = document.getElementById("theme-toggle");
  themeBtn.addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem(THEME_KEY, root.dataset.theme); } catch (e) { /* private mode */ }
  });

  for (const b of document.querySelectorAll('[data-action="reload"]')) {
    b.addEventListener("click", () => location.reload());
  }
}

async function api(path, opts = {}) {
  const r = await fetch(API + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!r.ok) {
    const text = await r.text().catch(() => "");
    throw new Error(`${r.status} ${r.statusText}: ${text}`);
  }
  if (r.status === 204) return null;
  return r.json();
}

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") {
      e.addEventListener(k.substring(2).toLowerCase(), v);
    } else if (v !== null && v !== undefined && v !== false) {
      e.setAttribute(k, v);
    }
  }
  for (const c of children) {
    if (c == null || c === false) continue;
    if (Array.isArray(c)) { for (const x of c) if (x != null) e.appendChild(x); continue; }
    if (typeof c === "string" || typeof c === "number") {
      e.appendChild(document.createTextNode(String(c)));
    } else e.appendChild(c);
  }
  return e;
}

function svgEl(tag, attrs = {}, ...children) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) e.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    if (typeof c === "string" || typeof c === "number") {
      e.appendChild(document.createTextNode(String(c)));
    } else e.appendChild(c);
  }
  return e;
}

function statusKind(status) {
  const s = String(status || "").toLowerCase();
  if (["success", "ready", "production", "passed", "approved_ok", "no_drift", "promoted", "completed"].includes(s)) return "success";
  if (["failed", "rejected", "blocked", "upstream_failed", "drift_detected"].includes(s)) return "failed";
  if (["running", "queued"].includes(s)) return "running";
  if (["pending", "training", "scheduled", "candidate"].includes(s)) return "pending";
  if (["cancelled", "archived", "skipped", "removed"].includes(s)) return "cancelled";
  if (s === "approved") return "approved";
  return "";
}

// A table cell with a primary value and a muted mono identifier
// underneath it (e.g. a pipeline id over the execution id it actually
// ran as). `sub` is optional — a row with nothing to add underneath
// (no execution id yet, PENDING runs) just renders the primary line.
function cellWithSub(primary, sub) {
  return el("div", {},
    el("div", {}, primary),
    sub ? el("div", { class: "cell-sub" }, sub) : null);
}

function statusBadge(status) {
  if (status == null || status === "") return el("span", { class: "faint" }, "—");
  return el("span", { class: `badge ${statusKind(status)}` }, String(status));
}

// The API serves timestamps straight off the ORM. On SQLite those come
// back without a timezone offset even though the framework writes UTC,
// and `new Date("...")` reads an offset-less string as *local* time —
// which silently shifts every "x ago" by the viewer's UTC offset. Pin
// them to UTC unless the string already says otherwise.
function parseTs(s) {
  if (!s) return null;
  // MLflow reports times as epoch milliseconds, the framework's own rows as
  // ISO strings. Both reach these formatters, so accept a number directly
  // rather than running it through the string path, where appending "Z"
  // would turn it into an Invalid Date and silently render "—".
  if (typeof s === "number") {
    const d = new Date(s);
    return isNaN(d.getTime()) ? null : d;
  }
  const hasZone = /[zZ]$|[+-]\d{2}:?\d{2}$/.test(s);
  const d = new Date(hasZone ? s : s + "Z");
  return isNaN(d.getTime()) ? null : d;
}

const fmt = {
  pct(x) { return x == null ? "—" : (x * 100).toFixed(1) + "%"; },
  num(x) { return x == null ? "—" : Number(x).toLocaleString(); },
  metric(x) {
    if (x == null) return "—";
    if (typeof x !== "number") return String(x);
    if (Number.isInteger(x)) return x.toLocaleString();
    return Math.abs(x) >= 1000 ? x.toFixed(1) : x.toFixed(4);
  },
  time(s) {
    const d = parseTs(s);
    return d ? d.toLocaleString() : "—";
  },
  ago(s) {
    const t = parseTs(s);
    if (!t) return "—";
    const d = (Date.now() - t.getTime()) / 1000;
    if (!isFinite(d)) return "—";
    if (d < 0) return "just now";
    if (d < 60) return `${Math.round(d)}s ago`;
    if (d < 3600) return `${Math.round(d / 60)}m ago`;
    if (d < 86400) return `${Math.round(d / 3600)}h ago`;
    return `${Math.round(d / 86400)}d ago`;
  },
  dur(sec) {
    if (sec == null) return "—";
    if (sec < 1) return `${(sec * 1000).toFixed(0)}ms`;
    if (sec < 60) return `${sec.toFixed(1)}s`;
    const m = Math.floor(sec / 60), s = Math.round(sec % 60);
    if (m < 60) return `${m}m ${s}s`;
    return `${Math.floor(m / 60)}h ${m % 60}m`;
  },
  bytes(b) {
    if (b == null) return "—";
    const u = ["B", "KB", "MB", "GB"];
    let i = 0, n = Number(b);
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
  },
  hash(h, n = 12) { return !h ? "—" : h.substring(0, n) + (h.length > n ? "…" : ""); },
};

function banner(msg, kind = "") {
  return el("div", { class: `banner ${kind}` }, msg);
}

function emptyRow(colspan, msg) {
  return el("tr", {}, el("td", { colspan: String(colspan), class: "empty" }, msg));
}

function setError(container, e) {
  container.replaceChildren(banner(`Could not load: ${e.message}`, "err"));
}

// `Node.replaceChildren` stringifies whatever it is handed, so a
// conditional child written as `cond ? node : null` renders the literal
// text "null". The el() helper drops nullish children; this makes the
// container-level call behave the same way.
function mount(container, ...children) {
  container.replaceChildren(...children.flat().filter((c) => c != null && c !== false));
}

/* ------------------------------------------------------------------ */
/* Charts — hand-rolled inline SVG. No chart library, no build step.    */
/* ------------------------------------------------------------------ */

// A line chart of {step, value} points. Sized in a viewBox so it scales
// with its container rather than needing a resize listener.
function lineChart(title, points, opts = {}) {
  const W = 320, H = 140, P = { t: 8, r: 8, b: 20, l: 38 };
  const wrap = el("div", { class: "chart" }, el("div", { class: "chart-title" }, title));
  if (!points || points.length === 0) {
    wrap.appendChild(el("div", { class: "empty" }, "No history"));
    return wrap;
  }

  const xs = points.map((p) => p.step);
  const ys = points.map((p) => p.value);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  if (yMin === yMax) { yMin -= Math.abs(yMin) * 0.1 || 0.5; yMax += Math.abs(yMax) * 0.1 || 0.5; }
  const pad = (yMax - yMin) * 0.08;
  yMin -= pad; yMax += pad;

  const sx = (x) => P.l + ((x - xMin) / (xMax - xMin || 1)) * (W - P.l - P.r);
  const sy = (y) => H - P.b - ((y - yMin) / (yMax - yMin || 1)) * (H - P.t - P.b);

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });

  for (let i = 0; i <= 3; i++) {
    const v = yMin + ((yMax - yMin) * i) / 3;
    const y = sy(v);
    svg.appendChild(svgEl("line", { class: "gridline", x1: P.l, y1: y, x2: W - P.r, y2: y }));
    svg.appendChild(svgEl("text", { class: "tick-label", x: P.l - 5, y: y + 3, "text-anchor": "end" }, v.toFixed(3)));
  }
  svg.appendChild(svgEl("line", { class: "axis", x1: P.l, y1: H - P.b, x2: W - P.r, y2: H - P.b }));

  const d = points.map((p, i) => `${i ? "L" : "M"}${sx(p.step).toFixed(2)},${sy(p.value).toFixed(2)}`).join(" ");
  svg.appendChild(svgEl("path", { class: "line", d, stroke: opts.color || null }));
  if (points.length <= 40) {
    for (const p of points) {
      svg.appendChild(svgEl("circle", { class: "point", cx: sx(p.step), cy: sy(p.value), r: 2.5, fill: opts.color || null }));
    }
  }
  svg.appendChild(svgEl("text", { class: "tick-label", x: P.l, y: H - 6 }, String(xMin)));
  svg.appendChild(svgEl("text", { class: "tick-label", x: W - P.r, y: H - 6, "text-anchor": "end" }, String(xMax)));

  wrap.appendChild(svg);
  return wrap;
}

// An overlay of several {step, value} series on one shared scale — one
// training curve per run, so "who learned faster/better" is a shape
// comparison, not just a table of final numbers. Colour is assigned by
// each series' fixed index (--series-1.. in app.css, see its comment),
// never by rank, so a run keeps its colour if another run is added to
// or removed from the comparison. Always paired with a text legend
// (below) and the caller's own metrics table (compareTable in
// initRunsCompare) — identity is never colour-alone. Caps at 8 series,
// the palette's validated slot count; a 9th run's series is dropped
// rather than assigned a repeated or unvalidated hue.
function multiLineChart(title, series) {
  const W = 320, H = 140, P = { t: 8, r: 8, b: 20, l: 38 };
  const wrap = el("div", { class: "chart" }, el("div", { class: "chart-title" }, title));
  const plotted = series.filter((s) => s.points && s.points.length).slice(0, 8);
  if (!plotted.length) {
    wrap.appendChild(el("div", { class: "empty" }, "No history"));
    return wrap;
  }

  const allPoints = plotted.flatMap((s) => s.points);
  const xs = allPoints.map((p) => p.step);
  const ys = allPoints.map((p) => p.value);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  if (yMin === yMax) { yMin -= Math.abs(yMin) * 0.1 || 0.5; yMax += Math.abs(yMax) * 0.1 || 0.5; }
  const pad = (yMax - yMin) * 0.08;
  yMin -= pad; yMax += pad;

  const sx = (x) => P.l + ((x - xMin) / (xMax - xMin || 1)) * (W - P.l - P.r);
  const sy = (y) => H - P.b - ((y - yMin) / (yMax - yMin || 1)) * (H - P.t - P.b);

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
  for (let i = 0; i <= 3; i++) {
    const v = yMin + ((yMax - yMin) * i) / 3;
    const y = sy(v);
    svg.appendChild(svgEl("line", { class: "gridline", x1: P.l, y1: y, x2: W - P.r, y2: y }));
    svg.appendChild(svgEl("text", { class: "tick-label", x: P.l - 5, y: y + 3, "text-anchor": "end" }, v.toFixed(3)));
  }
  svg.appendChild(svgEl("line", { class: "axis", x1: P.l, y1: H - P.b, x2: W - P.r, y2: H - P.b }));

  plotted.forEach((s, i) => {
    const slot = i + 1;
    const d = s.points.map((p, j) => `${j ? "L" : "M"}${sx(p.step).toFixed(2)},${sy(p.value).toFixed(2)}`).join(" ");
    svg.appendChild(svgEl("path", { class: `line s${slot}`, d }));
    if (s.points.length <= 40) {
      for (const p of s.points) {
        svg.appendChild(svgEl("circle", { class: `point s${slot}`, cx: sx(p.step), cy: sy(p.value), r: 2 }));
      }
    }
  });
  svg.appendChild(svgEl("text", { class: "tick-label", x: P.l, y: H - 6 }, String(xMin)));
  svg.appendChild(svgEl("text", { class: "tick-label", x: W - P.r, y: H - 6, "text-anchor": "end" }, String(xMax)));
  wrap.appendChild(svg);

  wrap.appendChild(el("div", { class: "legend", style: "margin-top:8px" },
    ...plotted.map((s, i) =>
      el("span", { class: "legend-item" },
        el("span", { class: "dot", style: `background:var(--series-${i + 1})` }),
        s.label))));

  return wrap;
}

// Horizontal bar comparison, used to compare one metric across runs or
// model versions. Bars share a scale so lengths are directly comparable.
function barChart(title, entries) {
  const wrap = el("div", { class: "chart" }, el("div", { class: "chart-title" }, title));
  if (!entries.length) {
    wrap.appendChild(el("div", { class: "empty" }, "No data"));
    return wrap;
  }
  const max = Math.max(...entries.map((e) => Math.abs(e.value)), 1e-9);
  const rows = el("div", {});
  for (const e of entries) {
    rows.appendChild(
      el("div", { style: "display:flex;align-items:center;gap:10px;margin:6px 0" },
        el("span", { class: "mono", style: "min-width:96px" }, e.label),
        el("div", { class: "bar-track", style: "flex:1" },
          el("div", { class: `bar-fill ${e.kind || ""}`, style: `width:${(Math.abs(e.value) / max) * 100}%` })),
        el("span", { class: "mono", style: "min-width:64px;text-align:right" }, fmt.metric(e.value)))
    );
  }
  wrap.appendChild(rows);
  return wrap;
}

// A shared dashboard-row tile: a title pinned to the top over a body
// that's vertically centred in whatever height the row's equal-height
// grid (grid-3's default stretch) hands it — so three cards built from
// very different content (a kv list, a ring, a table) read as one
// balanced row instead of the shortest one trailing off into a block
// of empty space. See .dash-tile-body in app.css.
function dashTile(title, body) {
  return el("div", { class: "chart dash-tile" },
    el("div", { class: "chart-title" }, title),
    el("div", { class: "dash-tile-body" }, body));
}

// A proportional ring — Airflow's own "DAG Run States" widget on its
// home page (http://localhost:8080 in local dev), reproduced compactly
// (legend beside the ring, not stacked above it, and sized to share a
// dashboard row with two other tiles) with this console's existing
// status tokens (the same ones statusKind()/badges/the run strip
// already use) rather than inventing new colors. entries:
// [{ label, value, kind }], kind one of the statusKind() names so the
// legend dot and the CSS below already agree on a color.
function donutChart(title, entries) {
  const total = entries.reduce((s, e) => s + (e.value || 0), 0);

  const legend = el("div", { class: "legend vertical" },
    ...entries.map((e) =>
      el("span", { class: `legend-item ${e.kind || ""}` },
        el("span", { class: "dot" }), `${e.label} (${fmt.num(e.value)})`)));

  const cx = 60, cy = 60, r = 44, sw = 18;
  const svg = svgEl("svg", { viewBox: "0 0 120 120", role: "img", "aria-label": `${title}, total ${total}` });
  const nonzero = entries.filter((e) => (e.value || 0) > 0);

  if (nonzero.length === 0) {
    // No runs at all yet — an honest "nothing to show" ring rather than
    // Airflow's own quirky habit of quartering the ring by legend slot
    // when the total is zero, which looks like data that isn't there.
    svg.appendChild(svgEl("circle", { class: "donut-arc empty", cx, cy, r, "stroke-width": sw, fill: "none" },
      svgEl("title", {}, "No runs yet")));
  } else if (nonzero.length === 1) {
    svg.appendChild(svgEl("circle", { class: `donut-arc ${nonzero[0].kind || ""}`, cx, cy, r, "stroke-width": sw, fill: "none" },
      svgEl("title", {}, `${nonzero[0].label}: ${fmt.num(nonzero[0].value)}`)));
  } else {
    // Fixed surface gap between segments (the same spacer every other
    // chart on this page uses to separate marks instead of a border),
    // expressed as an angle at this radius.
    const gapAngle = 5 / r;
    const spanAngle = 2 * Math.PI - gapAngle * nonzero.length;
    let angle = -Math.PI / 2; // start at 12 o'clock, sweep clockwise
    for (const e of nonzero) {
      const sweep = spanAngle * (e.value / total);
      const start = angle, end = angle + sweep;
      const x1 = cx + r * Math.cos(start), y1 = cy + r * Math.sin(start);
      const x2 = cx + r * Math.cos(end), y2 = cy + r * Math.sin(end);
      svg.appendChild(svgEl("path", {
        class: `donut-arc ${e.kind || ""}`,
        d: `M ${x1} ${y1} A ${r} ${r} 0 ${sweep > Math.PI ? 1 : 0} 1 ${x2} ${y2}`,
        "stroke-width": sw, fill: "none",
      }, svgEl("title", {}, `${e.label}: ${fmt.num(e.value)} (${fmt.pct(e.value / total)})`)));
      angle = end + gapAngle;
    }
  }

  return dashTile(title, el("div", { class: "donut-chart" },
    el("div", { class: "donut-body" }, legend, svg),
    el("div", { class: "donut-total" }, "on a total of ", el("strong", {}, fmt.num(total)))));
}

/* ------------------------------------------------------------------ */
/* Sortable table helper                                               */
/* ------------------------------------------------------------------ */

function makeSortable(table, rows, columns, render) {
  let sortKey = null, sortDir = -1;
  const thead = table.querySelector("thead tr");
  const tbody = table.querySelector("tbody");

  function draw() {
    const data = rows.slice();
    if (sortKey) {
      data.sort((a, b) => {
        const x = sortKey(a), y = sortKey(b);
        if (x == null && y == null) return 0;
        if (x == null) return 1;
        if (y == null) return -1;
        return (x > y ? 1 : x < y ? -1 : 0) * sortDir;
      });
    }
    tbody.replaceChildren(...data.map(render));
    if (!data.length) tbody.appendChild(emptyRow(columns.length, "Nothing to show."));
  }

  thead.replaceChildren(
    ...columns.map((c) => {
      const th = el("th", { class: c.sort ? "sortable" : "" }, c.label);
      if (c.sort) {
        th.addEventListener("click", () => {
          if (sortKey === c.sort) sortDir = -sortDir;
          else { sortKey = c.sort; sortDir = -1; }
          for (const other of thead.querySelectorAll(".arrow")) other.remove();
          th.appendChild(el("span", { class: "arrow" }, sortDir < 0 ? "▾" : "▴"));
          draw();
        });
      }
      return th;
    })
  );
  draw();
  return draw;
}

/* ------------------------------------------------------------------ */
/* Dashboard                                                           */
/* ------------------------------------------------------------------ */

async function initDashboard() {
  const grid = document.getElementById("kpi-grid");
  const inventory = document.getElementById("inventory-card");
  const outcomes = document.getElementById("outcomes-chart");
  const modelsCard = document.getElementById("models-card");
  let d;
  try {
    d = await api("/dashboard");
  } catch (e) {
    setError(grid, e);
    return;
  }

  // Hero tiles: only the signals that call for a decision right now —
  // is anything running, is anything failing, is anything actually
  // live, is the platform healthy overall. Everything else below is
  // inventory (how much exists), which matters less moment-to-moment
  // and reads better small than as an equally-weighted KPI tile next
  // to these — the previous version gave all nine the same visual
  // weight, which is exactly what made the section feel like a wall
  // of boxes.
  const rateKind = d.success_rate >= 0.8 ? "ok" : d.success_rate >= 0.5 ? "warn" : "err";
  const heroItems = [
    {
      label: "Active runs", value: d.active_runs,
      kind: d.active_runs > 0 ? "warn" : "",
      caption: d.active_runs > 0 ? `${d.active_runs} running now` : "Idle",
    },
    {
      label: "Failed", value: d.failed_runs,
      kind: d.failed_runs > 0 ? "err" : "",
      caption: d.failed_runs > 0 ? "Investigate now" : "None in range",
    },
    {
      label: "In production", value: d.production_models, kind: "ok",
      caption: `${d.production_models} model${d.production_models === 1 ? "" : "s"} live`,
    },
    {
      label: "Success rate",
      value: fmt.pct(d.success_rate),
      kind: rateKind,
      caption: rateKind === "ok" ? "Trending up" : rateKind === "warn" ? "Watch closely" : "Needs attention",
    },
  ];
  grid.replaceChildren(
    ...heroItems.map((i) =>
      el("div", { class: `kpi ${i.kind || ""}` },
        el("div", { class: "label" }, i.label),
        el("div", { class: "value" }, i.value == null ? "—" : String(i.value)),
        i.caption ? el("div", { class: "caption" }, i.caption) : null))
  );

  // Inventory: raw counts, de-emphasised (the existing .kv key-value
  // list run-detail already uses) rather than four more equal-weight
  // tiles.
  if (inventory) {
    inventory.replaceChildren(dashTile("Platform inventory",
      el("dl", { class: "kv" },
        el("dt", {}, "Datasets"), el("dd", {}, String(d.datasets)),
        el("dt", {}, "Dataset versions"), el("dd", {}, String(d.dataset_versions)),
        el("dt", {}, "Total runs"), el("dd", {}, String(d.total_runs)),
        el("dt", {}, "Models"), el("dd", {}, String(d.models)))));
  }

  // Outcomes: the same success/failed/active counts as above, as the
  // Airflow-style "run states" ring (see donutChart()) instead of a
  // sentence — needs no data beyond what /dashboard already returns.
  // kind names match statusKind() (success/failed/running) so the
  // legend dot and the .donut-arc CSS agree with the rest of the
  // console's status colors instead of the kpi tiles' own ok/err/warn.
  if (outcomes) {
    outcomes.replaceChildren(
      donutChart("Run outcomes", [
        { label: "Successful", value: d.success_runs, kind: "success" },
        { label: "Failed", value: d.failed_runs, kind: "failed" },
        { label: "Active", value: d.active_runs, kind: "running" },
      ]));
  }

  // Models: which ones are actually in production right now and how
  // they're performing — the "In production" KPI tile above only has
  // room for a count, and inventory only has room for a total. In-
  // production models sort first (the one list a reader opens this
  // card to check); best-effort like the activity fetch below, so a
  // failed call here doesn't blank out the tiles above it.
  if (modelsCard) {
    try {
      const models = await api("/models");
      const rows = models
        .slice()
        .sort((a, b) => (b.production_version ? 1 : 0) - (a.production_version ? 1 : 0))
        .slice(0, 5);
      modelsCard.replaceChildren(dashTile("Models",
        rows.length
          ? el("div", { class: "table-wrap" },
              el("table", {},
                el("thead", {}, el("tr", {},
                  el("th", {}, "Model"), el("th", {}, "Production"), el("th", {}, "Key metric"))),
                el("tbody", {},
                  ...rows.map((m) => {
                    const prod = m.production_version;
                    const best = prod ? bestMetric({ metrics: prod.metrics }) : null;
                    return el("tr", {},
                      el("td", { class: "truncate" }, el("a", { href: `/models/${m.id}` }, m.name)),
                      el("td", {}, prod ? statusBadge("PRODUCTION") : el("span", { class: "faint" }, "none")),
                      el("td", { class: "mono" }, best ? `${best.name} ${fmt.metric(best.value)}` : "—"));
                  }))))
          : el("div", { class: "empty" }, "No models yet")));
    } catch (e) {
      setError(modelsCard, e);
    }
  }

  // Recent activity — the Airflow-style run strip plus the latest rows.
  const recent = document.getElementById("recent-runs");
  if (!recent) return;
  try {
    const runs = await api("/training-runs?limit=40");
    const strip = el("div", { class: "run-strip" },
      ...runs.slice().reverse().map((r) =>
        el("div", {
          class: `tick ${statusKind(r.status)}`,
          title: `Run ${r.id} — ${r.status}${r.duration_seconds != null ? " — " + fmt.dur(r.duration_seconds) : ""}`,
          style: `height:${Math.max(8, Math.min(34, (r.duration_seconds || 1) / 8 + 8))}px`,
        })));

    const table = el("table", {}, el("thead", {}, el("tr", {})), el("tbody", {}));
    recent.replaceChildren(
      el("div", { class: "card", style: "margin-bottom:16px" },
        el("div", { class: "chart-title", style: "margin-bottom:8px" },
          "Recent runs — bar height is duration, colour is status"),
        strip),
      el("div", { class: "table-wrap" }, table));

    makeSortable(table, runs.slice(0, 10),
      [{ label: "Run" }, { label: "Status" }, { label: "Pipeline" }, { label: "Duration" }, { label: "Started" }],
      (r) => el("tr", {},
        el("td", {}, el("a", { href: `/runs/${r.id}` }, `#${r.id}`)),
        el("td", {}, statusBadge(r.status)),
        el("td", { class: "mono truncate", title: r.pipeline_id || "" }, r.pipeline_id || "—"),
        // "mono", not "num" (right-aligned) — this table's headers are
        // all left-aligned, so a lone right-aligned cell just drifts
        // away from "Duration" above it. Same fix as the Models page's
        // Versions/Key metric columns.
        el("td", { class: "mono" }, fmt.dur(r.duration_seconds)),
        el("td", { class: "muted nowrap" }, fmt.ago(r.started_at || r.created_at))));
  } catch (e) {
    setError(recent, e);
  }
}

/* ------------------------------------------------------------------ */
/* Datasets                                                            */
/* ------------------------------------------------------------------ */

async function initDatasets() {
  const table = document.querySelector("table");
  try {
    const rows = await api("/datasets");
    makeSortable(table, rows,
      [
        { label: "Name", sort: (d) => d.name },
        { label: "Description" },
        { label: "Versions", sort: (d) => d.version_count },
        { label: "Latest rows", sort: (d) => d.latest_version?.row_count },
        { label: "Schema" },
      ],
      (ds) => el("tr", {},
        el("td", {}, el("a", { href: `/datasets/${ds.id}` }, ds.name)),
        el("td", { class: "muted" }, ds.description || "—"),
        // "mono", not "num" — same headers-are-left-aligned mismatch as
        // the Models table's Versions/Key metric columns.
        el("td", { class: "mono" }, String(ds.version_count)),
        el("td", { class: "mono" }, fmt.num(ds.latest_version?.row_count)),
        el("td", { class: "mono faint" }, fmt.hash(ds.latest_version?.schema_hash, 10))));
  } catch (e) {
    setError(table.parentElement, e);
  }
}

// "Run check" on a version's drift panel. The framework cannot compute
// drift here — it never reads dataset files (see drift.py's module
// docstring) — so this queues an Airflow DAG that can, and the panel
// picks the verdict up on the next load. 202 with no result is the
// honest response, so the button says so rather than pretending to have
// an answer.
//
// Hidden on version 1 of a dataset: with nothing earlier to compare
// against, the endpoint answers 422, and a button that always fails is
// worse than no button.
function driftCheckButton(v) {
  if (!v.version_number || v.version_number < 2) return null;
  const btn = el("button", { class: "btn" }, "Run check");
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "Queuing…";
    try {
      const r = await apiWrite(`/drift/${v.id}/check`, {
        method: "POST", body: JSON.stringify({}),
      });
      flash(
        `Drift check queued on ${r.dag_id} (v${v.version_number} vs #${r.reference_dataset_version_id}). ` +
        "The verdict appears here when the DAG finishes.",
        "ok",
      );
    } catch (e) {
      flash(e.message, "err");
    } finally {
      btn.disabled = false;
      btn.textContent = "Run check";
    }
  });
  return btn;
}

// "Train now" on a dataset version. Creates the run and hands it to
// Airflow in one gated call (POST /training-runs) — the console never
// touches /api/internal/*, which is the DAG's own callback surface.
//
// The entrypoint is asked for rather than guessed: pipeline_id is a
// dag_id to AirflowOrchestrator, and the module:callable it actually
// runs is a separate thing the framework cannot infer.
function trainNowButton(v) {
  const btn = el("button", { class: "btn primary" }, "Train now");
  btn.addEventListener("click", async () => {
    const entrypoint = window.prompt(
      "Training entrypoint (module:callable) to run on this version:",
      "case_studies.fraud_detection.pipelines:train_xgboost",
    );
    if (!entrypoint) return;
    const modelName = window.prompt(
      "Register the result under which model? (blank = train only, register nothing)",
      "",
    );
    btn.disabled = true;
    btn.textContent = "Starting…";
    try {
      const body = { dataset_version_id: v.id, training_entrypoint: entrypoint.trim() };
      if (modelName && modelName.trim()) body.model_name = modelName.trim();
      const r = await apiWrite("/training-runs", {
        method: "POST", body: JSON.stringify(body),
      });
      flash(
        `Training run #${r.training_run_id} queued on ${r.pipeline_id}.`,
        "ok",
      );
      location.href = `/runs/${r.training_run_id}`;
    } catch (e) {
      flash(e.message, "err");
      btn.disabled = false;
      btn.textContent = "Train now";
    }
  });
  return btn;
}

// One version's facts/readiness/drift/schema panel — used both for the
// single-version view (the default) and reused nowhere else, since the
// two-version compare below needs its fields side by side, not stacked.
async function datasetVersionSection(v) {
  const meta = v.metadata || {};
  let readiness = null;
  try { readiness = await api(`/readiness/${v.id}`); } catch { /* optional */ }
  let drift = null;
  try { drift = await api(`/drift/${v.id}`); } catch { /* optional */ }

  const facts = el("dl", { class: "kv" },
    el("dt", {}, "Rows"), el("dd", {}, fmt.num(v.row_count)),
    el("dt", {}, "Storage URI"), el("dd", {}, v.storage_uri),
    el("dt", {}, "Content SHA-256"), el("dd", {}, meta.content_sha256 || "not recorded"),
    el("dt", {}, "Schema hash"), el("dd", {}, v.schema_hash),
    el("dt", {}, "Version checksum"), el("dd", {}, v.checksum),
    el("dt", {}, "Size"), el("dd", {}, fmt.bytes(meta.size_bytes)),
    el("dt", {}, "Immutable"), el("dd", {}, v.is_immutable ? "yes" : "no"),
    el("dt", {}, "Created"), el("dd", {}, fmt.time(v.created_at)));

  const classBalance = meta.n_fraud != null
    ? el("div", { class: "card" },
        el("div", { class: "chart-title" }, "Class balance"),
        el("div", { class: "metric-grid" },
          el("div", { class: "metric" },
            el("div", { class: "name" }, "positive"), el("div", { class: "val" }, fmt.num(meta.n_fraud))),
          el("div", { class: "metric" },
            el("div", { class: "name" }, "ratio"), el("div", { class: "val" }, fmt.pct(meta.fraud_ratio))),
          el("div", { class: "metric" },
            el("div", { class: "name" }, "missing"), el("div", { class: "val" }, fmt.num(meta.missing_values)))))
    : null;

  const schemaRows = (meta.columns || []).map((c) =>
    el("tr", {},
      el("td", { class: "mono" }, c.name),
      el("td", { class: "mono muted" }, c.dtype)));

  const readinessPanel = el("div", { class: "card" },
    el("div", { class: "chart-title" }, "Readiness"),
    readiness
      ? el("div", {},
          el("div", { style: "margin-bottom:8px" }, statusBadge(readiness.status)),
          el("div", { class: "task-grid" },
            ...Object.entries(readiness.checks || {}).map(([name, outcome]) =>
              el("div", { class: `task-cell ${statusKind(outcome)}` },
                el("span", { class: "dot" }), name,
                el("span", { class: "state" }, outcome)))),
          (readiness.reasons || []).length
            ? el("ul", { class: "muted", style: "margin:10px 0 0;padding-left:18px" },
                ...readiness.reasons.map((r) => el("li", {}, r)))
            : null)
      : el("div", { class: "muted" }, "Not evaluated yet."));

  // A version can be either side of a drift comparison — the API
  // resolves that; here we just render whatever the latest
  // evaluation involving this version says.
  const driftFeatures = (drift && drift.details && drift.details.feature_results) || [];
  const driftPanel = el("div", { class: "card" },
    el("div", { class: "section-head", style: "margin:0 0 10px" },
      el("div", { class: "chart-title", style: "margin:0" }, "Drift"),
      driftCheckButton(v)),
    drift
      ? el("div", {},
          el("div", { style: "margin-bottom:8px" }, statusBadge(drift.outcome)),
          el("div", { class: "muted", style: "margin-bottom:8px" },
            `method: ${drift.method} · score: ${fmt.metric(drift.score)}`),
          driftFeatures.length
            ? el("div", { class: "task-grid" },
                ...driftFeatures
                  .filter((f) => f.drift_detected)
                  .map((f) =>
                    el("div", { class: "task-cell failed" },
                      el("span", { class: "dot" }), f.feature,
                      el("span", { class: "state" }, f.method))))
            : null)
      : el("div", { class: "muted" }, "Not evaluated yet."));

  return el("section", {},
    el("div", { class: "section-head" },
      el("h3", {}, `Version ${v.version_number}`),
      el("div", { class: "row-actions" },
        el("span", { class: "faint" }, fmt.ago(v.created_at)),
        trainNowButton(v))),
    el("div", { class: "grid-2" },
      el("div", { class: "card" }, facts),
      el("div", {},
        readinessPanel,
        el("div", { style: "height:16px" }), driftPanel,
        classBalance ? el("div", { style: "height:16px" }) : null, classBalance)),
    schemaRows.length
      ? el("div", {},
          el("h3", {}, `Schema — ${schemaRows.length} columns`),
          el("div", { class: "table-wrap" },
            el("table", {},
              el("thead", {}, el("tr", {}, el("th", {}, "Column"), el("th", {}, "Dtype"))),
              el("tbody", {}, ...schemaRows))))
      : null);
}

// Two versions' facts/schema side by side, differing rows marked ●
// (same convention as initRunsCompare) — rendered inline in ds-body,
// no navigation, so picking a different pair is a couple of clicks away.
async function datasetCompareSection(vA, vB) {
  let readinessA = null, driftA = null, readinessB = null, driftB = null;
  try { readinessA = await api(`/readiness/${vA.id}`); } catch { /* optional */ }
  try { driftA = await api(`/drift/${vA.id}`); } catch { /* optional */ }
  try { readinessB = await api(`/readiness/${vB.id}`); } catch { /* optional */ }
  try { driftB = await api(`/drift/${vB.id}`); } catch { /* optional */ }

  const metaA = vA.metadata || {};
  const metaB = vB.metadata || {};
  const differs = (a, b) => JSON.stringify(a ?? null) !== JSON.stringify(b ?? null);
  const colA = `Version ${vA.version_number}`;
  const colB = `Version ${vB.version_number}`;

  const fields = [
    ["Rows", vA.row_count, vB.row_count, fmt.num],
    ["Storage URI", vA.storage_uri, vB.storage_uri, (x) => x || "—"],
    ["Content SHA-256", metaA.content_sha256, metaB.content_sha256, (x) => x || "not recorded"],
    ["Schema hash", vA.schema_hash, vB.schema_hash, (x) => x],
    ["Version checksum", vA.checksum, vB.checksum, (x) => x],
    ["Size", metaA.size_bytes, metaB.size_bytes, fmt.bytes],
    ["Immutable", vA.is_immutable, vB.is_immutable, (x) => (x ? "yes" : "no")],
    ["Created", vA.created_at, vB.created_at, fmt.time],
    ["Readiness", readinessA?.status, readinessB?.status, (x) => x || "not evaluated"],
    ["Drift outcome", driftA?.outcome, driftB?.outcome, (x) => x || "not evaluated"],
  ];
  const ordered = [...fields.filter(([, a, b]) => differs(a, b)), ...fields.filter(([, a, b]) => !differs(a, b))];

  const overview = el("div", {},
    el("h3", {}, "Overview — ● marks a value that differs"),
    el("div", { class: "table-wrap" },
      el("table", {},
        el("thead", {}, el("tr", {}, el("th", {}, "Field"), el("th", {}, colA), el("th", {}, colB))),
        el("tbody", {}, ...ordered.map(([label, a, b, f]) =>
          el("tr", {},
            el("td", { class: "mono" }, label, differs(a, b) ? el("span", { class: "faint" }, " ●") : null),
            el("td", {}, f(a)),
            el("td", {}, f(b))))))));

  const classFields = [
    ["Positive (fraud) count", metaA.n_fraud, metaB.n_fraud, fmt.num],
    ["Positive ratio", metaA.fraud_ratio, metaB.fraud_ratio, fmt.pct],
    ["Missing values", metaA.missing_values, metaB.missing_values, fmt.num],
  ];
  const classBalance = (metaA.n_fraud != null || metaB.n_fraud != null)
    ? el("div", {},
        el("h3", {}, "Class balance"),
        el("div", { class: "table-wrap" },
          el("table", {},
            el("thead", {}, el("tr", {}, el("th", {}, "Metric"), el("th", {}, colA), el("th", {}, colB))),
            el("tbody", {}, ...classFields.map(([label, a, b, f]) =>
              el("tr", {},
                el("td", { class: "mono" }, label, differs(a, b) ? el("span", { class: "faint" }, " ●") : null),
                el("td", {}, f(a)),
                el("td", {}, f(b))))))))
    : null;

  // Every column seen on either side; one present on only one side
  // renders "—" on the other, which reads the same as an added/removed
  // column would in a diff.
  const colsA = new Map((metaA.columns || []).map((c) => [c.name, c.dtype]));
  const colsB = new Map((metaB.columns || []).map((c) => [c.name, c.dtype]));
  const allCols = [...new Set([...colsA.keys(), ...colsB.keys()])].sort();
  const schemaDiff = allCols.length
    ? el("div", {},
        el("h3", {}, `Schema — ${allCols.length} columns · ● marks added, removed, or changed`),
        el("div", { class: "table-wrap" },
          el("table", {},
            el("thead", {}, el("tr", {}, el("th", {}, "Column"), el("th", {}, colA), el("th", {}, colB))),
            el("tbody", {}, ...allCols.map((name) => {
              const a = colsA.get(name), b = colsB.get(name);
              return el("tr", {},
                el("td", { class: "mono" }, name, a !== b ? el("span", { class: "faint" }, " ●") : null),
                el("td", { class: "mono muted" }, a ?? "—"),
                el("td", { class: "mono muted" }, b ?? "—"));
            })))))
    : null;

  return el("div", {}, overview, classBalance, schemaDiff);
}

async function initDatasetDetail(id) {
  const head = document.getElementById("ds-head");
  const body = document.getElementById("ds-body");
  const toolbar = document.getElementById("ds-toolbar");
  const versionSelect = document.getElementById("version-select");
  const compareLabel = document.getElementById("compare-label");
  const compareVs = document.getElementById("compare-vs");
  const compareA = document.getElementById("compare-a");
  const compareB = document.getElementById("compare-b");
  const compareBtn = document.getElementById("compare-btn");
  const compareClose = document.getElementById("compare-close");

  let versions;
  try {
    const [ds, vs] = await Promise.all([
      api(`/datasets/${id}`),
      api(`/datasets/${id}/versions`),
    ]);
    versions = vs;
    head.replaceChildren(
      el("div", { class: "breadcrumb" }, el("a", { href: "/datasets" }, "Datasets"), " / ", ds.name),
      el("h2", {}, ds.name),
      el("p", { class: "subtitle" }, ds.description || "No description"));
  } catch (e) {
    setError(body, e);
    return;
  }

  if (!versions.length) {
    mount(body, banner("No versions registered yet."));
    return;
  }

  // Newest first — a reader lands on a dataset's page to see what
  // changed most recently, not to scroll past every prior version to
  // reach it (which is what showing all of them, oldest-first-reversed,
  // used to do here).
  const newest = versions.slice().reverse();
  const label = (v) => `Version ${v.version_number} — ${fmt.ago(v.created_at)}`;
  const byId = new Map(versions.map((v) => [String(v.id), v]));
  const optionList = () => newest.map((v) => el("option", { value: String(v.id) }, label(v)));

  versionSelect.replaceChildren(...optionList());
  compareA.replaceChildren(...optionList());
  compareB.replaceChildren(...optionList());
  // Deep link: /datasets/{id}?version={dataset_version_id}, used by
  // "Dataset version #N" links elsewhere (run detail, dataset-inputs
  // card) that know a version id but not which of its own versions is
  // "current" — falls back to newest for a plain /datasets/{id} visit
  // or an id that doesn't belong to this dataset.
  const requestedVersion = new URLSearchParams(location.search).get("version");
  const initialId = requestedVersion && byId.has(requestedVersion) ? requestedVersion : String(newest[0].id);
  versionSelect.value = initialId;
  compareA.value = String(newest[0].id);
  compareB.value = String((newest[1] || newest[0]).id);
  toolbar.hidden = false;

  const hasCompare = versions.length > 1;
  for (const e of [compareLabel, compareVs, compareA, compareB, compareBtn]) e.hidden = !hasCompare;

  async function showVersion(versionId) {
    const v = byId.get(String(versionId));
    if (!v) return;
    mount(body, el("div", { class: "muted" }, "Loading…"));
    mount(body, await datasetVersionSection(v));
  }

  versionSelect.addEventListener("change", () => {
    compareClose.hidden = true;
    showVersion(versionSelect.value);
  });

  compareBtn.addEventListener("click", async () => {
    if (compareA.value === compareB.value) {
      mount(body, banner("Pick two different versions to compare."));
    } else {
      mount(body, el("div", { class: "muted" }, "Loading…"));
      mount(body, await datasetCompareSection(byId.get(compareA.value), byId.get(compareB.value)));
    }
    compareClose.hidden = false;
    versionSelect.disabled = true;
  });

  compareClose.addEventListener("click", () => {
    versionSelect.disabled = false;
    compareClose.hidden = true;
    showVersion(versionSelect.value);
  });

  await showVersion(versionSelect.value);
}

/* ------------------------------------------------------------------ */
/* Runs — Airflow-flavoured list with MLflow-style comparison          */
/* ------------------------------------------------------------------ */

// Runs page: the framework's own Training runs by default, or — once
// an experiment is picked — MLflow's own ranked view of that
// experiment's runs (including ones this framework never started).
// One page, one nav entry; see the module-level note above app.js's
// history for why these used to be two ("Training runs" vs
// "Experiments") and aren't anymore.
async function initRuns() {
  const out = document.getElementById("runs-out");
  const statusFilter = document.getElementById("status-filter");
  const search = document.getElementById("search");
  const compareBtn = document.getElementById("compare-btn");
  const expFilter = document.getElementById("experiment-filter");
  const rankByLabel = document.getElementById("rank-by-label");
  const rankBy = document.getElementById("rank-by");
  const rankDir = document.getElementById("rank-dir");
  const filterInput = document.getElementById("filter-string");
  const selected = new Set();

  function updateCompare() {
    compareBtn.disabled = selected.size < 2;
    compareBtn.textContent = selected.size
      ? `Compare ${selected.size} run${selected.size > 1 ? "s" : ""}`
      : "Compare runs";
  }
  compareBtn.addEventListener("click", () => {
    location.href = `/runs/compare?ids=${[...selected].join(",")}`;
  });

  // Populate the experiment picker, best-effort — MLflow may be down,
  // in which case this page just stays in "framework" mode. Pre-select
  // from ?experiment=<id> so a link from elsewhere (a run's provenance
  // card, an old /experiments/{id} bookmark — see the redirect in
  // mount.py) lands directly in that experiment's own view.
  const preselected = new URLSearchParams(location.search).get("experiment") || "";
  try {
    const p = await api("/mlflow/experiments");
    if (p.available) {
      const experiments = (p.data.experiments || []).filter((e) => e.lifecycle_stage === "active");
      expFilter.append(...experiments.map((e) => el("option", { value: e.experiment_id }, e.name)));
      if (preselected && experiments.some((e) => e.experiment_id === preselected)) {
        expFilter.value = preselected;
      }
    }
  } catch { /* the picker is a bonus, not a requirement */ }

  function setMode(isExperiment) {
    statusFilter.hidden = isExperiment;
    rankByLabel.hidden = !isExperiment;
    rankBy.hidden = !isExperiment;
    rankDir.hidden = !isExperiment;
    filterInput.hidden = !isExperiment;
  }

  let metricsSeen = null; // populated once per experiment selection

  async function loadFrameworkRuns() {
    let all;
    try {
      all = await api("/training-runs?limit=500");
    } catch (e) {
      setError(out, e);
      return;
    }
    const q = (search.value || "").toLowerCase();
    const st = statusFilter.value;
    const rows = all.filter((r) =>
      (!st || r.status === st) &&
      (!q || `${r.id} ${r.pipeline_id || ""} ${r.orchestrator || ""}`.toLowerCase().includes(q)));

    const maxDur = Math.max(...rows.map((r) => r.duration_seconds || 0), 1);
    const table = el("table", {}, el("thead", {}, el("tr", {})), el("tbody", {}));
    out.replaceChildren(el("div", { class: "table-wrap" }, table));

    makeSortable(table, rows,
      [
        { label: "" },
        { label: "Run", sort: (r) => r.id },
        { label: "Status", sort: (r) => r.status },
        { label: "Pipeline", sort: (r) => r.pipeline_id },
        { label: "Orchestrator" },
        { label: "Trigger", sort: (r) => r.trigger_type },
        { label: "Duration", sort: (r) => r.duration_seconds },
        { label: "Key metric", sort: (r) => bestMetric(r)?.value },
        { label: "Started", sort: (r) => r.started_at },
      ],
      (r) => {
        const best = bestMetric(r);
        const cb = el("input", { type: "checkbox" });
        cb.checked = selected.has(r.id);
        cb.addEventListener("change", () => {
          if (cb.checked) selected.add(r.id); else selected.delete(r.id);
          updateCompare();
        });
        return el("tr", {},
          el("td", { class: "checkbox-cell" }, cb),
          el("td", {}, el("a", { href: `/runs/${r.id}` }, `#${r.id}`)),
          el("td", {}, statusBadge(r.status)),
          el("td", {},
            cellWithSub(
              el("span", { class: "mono truncate", style: "display:block", title: r.pipeline_id || "" },
                r.pipeline_id || "—"),
              r.execution_id)),
          el("td", { class: "muted" }, r.orchestrator || "—"),
          el("td", { class: "muted" }, r.trigger_type || "—"),
          el("td", {},
            el("div", { style: "display:flex;align-items:center;gap:8px" },
              el("span", { class: "mono nowrap", style: "min-width:56px" }, fmt.dur(r.duration_seconds)),
              el("div", { class: "bar-track", style: "width:70px" },
                el("div", {
                  class: `bar-fill ${statusKind(r.status)}`,
                  style: `width:${((r.duration_seconds || 0) / maxDur) * 100}%`,
                })))),
          // "mono", not "num" — this table's headers are all left-aligned.
          el("td", { class: "mono" }, best ? `${best.name} ${fmt.metric(best.value)}` : "—"),
          el("td", { class: "muted nowrap" }, fmt.ago(r.started_at || r.created_at)));
      });
    updateCompare();
  }

  async function loadExperimentRuns(experimentId) {
    out.replaceChildren(el("div", { class: "card muted" }, "Loading…"));

    // Framework runs carry the MLflow run id, so a leaderboard row can
    // point back at the run that produced it; runs with none link to
    // the bare-MLflow run-detail page instead (see initMlflowRunDetail).
    let byMlflowId = new Map();
    try {
      const runs = await api("/training-runs?limit=500");
      byMlflowId = new Map(runs.filter((r) => r.mlflow_run_id).map((r) => [r.mlflow_run_id, r.id]));
    } catch { /* the cross-link is a bonus, not a requirement */ }

    const params = new URLSearchParams({ limit: "100", direction: rankDir.value });
    if (rankBy.value) params.set("order_by", rankBy.value);
    if (filterInput.value.trim()) params.set("filter_string", filterInput.value.trim());

    let p;
    try {
      p = await api(`/mlflow/experiments/${encodeURIComponent(experimentId)}/runs?${params}`);
    } catch (e) {
      setError(out, e);
      return;
    }
    if (!p.available) {
      out.replaceChildren(banner(p.reason, "warn"));
      return;
    }

    const runs = p.data.runs || [];
    const metricKeys = [...new Set(runs.flatMap((r) => Object.keys(r.metrics || {})))].sort();

    // Populate the rank-by choices once per experiment, from what its
    // runs actually have.
    if (metricsSeen === null && metricKeys.length) {
      metricsSeen = metricKeys;
      rankBy.replaceChildren(
        el("option", { value: "" }, "start time"),
        ...metricKeys.map((k) => el("option", { value: k }, k)));
      const preferred = METRIC_PRIORITY.find((m) => metricKeys.includes(m));
      if (preferred) {
        rankBy.value = preferred;
        await loadExperimentRuns(experimentId);
        return;
      }
    }

    const shown = metricKeys.filter((k) => METRIC_PRIORITY.includes(k) || k === rankBy.value);
    const best = {};
    for (const k of shown) {
      best[k] = Math.max(...runs.map((r) => (r.metrics || {})[k] ?? -Infinity));
    }

    const table = el("table", {},
      el("thead", {}, el("tr", {},
        el("th", {}, ""), el("th", {}, "#"), el("th", {}, "Run"), el("th", {}, "Status"),
        ...shown.map((k) => el("th", {}, k)),
        el("th", {}, "Training run"), el("th", {}, "Started"))),
      el("tbody", {}, ...(runs.length ? runs.map((r, i) => {
        const fwId = byMlflowId.get(r.run_id);
        // Compare only knows framework training runs (initRunsCompare fetches
        // /training-runs/{id}) — a run this framework never started has no
        // such id to select, so its checkbox cell stays empty rather than
        // disabled-and-confusing.
        const cb = fwId ? el("input", { type: "checkbox" }) : null;
        if (cb) {
          cb.checked = selected.has(fwId);
          cb.addEventListener("change", () => {
            if (cb.checked) selected.add(fwId); else selected.delete(fwId);
            updateCompare();
          });
        }
        return el("tr", {},
          el("td", { class: "checkbox-cell" }, cb),
          // "mono", not "num" — this table's headers are all left-aligned.
          el("td", { class: "mono" }, String(i + 1)),
          el("td", {}, el("a", {
            class: "mono", title: r.run_id,
            href: fwId ? `/runs/${fwId}` : `/mlflow-runs/${encodeURIComponent(r.run_id)}`,
          }, r.run_name || r.run_id.slice(0, 8))),
          el("td", {}, statusBadge(r.status)),
          ...shown.map((k) => {
            const v = (r.metrics || {})[k];
            const isBest = typeof v === "number" && v === best[k] && runs.length > 1;
            // "mono", not "num" — this table's headers are all left-aligned.
            return el("td", { class: "mono" },
              isBest ? el("strong", { style: "color:var(--ok)" }, fmt.metric(v)) : fmt.metric(v));
          }),
          el("td", {}, fwId ? el("a", { href: `/runs/${fwId}` }, `#${fwId}`)
                            : el("span", { class: "faint" }, "—")),
          el("td", { class: "muted nowrap" }, fmt.ago(r.start_time)));
      }) : [emptyRow(shown.length + 6, "No runs matched.")])));
    updateCompare();

    const headline = rankBy.value || METRIC_PRIORITY.find((m) => shown.includes(m));
    const chart = headline && runs.length > 1
      ? barChart(`${headline} by run`, runs.slice(0, 12).map((r) => ({
          label: r.run_name || r.run_id.slice(0, 6),
          value: (r.metrics || {})[headline] ?? 0,
        })))
      : null;

    mount(out,
      el("p", { class: "muted", style: "margin:0 0 10px" },
        `${runs.length} run${runs.length === 1 ? "" : "s"} · ordered by ${p.data.order_by}`),
      el("div", { class: "table-wrap" }, table),
      chart ? el("div", { style: "margin-top:16px;max-width:560px" }, chart) : null);
  }

  async function load() {
    if (expFilter.value) {
      setMode(true);
      await loadExperimentRuns(expFilter.value);
    } else {
      setMode(false);
      await loadFrameworkRuns();
    }
  }

  expFilter.addEventListener("change", () => {
    metricsSeen = null;
    const url = new URL(location);
    if (expFilter.value) url.searchParams.set("experiment", expFilter.value);
    else url.searchParams.delete("experiment");
    history.replaceState(null, "", url);
    load();
  });
  statusFilter.addEventListener("change", load);
  search.addEventListener("input", load);
  document.getElementById("refresh").addEventListener("click", load);
  rankBy.addEventListener("change", load);
  rankDir.addEventListener("change", load);
  filterInput.addEventListener("keydown", (e) => { if (e.key === "Enter") load(); });

  await load();
}

// Pick the metric worth showing in a list. Ordered by how much it says
// about an imbalanced classifier, which is what this framework trains.
const METRIC_PRIORITY = ["average_precision", "f1", "roc_auc", "recall", "precision", "accuracy"];

function bestMetric(run) {
  const m = run.metrics || {};
  for (const name of METRIC_PRIORITY) {
    if (typeof m[name] === "number") return { name, value: m[name] };
  }
  const first = Object.entries(m).find(([, v]) => typeof v === "number");
  return first ? { name: first[0], value: first[1] } : null;
}

// The console has no login, so there is nothing to attach a write token
// to but the browser tab: cached in sessionStorage (cleared with the
// tab, never sent anywhere but this origin) so a user fixing several
// tasks in one sitting is only prompted once. Not real auth — see
// api/security.py's module docstring — just where the one shared secret
// that gate expects has to come from on this side.
function writeToken(forcePrompt = false) {
  if (forcePrompt) sessionStorage.removeItem("gateflow-write-token");
  let t = sessionStorage.getItem("gateflow-write-token");
  if (!t) {
    t = window.prompt(
      "Credential for actions that change state.\n\n" +
      "An API key (mlops_ak_…) is preferred — it names you in the audit " +
      "trail. CONSOLE_WRITE_TOKEN also works, but records every action " +
      "as \"system\"."
    );
    if (t) sessionStorage.setItem("gateflow-write-token", t.trim());
  }
  return t || null;
}

// An mlops_ak_ key goes in Authorization: Bearer; anything else is
// assumed to be the legacy shared secret. Sniffing the prefix rather
// than asking the user which kind they pasted — the prefix exists
// precisely so a key is self-identifying (see auth/manager.py).
function writeAuthHeaders(token) {
  return token.startsWith("mlops_ak_")
    ? { Authorization: `Bearer ${token}` }
    : { "X-Console-Token": token };
}

// api() plus the write token, for any gated endpoint (see
// api/security.py). On a 401/403 the cached token is dropped and the
// prompt fires once more — covers both "never set one" and "operator
// rotated CONSOLE_WRITE_TOKEN since last time". A 503 is passed straight
// through instead: that means the server has no token configured at all,
// which re-prompting cannot fix and the user needs to read.
async function apiWrite(path, options = {}, _retried = false) {
  const token = writeToken();
  if (!token) throw new Error("Write token required — action cancelled.");
  try {
    return await api(path, {
      ...options,
      headers: { ...(options.headers || {}), ...writeAuthHeaders(token) },
    });
  } catch (e) {
    if (!_retried && /^40[13]/.test(e.message)) {
      writeToken(true);
      return apiWrite(path, options, true);
    }
    throw e;
  }
}

// POSTs to the gated clear/retry route (see airflow_views.py).
async function taskAction(runId, taskId, action) {
  return apiWrite(`/training-runs/${runId}/tasks/${encodeURIComponent(taskId)}/${action}`, {
    method: "POST",
  });
}

// Fetches one task attempt's log and renders it as plain text below the
// task grid, plus Clear/Retry buttons scoped to that one task — see
// taskAction() and airflow_views.py's clear_task/retry_task routes. Not
// an api()-wrapped call for the log fetch itself: the endpoint answers
// with the log body directly (mirroring how mlflow_views.get_run_artifact
// serves raw bytes), including a 200 whose *body* is Airflow's own error
// message when it could not reach where the log actually lives — that
// text is shown as-is, since it is the accurate answer, not a failure to
// hide.
function showTaskLog(host, runId, taskId, tryNumber) {
  const msg = el("span", { class: "faint" });
  const clearBtn = el("button", { class: "btn" }, "Clear");
  const retryBtn = el("button", { class: "btn" }, "Retry");

  async function run(action, btn) {
    btn.disabled = true;
    msg.textContent = "Working…";
    try {
      const result = await taskAction(runId, taskId, action);
      msg.textContent = `${action === "clear" ? "Cleared" : "Retried"} — ` +
        `${result.cleared_task_instances} task instance(s) reset. Reload the page to see the updated state.`;
    } catch (e) {
      msg.textContent = `Failed: ${e.message}`;
    } finally {
      btn.disabled = false;
    }
  }
  clearBtn.addEventListener("click", () => run("clear", clearBtn));
  retryBtn.addEventListener("click", () => run("retry", retryBtn));

  host.replaceChildren(
    el("div", { class: "section-head", style: "margin-top:12px" },
      el("div", { class: "chart-title" }, `Log — ${taskId} (attempt ${tryNumber})`),
      clearBtn, retryBtn, msg),
    el("pre", { class: "log" }, "Loading…"));
  const pre = host.querySelector("pre");
  fetch(`${API}/training-runs/${runId}/tasks/${encodeURIComponent(taskId)}/log?try_number=${tryNumber}`)
    .then((r) => r.text().then((text) => ({ ok: r.ok, status: r.status, text })))
    .then(({ ok, status, text }) => {
      pre.textContent = ok ? (text || "(empty log)") : `${status}: ${text || "could not load log"}`;
    })
    .catch(() => { pre.textContent = "Could not load log."; });
}

// PENDING/RUNNING runs are the only ones worth a live connection —
// listRuns()'s own polling of the whole table stays out of scope here
// (this is a single-run detail page); a run already terminal on load
// gets a normal one-shot fetch, no EventSource at all.
const LIVE_RUN_STATUSES = new Set(["PENDING", "RUNNING"]);

async function initRunDetail(id) {
  const head = document.getElementById("run-head");
  const body = document.getElementById("run-body");
  let run;
  try {
    run = await api(`/training-runs/${id}`);
  } catch (e) {
    setError(head, e);
    return;
  }

  renderRunDetail(head, body, id, run);

  if (LIVE_RUN_STATUSES.has(run.status)) {
    subscribeToRunEvents(id, async (status) => {
      // The SSE payload already carries the framework row; re-fetching
      // isn't required for the status itself, but every other panel
      // this page renders (metrics, the Airflow task grid, MLflow
      // curves) is easiest to keep correct by re-running the exact same
      // render path a manual reload would take, rather than hand-
      // patching a dozen DOM fragments in two places that could drift
      // apart. One extra GET per transition is cheap next to that.
      let fresh;
      try {
        fresh = await api(`/training-runs/${id}`);
      } catch {
        return;
      }
      renderRunDetail(head, body, id, fresh);
    });
  }
}

// EventSource wrapper for one run's SSE stream (see
// api/routers/runs.py::stream_run_events). Calls `onStatus(status)` for
// every "status" event the server sends and closes the connection itself
// once the server says the run reached a terminal state or the stream
// times out — nothing here needs to inspect the payload to know when to
// stop, the server's own close is authoritative.
function subscribeToRunEvents(id, onStatus) {
  const source = new EventSource(`${API}/training-runs/${id}/events`);
  source.addEventListener("status", (e) => {
    const data = JSON.parse(e.data);
    onStatus(data.status);
    if (!LIVE_RUN_STATUSES.has(data.status)) source.close();
  });
  source.addEventListener("timeout", () => source.close());
  source.addEventListener("error", () => source.close());
  // A dropped connection (server restart, network blip) falls back to
  // native EventSource reconnect for anything transient; nothing to do
  // here beyond letting that default behaviour run.
}

// A run (or an MLflow dataset-input entry) knows a dataset_version_id
// but not which Dataset it belongs to — that takes a lookup. Used to
// link straight to "which version is this exactly", so renders the
// "#N" label immediately and swaps in the real link (to that version's
// own page — /datasets/{dataset_id}?version={id}, see the deep-link
// support in initDatasetDetail) once the lookup resolves, best-effort,
// same pattern as renderRegistrySummary's fetch. Previously these
// linked straight to the Lineage graph instead, which doesn't carry
// this version's own facts (checksum, row count, schema) at all — the
// wrong destination for "go see this dataset version".
function datasetVersionLink(versionId) {
  const label = el("span", {}, `#${versionId}`);
  api(`/dataset-versions/${versionId}`).then((dv) => {
    mount(label, el("a", { href: `/datasets/${dv.dataset_id}?version=${dv.id}` }, `#${dv.id}`));
  }).catch(() => { /* link just stays plain text */ });
  return label;
}

function renderRunDetail(head, body, id, run) {
  head.replaceChildren(
    el("div", { class: "breadcrumb" }, el("a", { href: "/runs" }, "Runs"), " / ", `#${run.id}`),
    el("h2", {}, `Training run #${run.id} `, statusBadge(run.status),
      LIVE_RUN_STATUSES.has(run.status)
        ? el("span", { class: "faint", style: "font-size:12px;margin-left:8px" }, "● live")
        : null),
    el("p", { class: "subtitle mono" }, run.pipeline_id || "no pipeline"));

  const sections = [];

  if (run.error_message) {
    sections.push(el("div", {},
      el("h3", {}, "Failure"),
      el("pre", { class: "log" }, run.error_message)));
  }

  // Training itself can succeed while a side-effect (logging to MLflow,
  // typically an S3/MinIO credentials issue on the artifact upload) fails —
  // that must not stay buried in a worker's stdout. See train_xgboost() in
  // case_studies/fraud_detection/pipelines.py.
  const mlflowWarning = run.metadata?.orchestrator_result?.mlflow_logging_warning;
  if (mlflowWarning) {
    sections.push(el("div", {},
      el("h3", {}, "MLflow logging warning"),
      el("p", { class: "muted" }, "Training succeeded; logging the run to MLflow did not."),
      el("pre", { class: "log" }, mlflowWarning)));
  }

  sections.push(el("div", { class: "grid-2" },
    el("div", { class: "card" },
      el("div", { class: "chart-title" }, "Run"),
      el("dl", { class: "kv" },
        el("dt", {}, "Status"), el("dd", {}, run.status),
        el("dt", {}, "Trigger"), el("dd", {}, run.trigger_type || "—"),
        el("dt", {}, "Orchestrator"), el("dd", {}, run.orchestrator || "—"),
        el("dt", {}, "Execution id"), el("dd", {}, run.execution_id || "—"),
        el("dt", {}, "Dataset version"), el("dd", {},
          run.dataset_version_id ? datasetVersionLink(run.dataset_version_id) : "—"),
        el("dt", {}, "MLflow run"), el("dd", {}, run.mlflow_run_id || "—"),
        el("dt", {}, "Started"), el("dd", {}, fmt.time(run.started_at)),
        el("dt", {}, "Completed"), el("dd", {}, fmt.time(run.completed_at)),
        el("dt", {}, "Duration"), el("dd", {}, fmt.dur(run.duration_seconds)))),
    el("div", { class: "card" },
      el("div", { class: "chart-title" }, "Parameters"),
      Object.keys(run.parameters || {}).length
        ? el("dl", { class: "kv" },
            ...Object.entries(run.parameters).flatMap(([k, v]) => [
              el("dt", {}, k), el("dd", {}, String(v))]))
        : el("div", { class: "muted" }, "No parameters recorded."))));

  const metrics = run.metrics || {};
  if (Object.keys(metrics).length) {
    const best = bestMetric(run);
    sections.push(el("div", {},
      el("h3", {}, "Metrics"),
      el("div", { class: "metric-grid" },
        ...Object.entries(metrics).map(([k, v]) =>
          el("div", { class: `metric ${best && best.name === k ? "best" : ""}` },
            el("div", { class: "name" }, k),
            el("div", { class: "val" }, fmt.metric(v)))))));
  }

  // Airflow: the DAG run's own state/dates/conf, plus a full task grid —
  // only rendered when the run actually ran there. Each task cell opens
  // its log on click; run.execution_id is already shown in the Run card
  // above, so this panel does not repeat it.
  const tasksPanel = el("div", {});
  sections.push(tasksPanel);
  api(`/training-runs/${id}/tasks`).then((p) => {
    if (!p.available) {
      if (run.orchestrator === "AirflowOrchestrator") {
        tasksPanel.replaceChildren(el("h3", {}, "Airflow"), banner(p.reason, "warn"));
      }
      return;
    }
    const dagRun = p.data.dag_run || {};
    const tasks = p.data.tasks || [];
    const logHost = el("div", {});

    const cells = tasks.map((t) => {
      const isRetry = t.try_number != null && t.try_number > 1;
      const cell = el("div", {
        class: `task-cell ${statusKind(t.state)}`,
        role: "button",
        tabindex: "0",
        title: "View log",
      },
        el("span", { class: "dot" }),
        el("div", {},
          el("div", {}, t.task_id,
            isRetry ? el("span", { class: "retry-badge" },
              `retry ${t.try_number}${t.max_tries ? `/${t.max_tries + 1}` : ""}`) : null),
          el("div", { class: "state" }, [
            t.state,
            t.duration != null ? fmt.dur(t.duration) : null,
            t.hostname || null,
          ].filter(Boolean).join(" · "))));
      const open = () => showTaskLog(logHost, id, t.task_id, t.try_number || 1);
      cell.addEventListener("click", open);
      cell.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
      return cell;
    });

    const confEntries = Object.entries(dagRun.conf || {});
    tasksPanel.replaceChildren(
      el("h3", {}, "Airflow"),
      el("div", { class: "card" },
        el("dl", { class: "kv", style: "margin-bottom:14px" },
          el("dt", {}, "DAG run"), el("dd", {}, statusBadge(dagRun.state)),
          el("dt", {}, "Started"), el("dd", {}, fmt.time(dagRun.started_at)),
          el("dt", {}, "Finished"), el("dd", {}, fmt.time(dagRun.finished_at))),
        confEntries.length
          ? el("details", { style: "margin-bottom:14px" },
              el("summary", { class: "faint" }, "Run conf"),
              el("pre", { class: "log", style: "margin-top:8px" },
                JSON.stringify(dagRun.conf, null, 2)))
          : null,
        tasks.length
          ? el("div", { class: "task-grid" }, ...cells)
          : el("div", { class: "muted" }, "No task instances — the DAG has not been scheduled."),
        logHost));
  }).catch(() => {});

  // MLflow: provenance, training curves, artifacts, model signature. Each
  // panel is appended in place so a slow or missing tracking server never
  // holds up the rest of the page.
  const mlPanel = el("div", {});
  const nestedPanel = el("div", {});
  const artifactPanel = el("div", {});
  const modelPanel = el("div", {});
  sections.push(mlPanel, nestedPanel, artifactPanel, modelPanel);

  renderMlflowSummary(mlPanel, `/training-runs/${id}/mlflow`, run);

  if (run.mlflow_run_id) {
    const basePath = `/training-runs/${id}`;
    renderNestedRuns(nestedPanel, basePath);
    renderArtifacts(artifactPanel, basePath, "");
    renderModelInfo(modelPanel, basePath);
  }

  mount(body, ...sections);
}

// Provenance, training curves, system resources — the "MLflow" card on
// both the framework run-detail page (``url`` = /training-runs/{id}/mlflow)
// and the bare-MLflow run-detail page (``url`` = /mlflow/runs/{id}).
// ``run`` is the framework TrainingRun object when there is one, for
// datasetInputsCard's cross-link to framework lineage — null otherwise.
function renderMlflowSummary(host, url, run) {
  api(url).then((p) => {
    if (!p.available) {
      if (!run || run.mlflow_run_id) {
        host.replaceChildren(el("h3", {}, "MLflow"), banner(p.reason, "warn"));
      }
      return;
    }
    const d = p.data;
    const hist = d.history || {};
    const charts = Object.entries(hist)
      .filter(([, series]) => series.length > 1)
      .map(([name, series]) => lineChart(name, series));

    // The experiment id comes from the run itself. It used to be hardcoded
    // to 0, which is wrong for every run outside the Default experiment.
    const expId = d.info?.experiment_id;
    const deepLink = `${d.tracking_uri}/#/experiments/${expId}/runs/${d.mlflow_run_id}`;

    // Resource usage is a different question from model quality, so it
    // gets its own row rather than being mixed into the training charts.
    const sysCharts = Object.entries(d.system_history || {})
      .filter(([, series]) => series.length > 1)
      .map(([name, series]) => lineChart(name.replace(/^system\//, ""), series));

    mount(host,
      el("div", { class: "section-head" },
        el("h3", {}, "MLflow"),
        el("a", { class: "faint", href: deepLink, target: "_blank", rel: "noopener" },
          "open in MLflow ↗")),
      provenanceCard(d),
      datasetInputsCard(d.dataset_inputs, run),
      charts.length
        ? el("div", { class: "grid-3", style: "margin-top:16px" }, ...charts)
        : el("div", { class: "card", style: "margin-top:16px" },
            el("div", { class: "muted" },
              "Metrics were logged once, so there is no series to plot. " +
              "Values are shown above."),
            el("dl", { class: "kv", style: "margin-top:10px" },
              ...Object.entries(d.metrics || {}).flatMap(([k, v]) => [
                el("dt", {}, k), el("dd", {}, fmt.metric(v))]))),
      sysCharts.length
        ? el("div", {},
            el("div", { class: "chart-title", style: "margin:20px 0 8px" },
              "System resources during the run"),
            el("div", { class: "grid-3" }, ...sysCharts))
        : null);
  }).catch(() => {});
}

// The bare-MLflow-run detail page: everything initRunDetail() shows
// under its "MLflow" heading, for a run this framework has no
// TrainingRun row for at all — reached from /runs?experiment={id}'s
// leaderboard, for whichever rows have no "Training run" cross-link
// (see initRuns()'s loadExperimentRuns()).
async function initMlflowRunDetail(mlflowRunId) {
  const head = document.getElementById("run-head");
  const body = document.getElementById("run-body");
  const basePath = `/mlflow/runs/${encodeURIComponent(mlflowRunId)}`;

  head.replaceChildren(
    el("div", { class: "breadcrumb" },
      el("a", { href: "/runs" }, "Runs"), " / ", mlflowRunId.slice(0, 12)),
    el("h2", {}, "MLflow run ", el("span", { class: "mono" }, mlflowRunId)),
    el("p", { class: "subtitle" },
      "Not a training run this framework started — no TrainingRun record, just what MLflow itself holds."));

  const mlPanel = el("div", {});
  const nestedPanel = el("div", {});
  const artifactPanel = el("div", {});
  const modelPanel = el("div", {});
  mount(body, mlPanel, nestedPanel, artifactPanel, modelPanel);

  renderMlflowSummary(mlPanel, basePath, null);
  renderNestedRuns(nestedPanel, basePath);
  renderArtifacts(artifactPanel, basePath, "");
  renderModelInfo(modelPanel, basePath);
}

/* ------------------------------------------------------------------ */
/* MLflow panels on the run detail page                                */
/* ------------------------------------------------------------------ */

// Tags MLflow sets itself. Shown as named fields rather than raw keys.
const PROVENANCE_FIELDS = [
  ["mlflow.runName", "Run name"],
  ["mlflow.user", "User"],
  ["mlflow.source.name", "Source"],
  ["mlflow.source.type", "Source type"],
  ["mlflow.source.git.commit", "Git commit"],
  ["mlflow.source.git.branch", "Git branch"],
];

function provenanceCard(d) {
  const tags = d.tags || {};
  const info = d.info || {};
  const rows = [];

  for (const [key, label] of PROVENANCE_FIELDS) {
    const v = tags[key];
    if (!v) continue;
    rows.push(el("dt", {}, label));
    rows.push(el("dd", { class: key.endsWith("commit") ? "mono" : "" }, v));
  }
  if (info.experiment_id != null) {
    rows.push(el("dt", {}, "Experiment"));
    rows.push(el("dd", {},
      el("a", { href: `/runs?experiment=${encodeURIComponent(info.experiment_id)}` },
        `#${info.experiment_id}`)));
  }
  // MLflow's own view of the run, which can disagree with the framework's
  // row — a run the framework recorded as SUCCESS may be FAILED here if the
  // process died after the framework wrote its status.
  for (const [key, label] of [
    ["status", "MLflow status"],
    ["lifecycle_stage", "Lifecycle"],
    ["user_id", "Logged by"],
  ]) {
    if (info[key]) {
      rows.push(el("dt", {}, label));
      rows.push(el("dd", {}, String(info[key])));
    }
  }
  if (info.start_time) {
    rows.push(el("dt", {}, "MLflow start"));
    rows.push(el("dd", {}, fmt.time(info.start_time)));
  }
  if (info.end_time) {
    rows.push(el("dt", {}, "MLflow end"));
    rows.push(el("dd", {}, fmt.time(info.end_time)));
  }
  if (info.artifact_uri) {
    rows.push(el("dt", {}, "Artifact URI"));
    rows.push(el("dd", { class: "mono" }, info.artifact_uri));
  }

  // Tags the user set themselves, which is where domain meaning lives.
  const custom = Object.entries(tags).filter(([k]) => !k.startsWith("mlflow."));
  for (const [k, v] of custom) {
    rows.push(el("dt", {}, k));
    rows.push(el("dd", {}, v));
  }

  return el("div", { class: "card" },
    el("div", { class: "chart-title" }, "Provenance"),
    rows.length
      ? el("dl", { class: "kv" }, ...rows)
      : el("div", { class: "muted" }, "No tags recorded for this run."));
}

// What a run declared it trained on, per mlflow.log_input. The digest is
// content-derived, so it is the field worth holding against the framework's
// own dataset-version checksum.
function datasetInputsCard(inputs, run) {
  if (!inputs || !inputs.length) return null;
  return el("div", { class: "card", style: "margin-top:16px" },
    el("div", { class: "chart-title" }, "Dataset inputs (MLflow)"),
    el("div", { class: "table-wrap", style: "box-shadow:none;border:none" },
      el("table", {},
        el("thead", {}, el("tr", {},
          el("th", {}, "Name"), el("th", {}, "Digest"), el("th", {}, "Source"),
          el("th", {}, "Context"))),
        el("tbody", {}, ...inputs.map((i) =>
          el("tr", {},
            el("td", { class: "mono" }, i.name || "—"),
            el("td", { class: "mono" }, i.digest || "—"),
            el("td", { class: "mono truncate", title: i.source || "" },
              `${i.source_type || "?"}${i.source ? " · " + i.source : ""}`),
            el("td", { class: "muted" },
              (i.tags || {})["mlflow.data.context"] || "—")))))),
    run && run.dataset_version_id
      ? el("p", { class: "faint", style: "margin:10px 0 0;font-size:12.5px" },
          "Framework lineage records dataset version ",
          datasetVersionLink(run.dataset_version_id),
          ". Compare the digest above against that version's checksum to " +
          "confirm the run trained on what the lineage claims.")
      : null);
}

// ``basePath`` is either `/training-runs/{id}` (framework run) or
// `/mlflow/runs/{id}` (bare MLflow run, no framework row) — see
// initRunDetail() and initMlflowRunDetail(), which share these three
// render functions rather than each keeping its own copy.
function renderNestedRuns(host, basePath) {
  api(`${basePath}/nested`).then((p) => {
    if (!p.available) return;  // the MLflow panel above already said why
    const d = p.data;
    if (!d.parent && !(d.children || []).length) return;  // a standalone run

    const metricKeys = [...new Set((d.children || [])
      .flatMap((c) => Object.keys(c.metrics || {})))]
      .filter((k) => !k.startsWith("system/"))
      .sort();
    const best = {};
    for (const k of metricKeys) {
      best[k] = Math.max(...d.children.map((c) => (c.metrics || {})[k] ?? -Infinity));
    }
    const paramKeys = [...new Set((d.children || [])
      .flatMap((c) => Object.keys(c.params || {})))].sort();

    host.replaceChildren(
      el("div", { class: "section-head" },
        el("h3", {}, "Sweep"),
        el("span", { class: "faint" },
          d.is_child
            ? `this run is one trial of ${d.parent?.run_name || "a parent run"}`
            : "this run is the parent of the trials below")),
      el("div", { class: "table-wrap" },
        el("table", {},
          el("thead", {}, el("tr", {},
            el("th", {}, "Trial"),
            ...paramKeys.map((k) => el("th", {}, k)),
            ...metricKeys.map((k) => el("th", {}, k)),
            el("th", {}, "Started"))),
          el("tbody", {}, ...d.children.map((c) =>
            el("tr", { style: c.is_self ? "background:var(--accent-soft)" : null },
              el("td", {},
                el("span", { class: "mono", title: c.run_id },
                  c.run_name || c.run_id.slice(0, 8)),
                c.is_self ? el("span", { class: "faint" }, "  ← this run") : null),
              ...paramKeys.map((k) => el("td", { class: "mono" }, (c.params || {})[k] ?? "—")),
              ...metricKeys.map((k) => {
                const v = (c.metrics || {})[k];
                const isBest = typeof v === "number" && v === best[k] && d.children.length > 1;
                // "mono", not "num" — this table's headers are all left-aligned.
                return el("td", { class: "mono" },
                  isBest ? el("strong", { style: "color:var(--ok)" }, fmt.metric(v))
                         : fmt.metric(v));
              }),
              el("td", { class: "muted nowrap" }, fmt.ago(c.start_time))))))));
  }).catch(() => {});
}

const IMAGE_RE = /\.(png|jpe?g|gif|svg|webp)$/i;
const TEXT_RE = /\.(txt|json|ya?ml|csv|md|log|cfg|ini|requirements)$/i;

// Renders one directory of a run's artifacts, and recurses on click. Kept
// as an explicit re-render rather than a tree widget: the API is already
// per-directory, so this matches what the server can answer in one call.
function renderArtifacts(host, basePath, path) {
  host.replaceChildren(el("h3", {}, "Artifacts"),
    el("div", { class: "card muted" }, "Loading…"));

  api(`${basePath}/artifacts?path=${encodeURIComponent(path)}`).then((p) => {
    if (!p.available) {
      host.replaceChildren(el("h3", {}, "Artifacts"), banner(p.reason, "warn"));
      return;
    }
    const entries = p.data.entries || [];
    const crumbs = el("div", { class: "breadcrumb" },
      el("a", { href: "#", onclick: "return false" }, "artifacts"));
    crumbs.firstChild.addEventListener("click", () => renderArtifacts(host, basePath, ""));
    let acc = "";
    for (const part of (path ? path.split("/") : [])) {
      acc = acc ? `${acc}/${part}` : part;
      const here = acc;
      crumbs.appendChild(document.createTextNode(" / "));
      const link = el("a", { href: "#" }, part);
      link.addEventListener("click", (e) => { e.preventDefault(); renderArtifacts(host, basePath, here); });
      crumbs.appendChild(link);
    }

    const rawUrl = (p2) =>
      `${API}${basePath}/artifacts/raw?path=${encodeURIComponent(p2)}`;

    const list = el("div", { class: "table-wrap" },
      el("table", {},
        el("thead", {}, el("tr", {},
          el("th", {}, "Name"), el("th", {}, "Size"), el("th", {}, ""))),
        el("tbody", {}, ...(entries.length ? entries.map((e) => {
          const nameCell = el("td", {});
          if (e.is_dir) {
            const a = el("a", { href: "#" }, `${e.name}/`);
            a.addEventListener("click", (ev) => {
              ev.preventDefault();
              renderArtifacts(host, basePath, e.path);
            });
            nameCell.appendChild(a);
          } else {
            nameCell.appendChild(el("span", { class: "mono" }, e.name));
          }
          return el("tr", {},
            nameCell,
            // "mono", not "num" — "Size" is left-aligned like "Name".
            el("td", { class: "mono" }, e.is_dir ? "—" : fmt.bytes(e.file_size)),
            el("td", {}, e.is_dir ? "" :
              el("a", { href: rawUrl(e.path), target: "_blank", rel: "noopener" }, "open")));
        }) : [emptyRow(3, "No artifacts in this directory.")]))));

    // Inline previews for the things people actually came to look at:
    // the confusion matrix and the pinned dependency list.
    const previews = [];
    for (const e of entries) {
      if (e.is_dir) continue;
      if (IMAGE_RE.test(e.name)) {
        previews.push(el("div", { class: "card" },
          el("div", { class: "chart-title" }, e.name),
          el("img", {
            src: rawUrl(e.path), alt: e.name, loading: "lazy",
            style: "max-width:100%;height:auto;display:block;border-radius:4px",
          })));
      } else if (TEXT_RE.test(e.name) && (e.file_size || 0) <= 64 * 1024) {
        const pre = el("pre", { class: "log" }, "Loading…");
        fetch(rawUrl(e.path))
          .then((r) => r.text())
          .then((t) => { pre.textContent = t; })
          .catch(() => { pre.textContent = "Could not load."; });
        previews.push(el("div", { class: "card" },
          el("div", { class: "chart-title" }, e.name), pre));
      }
    }

    mount(host,
      el("h3", {}, "Artifacts"),
      crumbs,
      list,
      previews.length
        ? el("div", { class: "grid-2", style: "margin-top:16px" }, ...previews)
        : null);
  }).catch(() => {});
}

function renderModelInfo(host, basePath) {
  api(`${basePath}/model-info`).then((p) => {
    if (!p.available) {
      host.replaceChildren(el("h3", {}, "Model"), banner(p.reason, "warn"));
      return;
    }
    const d = p.data;
    if (!d.found) {
      host.replaceChildren(el("h3", {}, "Model"), banner(d.note || "No model logged."));
      return;
    }

    const sigTable = (title, specs) => {
      if (!Array.isArray(specs) || !specs.length) return null;
      return el("div", {},
        el("div", { class: "chart-title", style: "margin:12px 0 6px" }, title),
        el("div", { class: "table-wrap" },
          el("table", {},
            el("thead", {}, el("tr", {},
              el("th", {}, "Name"), el("th", {}, "Type"), el("th", {}, "Shape"))),
            el("tbody", {}, ...specs.map((s) => {
              const spec = s["tensor-spec"] || {};
              return el("tr", {},
                el("td", { class: "mono" }, s.name != null ? String(s.name) : "—"),
                el("td", { class: "mono" }, spec.dtype || s.type || "—"),
                el("td", { class: "mono" },
                  spec.shape ? `[${spec.shape.join(", ")}]` : "—"));
            })))));
    };

    const sig = d.signature || {};
    const env = [];
    for (const [flavor, detail] of Object.entries(d.flavor_detail || {})) {
      for (const [k, v] of Object.entries(detail || {})) {
        env.push(el("dt", {}, `${flavor}.${k}`));
        env.push(el("dd", { class: "mono" }, String(v)));
      }
    }

    host.replaceChildren(
      el("h3", {}, "Model"),
      el("div", { class: "grid-2" },
        el("div", { class: "card" },
          el("div", { class: "chart-title" }, "Flavors and environment"),
          el("dl", { class: "kv" },
            el("dt", {}, "Flavors"),
            el("dd", {}, (d.flavors || []).join(", ") || "—"),
            el("dt", {}, "MLflow version"),
            el("dd", {}, d.mlflow_version || "—"),
            el("dt", {}, "Logged at"),
            el("dd", {}, d.utc_time_created || "—"),
            el("dt", {}, "Layout"),
            el("dd", { class: "faint" },
              d.layout === "logged-model"
                ? "MLflow 3 logged model"
                : "run artifact (MLflow 2)"),
            ...env)),
        el("div", { class: "card" },
          el("div", { class: "chart-title" }, "Signature"),
          sigTable("Inputs", sig.inputs) || el("div", { class: "muted" }, "No input schema."),
          sigTable("Outputs", sig.outputs))));
  }).catch(() => {});
}

async function initRunsCompare() {
  const body = document.getElementById("compare-body");
  const ids = (new URLSearchParams(location.search).get("ids") || "")
    .split(",").map((s) => s.trim()).filter(Boolean);

  if (ids.length < 2) {
    body.replaceChildren(banner("Pick at least two runs on the Runs page to compare them."));
    return;
  }

  let runs;
  try {
    runs = await Promise.all(ids.map((i) => api(`/training-runs/${i}`)));
  } catch (e) {
    setError(body, e);
    return;
  }

  const paramKeys = [...new Set(runs.flatMap((r) => Object.keys(r.parameters || {})))].sort();
  const metricKeys = [...new Set(runs.flatMap((r) => Object.keys(r.metrics || {})))].sort();

  // Differing rows first — that is the whole reason to open this page.
  function differs(get) {
    const vals = runs.map((r) => JSON.stringify(get(r) ?? null));
    return new Set(vals).size > 1;
  }

  function compareTable(title, keys, get) {
    const changed = keys.filter((k) => differs((r) => get(r)[k]));
    const same = keys.filter((k) => !changed.includes(k));
    const rows = [...changed, ...same].map((k) => {
      const values = runs.map((r) => get(r)[k]);
      const numeric = values.filter((v) => typeof v === "number");
      const max = numeric.length ? Math.max(...numeric) : null;
      return el("tr", {},
        el("td", { class: "mono" }, k, changed.includes(k) ? el("span", { class: "faint" }, " ●") : null),
        ...values.map((v) =>
          el("td", { class: "num" },
            typeof v === "number" && v === max && numeric.length > 1
              ? el("strong", { style: "color:var(--ok)" }, fmt.metric(v))
              : fmt.metric(v))));
    });
    return el("div", {},
      el("h3", {}, title),
      el("div", { class: "table-wrap" },
        el("table", {},
          el("thead", {}, el("tr", {},
            el("th", {}, "Key"),
            ...runs.map((r) => el("th", { class: "num" }, el("a", { href: `/runs/${r.id}` }, `#${r.id}`))))),
          el("tbody", {}, ...(rows.length ? rows : [emptyRow(runs.length + 1, "Nothing recorded.")])))));
  }

  const overview = el("div", { class: "table-wrap" },
    el("table", {},
      el("thead", {}, el("tr", {},
        el("th", {}, "Run"), el("th", {}, "Status"), el("th", {}, "Pipeline"),
        el("th", {}, "Orchestrator"), el("th", {}, "Duration"), el("th", {}, "Started"))),
      el("tbody", {}, ...runs.map((r) =>
        el("tr", {},
          el("td", {}, el("a", { href: `/runs/${r.id}` }, `#${r.id}`)),
          el("td", {}, statusBadge(r.status)),
          el("td", { class: "mono" }, r.pipeline_id || "—"),
          el("td", { class: "muted" }, r.orchestrator || "—"),
          // "mono", not "num" — "Duration" is left-aligned like the rest.
          el("td", { class: "mono" }, fmt.dur(r.duration_seconds)),
          el("td", { class: "muted nowrap" }, fmt.time(r.started_at)))))));

  const headline = METRIC_PRIORITY.find((m) => runs.some((r) => typeof (r.metrics || {})[m] === "number"));
  const chart = headline
    ? barChart(headline, runs.map((r) => ({
        label: `#${r.id}`,
        value: (r.metrics || {})[headline] ?? 0,
        kind: statusKind(r.status),
      })))
    : null;

  // Training curves — best-effort on top of everything above, which
  // already renders from the framework's own final-value metrics alone.
  // MLflow's per-step history is what makes a *curve* possible (see
  // runs.py::_run_mlflow_panel's docstring); a run with no MLflow panel
  // (unconfigured, unreachable, or this run never logged to it) just
  // contributes nothing to these charts rather than blocking the page —
  // same degrade-a-card contract every other MLflow-backed view follows.
  const curvesHost = el("div", {});
  Promise.all(runs.map((r) => api(`/training-runs/${r.id}/mlflow`).catch(() => ({ available: false }))))
    .then((panels) => {
      const historyByRun = panels.map((p) => (p.available ? p.data.history || {} : {}));
      const curveMetrics = [...new Set(historyByRun.flatMap((h) => Object.keys(h)))]
        .filter((m) => historyByRun.some((h) => (h[m] || []).length > 1))
        .sort();
      if (!curveMetrics.length) return;
      const charts = curveMetrics.map((m) =>
        multiLineChart(m, runs.map((r, i) => ({ label: `#${r.id}`, points: historyByRun[i][m] || [] }))));
      mount(curvesHost,
        el("h3", {}, "Training curves"),
        el("div", { class: "grid-3" }, ...charts));
    })
    .catch(() => {});

  mount(body,
    overview,
    chart ? el("div", { style: "margin-top:16px;max-width:520px" }, chart) : null,
    curvesHost,
    compareTable("Metrics — ● marks a value that differs", metricKeys, (r) => r.metrics || {}),
    compareTable("Parameters — ● marks a value that differs", paramKeys, (r) => r.parameters || {}));
}

/* ------------------------------------------------------------------ */
/* Models — registry view                                             */
/* ------------------------------------------------------------------ */

async function initModels() {
  const table = document.querySelector("table");
  try {
    const rows = await api("/models");
    makeSortable(table, rows,
      [
        { label: "Model", sort: (m) => m.name },
        { label: "Task" },
        { label: "Versions", sort: (m) => m.version_count },
        { label: "Production" },
        { label: "Key metric", sort: (m) => bestMetric({ metrics: m.production_version?.metrics })?.value },
      ],
      (m) => {
        const prod = m.production_version;
        const best = prod ? bestMetric({ metrics: prod.metrics }) : null;
        return el("tr", {},
          el("td", {}, el("a", { href: `/models/${m.id}` }, m.name)),
          el("td", { class: "muted" }, m.task || "—"),
          // "num" (right-aligned) reads fine in a column of nothing but
          // numbers stacked under a right-aligned header — this table's
          // headers are all left-aligned, so a right-aligned cell here
          // just drifts away from its own header. "mono" keeps the
          // tabular-nums/monospace look without the alignment mismatch.
          el("td", { class: "mono" }, String(m.version_count)),
          // Just the badge — the "Versions" column already carries the
          // number that matters here (how many), and appending " vN"
          // put a second, differently-scoped version number right next
          // to it (which one is in production) that read as the same
          // fact repeated. The specific number is one click away on the
          // model's own page (production card + Versions table).
          el("td", {}, prod ? statusBadge("PRODUCTION") : el("span", { class: "faint" }, "none")),
          el("td", { class: "mono" }, best ? `${best.name} ${fmt.metric(best.value)}` : "—"));
      });
  } catch (e) {
    setError(table.parentElement, e);
  }
}


// "Roll back to this version" — only offered for a version that could
// actually take over: ARCHIVED (retired, the normal case) or APPROVED
// (promoted-adjacent but never served). PRODUCTION is already live and
// CANDIDATE/REJECTED have never been a known-good production version,
// so those rows get nothing rather than a button that 409s.
//
// Confirmation is deliberate and names both sides: this swaps what is
// being served, and the endpoint applies no metric policy to second-
// guess the operator (see ModelManager.rollback_to).
function rollbackButton(v, reload) {
  if (v.state !== "ARCHIVED" && v.state !== "APPROVED") return null;
  const btn = el("button", { class: "btn" }, "Roll back");
  btn.addEventListener("click", async () => {
    if (!confirm(
      `Roll production back to v${v.version_number}?\n\n` +
      "The current production version will be archived and the serving " +
      "bridge asked to reload. This is recorded in the audit trail."
    )) return;
    btn.disabled = true;
    btn.textContent = "Rolling back…";
    try {
      const r = await apiWrite(`/model-versions/${v.id}/rollback`, { method: "POST" });
      flash(
        `${r.model_name}: production is now v${r.restored_version}` +
        (r.previous_production_version ? ` (v${r.previous_production_version} archived)` : "") +
        (r.serving_reloaded ? "." : " — the serving bridge did not confirm the reload."),
        r.serving_reloaded ? "ok" : "warn",
      );
      reload();
    } catch (e) {
      flash(e.message, "err");
      btn.disabled = false;
      btn.textContent = "Roll back";
    }
  });
  return btn;
}

async function initModelDetail(id) {
  const head = document.getElementById("model-head");
  const body = document.getElementById("model-body");
  try {
    const [model, versions] = await Promise.all([
      api(`/models/${id}`),
      api(`/models/${id}/versions`),
    ]);

    mount(head,
      el("div", { class: "breadcrumb" }, el("a", { href: "/models" }, "Models"), " / ", model.name),
      el("h2", {}, model.name),
      el("p", { class: "subtitle" }, model.description || "No description",
        model.task ? el("span", { class: "faint" }, `  ·  ${model.task}`) : null));

    // Best-effort: registry-reconciliation is an ExternalPanel (MLflow may
    // be down/unconfigured), so its columns are added to the table once it
    // resolves rather than blocking the framework's own data from showing.
    // This *is* the "does Gateflow's registry match MLflow's" answer — see
    // renderRegistrySummary below for the same call's top-of-page banner.
    const reconcilePanel = el("div", {});
    const ordered = versions.slice().reverse();
    const metricKeys = [...new Set(versions.flatMap((v) => Object.keys(v.metrics || {})))]
      .filter((k) => METRIC_PRIORITY.includes(k))
      .sort((a, b) => METRIC_PRIORITY.indexOf(a) - METRIC_PRIORITY.indexOf(b));

    const tableHost = el("div", { class: "table-wrap" },
      el("table", {}, el("thead", {}, el("tr", {})), el("tbody", {}, emptyRow(1, "Loading…"))));

    // No "MLflow" column here on purpose — the framework's own State is
    // the one badge a reader needs per row; MLflow's registry view of
    // the same version (its own numbering, aliases, stage) lives in the
    // collapsed "MLflow registry" panel above instead of repeating a
    // second status vocabulary on every row. A real disagreement still
    // surfaces right here, inline, via the "MLflow disagrees" badge —
    // that is the one case worth interrupting this table for.
    function renderVersionsTable(byVersionId) {
      const table = el("table", {},
        el("thead", {}, el("tr", {},
          el("th", {}, "Version"), el("th", {}, "State"),
          ...metricKeys.map((k) => el("th", {}, k)),
          el("th", {}, "Run"), el("th", {}, "Dataset"), el("th", {}, "Created"),
          el("th", {}, "Report"), el("th", {}, ""))),
        el("tbody", {}, ...(ordered.length ? ordered.map((v) => {
          const best = {};
          for (const k of metricKeys) {
            best[k] = Math.max(...versions.map((x) => (x.metrics || {})[k] ?? -Infinity));
          }
          const reg = byVersionId.get(v.id);
          return el("tr", {},
            el("td", {}, el("strong", {}, `v${v.version_number}`)),
            el("td", {}, statusBadge(v.state),
              reg?.drift ? el("span", {
                class: "badge failed", style: "margin-left:6px",
                title: reg.drift_reason || "",
              }, "MLflow disagrees") : null),
            ...metricKeys.map((k) => {
              const val = (v.metrics || {})[k];
              const isBest = typeof val === "number" && val === best[k] && versions.length > 1;
              // "mono", not "num" — this table's headers are all left-aligned.
              return el("td", { class: "mono" },
                isBest ? el("strong", { style: "color:var(--ok)" }, fmt.metric(val)) : fmt.metric(val));
            }),
            el("td", {}, v.training_run_id ? el("a", { href: `/runs/${v.training_run_id}` }, `#${v.training_run_id}`) : "—"),
            el("td", { class: "muted" }, v.dataset_version_id ? `#${v.dataset_version_id}` : "—"),
            el("td", { class: "muted nowrap" }, fmt.ago(v.created_at)),
            // Content-Disposition: attachment on the endpoint itself
            // triggers a real browser download — no download="" attribute
            // needed here (and Artifact-style sandboxes don't apply to
            // this deployed app, only to claude.ai's own preview).
            el("td", {}, el("a", { href: `${API}/model-versions/${v.id}/report` }, "report")),
            el("td", { class: "row-actions" },
              // initModelDetail re-mounts both regions, so re-running
              // it is the refresh — this page has no load() of its own.
              rollbackButton(v, () => initModelDetail(id))));
        }) : [emptyRow(metricKeys.length + 7, "No versions registered yet.")])));
      tableHost.replaceChildren(table);
    }

    renderVersionsTable(new Map());

    const prod = versions.find((v) => v.state === "PRODUCTION");
    const chart = metricKeys.length && versions.length > 1
      ? barChart(`${metricKeys[0]} by version`, ordered.map((v) => ({
          label: `v${v.version_number}`,
          value: (v.metrics || {})[metricKeys[0]] ?? 0,
          kind: v.state === "PRODUCTION" ? "success" : "",
        })))
      : null;

    mount(body,
      prod ? el("div", { class: "card", style: "margin-bottom:16px" },
        el("div", { class: "chart-title" }, "Current production version"),
        el("dl", { class: "kv" },
          el("dt", {}, "Version"), el("dd", {}, `v${prod.version_number}`),
          el("dt", {}, "Artifact"), el("dd", {}, prod.artifact_uri || "—"),
          el("dt", {}, "Training run"), el("dd", {}, prod.training_run_id ? `#${prod.training_run_id}` : "—"),
          el("dt", {}, "Promoted"), el("dd", {}, fmt.time(prod.created_at)))) : null,
      reconcilePanel,
      el("h3", {}, "Versions"),
      tableHost,
      chart ? el("div", { style: "margin-top:16px;max-width:520px" }, chart) : null,
      prod ? el("p", { style: "margin-top:16px" },
        el("a", { class: "btn", href: `/lineage?kind=model-version&id=${prod.id}` }, "View lineage")) : null);

    renderRegistrySummary(reconcilePanel, id, renderVersionsTable);
  } catch (e) {
    setError(body, e);
  }
}

/* ------------------------------------------------------------------ */
/* Scheduling                                                           */
/* ------------------------------------------------------------------ */

function humanizeCron(expr) {
  // A handful of common shapes rendered in words; anything else just
  // shows the raw expression — not a full cron humanizer, which is
  // more machinery than a tooltip needs.
  const m = (expr || "").trim().split(/\s+/);
  if (m.length !== 5) return expr;
  const [min, hour, dom, mon, dow] = m;
  const at = (h, mi) => `${String(h).padStart(2, "0")}:${String(mi).padStart(2, "0")}`;
  if (dom === "*" && mon === "*" && dow === "*") {
    if (/^\d+$/.test(min) && /^\d+$/.test(hour)) return `daily at ${at(hour, min)}`;
    if (min === "*" && hour === "*") return "every minute";
    if (min.startsWith("*/") && hour === "*") return `every ${min.slice(2)} minutes`;
  }
  if (dom === "*" && mon === "*" && /^\d+$/.test(dow) && /^\d+$/.test(min) && /^\d+$/.test(hour)) {
    const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    return `weekly on ${days[Number(dow)] ?? dow} at ${at(hour, min)}`;
  }
  if (/^\d+$/.test(dom) && mon === "*" && dow === "*" && /^\d+$/.test(min) && /^\d+$/.test(hour)) {
    return `monthly on day ${dom} at ${at(hour, min)}`;
  }
  return expr;
}

async function initSchedules() {
  const formHost = document.getElementById("schedule-form-host");
  const statusHost = document.getElementById("schedule-status");
  const out = document.getElementById("schedules-out");
  const newBtn = document.getElementById("new-schedule-btn");

  let models = [], datasets = [];
  try {
    [models, datasets] = await Promise.all([api("/models"), api("/datasets")]);
  } catch (e) {
    setError(out, e);
    return;
  }

  function flash(msg, kind = "") {
    statusHost.replaceChildren(banner(msg, kind));
    setTimeout(() => { if (statusHost.firstChild?.textContent === msg) statusHost.replaceChildren(); }, 6000);
  }

  async function load() {
    let schedules;
    try {
      schedules = await api("/schedules");
    } catch (e) {
      setError(out, e);
      return;
    }

    const table = el("table", {},
      el("thead", {}, el("tr", {},
        el("th", {}, "Model"), el("th", {}, "Dataset"), el("th", {}, "Cron"),
        el("th", {}, "Status"), el("th", {}, "Next fire"), el("th", {}, "Last run"),
        el("th", {}, ""))),
      el("tbody", {}, ...(schedules.length ? schedules.map((s) => {
        const toggleBtn = el("button", { class: "btn" }, s.enabled ? "Disable" : "Enable");
        toggleBtn.addEventListener("click", async () => {
          try {
            await apiWrite(`/schedules/${s.id}`, {
              method: "PATCH", body: JSON.stringify({ enabled: !s.enabled }),
            });
            flash(`Schedule #${s.id} ${s.enabled ? "disabled" : "enabled"}.`, "ok");
            load();
          } catch (e) { flash(e.message, "err"); }
        });

        const runNowBtn = el("button", { class: "btn" }, "Run now");
        runNowBtn.addEventListener("click", async () => {
          runNowBtn.disabled = true;
          runNowBtn.textContent = "Running…";
          try {
            const result = await apiWrite(`/schedules/${s.id}/run-now`, { method: "POST" });
            flash(
              result.fired
                ? `Schedule #${s.id} ran — promoted=${result.promoted} (training run #${result.training_run_id}).`
                : `Schedule #${s.id} did not fire: ${result.skipped_reason}.`,
              result.fired && result.promoted ? "ok" : "warn",
            );
            load();
          } catch (e) {
            flash(e.message, "err");
            runNowBtn.disabled = false;
            runNowBtn.textContent = "Run now";
          }
        });

        const deleteBtn = el("button", { class: "btn" }, "Delete");
        deleteBtn.addEventListener("click", async () => {
          if (!confirm(`Delete schedule #${s.id}? This cannot be undone.`)) return;
          try {
            await apiWrite(`/schedules/${s.id}`, { method: "DELETE" });
            flash(`Schedule #${s.id} deleted.`, "ok");
            load();
          } catch (e) { flash(e.message, "err"); }
        });

        return el("tr", {},
          el("td", {}, s.model_name
            ? el("a", { href: `/models/${s.model_id}` }, s.model_name)
            : `#${s.model_id}`),
          el("td", { class: "muted" }, s.dataset_name || `#${s.dataset_id}`),
          el("td", {},
            el("span", { title: s.cron_expression }, humanizeCron(s.cron_expression)),
            el("div", { class: "faint mono", style: "font-size:11px" }, s.cron_expression)),
          el("td", {}, s.enabled
            ? el("span", { class: "badge success" }, "enabled")
            : el("span", { class: "faint" }, "disabled")),
          el("td", { class: "muted nowrap" }, s.enabled ? fmt.time(s.next_fire_at) : "—"),
          el("td", {}, s.last_training_run_id
            ? el("a", { href: `/runs/${s.last_training_run_id}` }, `#${s.last_training_run_id}`)
            : el("span", { class: "faint" }, "never")),
          el("td", { class: "row-actions" }, toggleBtn, runNowBtn, deleteBtn));
      }) : [emptyRow(7, "No schedules yet — click “New schedule” to add one.")])));

    out.replaceChildren(el("div", { class: "table-wrap" }, table));
  }

  function showForm() {
    if (!models.length || !datasets.length) {
      formHost.replaceChildren(banner(
        "Create a model and a dataset (with at least one version) first — "
        + "a schedule needs both to know what to train.", "warn"));
      return;
    }
    const modelSel = el("select", {}, ...models.map((m) => el("option", { value: m.id }, m.name)));
    const datasetSel = el("select", {}, ...datasets.map((d) => el("option", { value: d.id }, d.name)));
    const pipelineInput = el("input", { type: "text", placeholder: "package.module:callable", size: "34" });
    const cronInput = el("input", { type: "text", placeholder: "0 2 * * *", size: "16" });
    const minF1Input = el("input", { type: "number", step: "0.01", min: "0", max: "1", value: "0.0", size: "6" });
    const submitBtn = el("button", { class: "btn primary" }, "Create");
    const cancelBtn = el("button", { class: "btn" }, "Cancel");

    submitBtn.addEventListener("click", async () => {
      if (!pipelineInput.value.trim() || !cronInput.value.trim()) {
        flash("Pipeline id and cron expression are both required.", "err");
        return;
      }
      try {
        await apiWrite("/schedules", {
          method: "POST",
          body: JSON.stringify({
            model_id: Number(modelSel.value),
            dataset_id: Number(datasetSel.value),
            pipeline_id: pipelineInput.value.trim(),
            cron_expression: cronInput.value.trim(),
            min_f1: Number(minF1Input.value) || 0,
          }),
        });
        formHost.replaceChildren();
        flash("Schedule created.", "ok");
        load();
      } catch (e) {
        flash(e.message, "err");
      }
    });
    cancelBtn.addEventListener("click", () => formHost.replaceChildren());

    formHost.replaceChildren(
      el("div", { class: "card", style: "margin-bottom:16px" },
        el("div", { class: "chart-title" }, "New schedule"),
        el("div", { class: "form-grid" },
          el("label", {}, "Model", modelSel),
          el("label", {}, "Dataset", datasetSel),
          el("label", {}, "Pipeline entrypoint", pipelineInput),
          el("label", {}, "Cron expression", cronInput),
          el("label", {}, "Min F1 to promote", minF1Input)),
        el("div", { style: "margin-top:12px;display:flex;gap:8px" }, submitBtn, cancelBtn)));
  }

  newBtn.addEventListener("click", showForm);
  document.getElementById("refresh").addEventListener("click", load);
  await load();
}

/* ------------------------------------------------------------------ */
/* Settings                                                            */
/* ------------------------------------------------------------------ */

// A reachability badge distinct from statusBadge()'s run/model vocabulary:
// "not configured" (no URL set) reads differently from "configured but
// unreachable" (URL set, ping failed) — collapsing them into one grey
// badge would hide the more actionable of the two states.
function reachabilityBadge(system) {
  if (!system.configured) return el("span", { class: "badge plain" }, "not configured");
  return system.reachable
    ? el("span", { class: "badge success" }, "reachable")
    : el("span", { class: "badge failed" }, "unreachable");
}

function settingsCard(title, system) {
  const rows = Object.entries(system.fields).flatMap(([k, v]) => [
    el("dt", {}, k.replace(/_/g, " ")),
    el("dd", {}, v == null || v === "" ? "—" : String(v)),
  ]);
  return el("div", { class: "card" },
    el("div", { class: "chart-title", style: "display:flex;align-items:center;justify-content:space-between" },
      title, reachabilityBadge(system)),
    system.reason ? el("p", { class: "muted", style: "margin:0 0 10px" }, system.reason) : null,
    el("dl", { class: "kv" }, ...rows));
}

// The four governance policies FrameworkSettingsManager persists (see
// framework_settings/manager.py) — key order here is display order, and
// the title is the one place their internal key names get a human label.
const POLICY_TITLES = {
  promotion: "Promotion",
  eligibility: "Training eligibility",
  training_policy: "Dataset readiness",
  drift: "Drift detection",
};
const POLICY_ORDER = ["promotion", "eligibility", "training_policy", "drift"];

// Same badge either place a policy's default/customized state shows —
// the Settings list row and the policy's own detail card (paint()
// below) — so the two never drift into describing it differently.
function policyBadge(entry) {
  return entry.is_default
    ? el("span", { class: "badge plain" }, "default")
    : el("span", { class: "badge success" }, "customized");
}

// api()'s errors carry the raw response body after the status line
// (see api() above) — for a 4xx from policy_settings.py that's a JSON
// {"detail": "..."} blob. Pull the message out when it parses as that;
// fall back to the raw text for anything else (a network failure, a
// non-JSON 5xx) rather than showing "[object Object]" or nothing.
function apiErrorDetail(e) {
  const brace = e.message.indexOf("{");
  if (brace === -1) return e.message;
  try {
    return JSON.parse(e.message.slice(brace)).detail || e.message;
  } catch {
    return e.message;
  }
}

// One policy's card: a raw-JSON textarea (pre-filled with its effective
// value, pretty-printed) is the whole edit surface rather than ~7-9
// bespoke fields per policy (26+ fields total across all four) — same
// idiom this app already uses for structured data it shows but doesn't
// build a form for (detailCell()'s <pre> on the Activity page). Save
// round-trips through the same from_dict()/to_dict() validation
// constructing the dataclass directly would apply (see
// FrameworkSettingsManager.set_raw), so a malformed edit is rejected
// with the real reason, not silently coerced.
// `onUpdate`, when given, is told about every successful save/reset —
// the Settings page's list view uses it to keep the badge it shows for
// this policy current after a visit to the detail view, without a
// second fetch.
function policyCard(key, entry, onUpdate) {
  const textarea = el("textarea", {
    class: "policy-json", spellcheck: "false", rows: "9",
    "aria-label": `${POLICY_TITLES[key] || key} policy JSON`,
  });
  const badgeSlot = el("span", {});
  const errorBox = el("div", { class: "policy-error" });
  const saveBtn = el("button", { class: "btn primary", type: "button" }, "Save");
  const resetBtn = el("button", { class: "btn", type: "button" }, "Reset to default");

  function paint(e) {
    textarea.value = JSON.stringify(e.value, null, 2);
    mount(badgeSlot, policyBadge(e));
    resetBtn.disabled = e.is_default;
    mount(errorBox);
    onUpdate?.(e);
  }
  paint(entry);

  saveBtn.addEventListener("click", async () => {
    let parsed;
    try {
      parsed = JSON.parse(textarea.value);
    } catch (parseErr) {
      mount(errorBox, banner(`Invalid JSON: ${parseErr.message}`, "err"));
      return;
    }
    saveBtn.disabled = true;
    try {
      paint(await api(`/settings/policies/${key}`, {
        method: "PUT",
        body: JSON.stringify({ value: parsed }),
      }));
    } catch (apiErr) {
      mount(errorBox, banner(apiErrorDetail(apiErr), "err"));
    } finally {
      saveBtn.disabled = false;
    }
  });

  resetBtn.addEventListener("click", async () => {
    resetBtn.disabled = true;
    try {
      paint(await api(`/settings/policies/${key}/reset`, { method: "POST" }));
    } catch (apiErr) {
      mount(errorBox, banner(apiErrorDetail(apiErr), "err"));
      resetBtn.disabled = false;
    }
  });

  return el("div", { class: "card policy-card" },
    el("div", { class: "chart-title", style: "display:flex;align-items:center;justify-content:space-between" },
      POLICY_TITLES[key] || key, badgeSlot),
    textarea,
    errorBox,
    el("div", { class: "policy-actions" }, saveBtn, resetBtn));
}

// The Settings page's 5 sections — the read-only connectivity panel
// plus the 4 editable policies — each get one row on the landing list;
// clicking a row swaps the page into that section's own detail view
// (a plain in-memory state switch, same idea as initActivity()'s two
// tabs, not a real navigation) with a breadcrumb back to the list.
// One line of context per section, shown both on its list row and atop
// its own detail view, so the two never describe it differently.
const SETTINGS_SECTION_DESCRIPTIONS = {
  connectivity: "What Gateflow is pointed at right now — database, MLflow, Airflow — and whether each one actually answered.",
  promotion: "Thresholds a candidate model version must clear before it can be promoted to production.",
  eligibility: "Whether a retrain should actually run right now — new data, drift, cooldown, and production-quality gates.",
  training_policy: "Structural checks a dataset version must pass before training can start.",
  drift: "Sensitivity of the statistical tests that flag a dataset version as drifted.",
};
const SETTINGS_SECTIONS = ["connectivity", ...POLICY_ORDER];

function settingsSectionTitle(key) {
  return key === "connectivity" ? "Connectivity" : POLICY_TITLES[key];
}

// "connected" only once every *configured* system actually answered —
// MLflow/Airflow being unconfigured is the app's normal unconfigured
// state (reachabilityBadge's own "not configured", neutral), not a
// reason to warn on the list row; Database is always configured.
function connectivitySummaryBadge(s) {
  const attention = [s.database, s.mlflow, s.airflow].some(
    (sys) => sys.configured && !sys.reachable);
  return attention
    ? el("span", { class: "badge failed" }, "needs attention")
    : el("span", { class: "badge success" }, "connected");
}

// One clickable row: title + one-line description on the left, a
// status badge and a chevron (this row leads somewhere, unlike every
// other badge-bearing row in the app) on the right.
function settingsMenuItem(title, desc, badge, onClick) {
  const btn = el("button", { class: "settings-menu-item", type: "button" },
    el("span", { class: "text" },
      el("span", { class: "title" }, title),
      el("span", { class: "desc" }, desc)),
    el("span", { class: "right" }, badge, el("span", { class: "chevron", "aria-hidden": "true" }, "›")));
  btn.addEventListener("click", onClick);
  return btn;
}

async function initSettings() {
  const out = document.getElementById("settings-out");
  let s, policies;

  function backCrumb(title) {
    const back = el("a", { href: "#" }, "Settings");
    back.addEventListener("click", (e) => { e.preventDefault(); goTo(null); });
    return el("div", { class: "breadcrumb" }, back, " / ", title);
  }

  function renderList() {
    out.replaceChildren(
      el("div", { class: "card" },
        el("div", { class: "settings-menu" },
          settingsMenuItem(
            settingsSectionTitle("connectivity"), SETTINGS_SECTION_DESCRIPTIONS.connectivity,
            connectivitySummaryBadge(s), () => goTo("connectivity", { push: true })),
          ...POLICY_ORDER.map((key) =>
            settingsMenuItem(
              settingsSectionTitle(key), SETTINGS_SECTION_DESCRIPTIONS[key],
              policyBadge(policies[key]), () => goTo(key, { push: true }))))));
  }

  function renderConnectivity() {
    out.replaceChildren(
      backCrumb("Connectivity"),
      el("div", { class: "grid-2" },
        settingsCard("Database", s.database),
        settingsCard("MLflow", s.mlflow),
        settingsCard("Airflow", s.airflow),
        el("div", { class: "card" },
          el("div", { class: "chart-title" }, "Application"),
          el("dl", { class: "kv" },
            el("dt", {}, "name"), el("dd", {}, s.app_name),
            el("dt", {}, "version"), el("dd", {}, s.app_version),
            el("dt", {}, "scheduler enabled"), el("dd", {}, String(s.scheduler.enabled)),
            el("dt", {}, "scheduler poll seconds"), el("dd", {}, String(s.scheduler.poll_seconds))))));
  }

  function renderPolicy(key) {
    out.replaceChildren(
      backCrumb(POLICY_TITLES[key]),
      el("p", { class: "muted", style: "margin:0 0 14px" },
        SETTINGS_SECTION_DESCRIPTIONS[key] + " A more specific value — a schedule's own min F1, " +
        "a manual promote/readiness request's own values — still takes precedence over this. ",
        el("a", { href: "/activity" }, "Every change is on the Activity page"), "."),
      policyCard(key, policies[key], (updated) => { policies[key] = updated; }));
  }

  function render(section) {
    if (section === "connectivity") renderConnectivity();
    else if (POLICY_ORDER.includes(section)) renderPolicy(section);
    else renderList();
  }

  // ?section= makes a detail view directly linkable/refreshable, same
  // convention as initRuns()'s ?experiment= (see expFilter's change
  // handler above). List -> detail (a row click) *pushes* a real
  // history entry — without one, the browser's own back button had
  // nothing to land on between "viewing a policy" and wherever the
  // user was before they opened Settings at all, and skipped straight
  // to that (reported as "back goes to the dashboard, not the list").
  // Detail -> list (the breadcrumb link) still replaces: it's just
  // converging back onto the list state that a preceding push already
  // left one step behind, not a new place to arrive at from two
  // different directions.
  function goTo(section, { push = false } = {}) {
    const url = new URL(location);
    if (section) url.searchParams.set("section", section);
    else url.searchParams.delete("section");
    if (push) history.pushState(null, "", url);
    else history.replaceState(null, "", url);
    render(section);
  }

  // pushState/replaceState never themselves trigger navigation or a
  // page load — only the browser's actual back/forward does, as a
  // "popstate" event, which is otherwise invisible to a page that
  // never listens for it. Without this, following the button back to
  // a bare /settings after the fix above would leave the *rendered*
  // content stuck on whatever detail view was last drawn, our own URL
  // bar the only thing that changed.
  window.addEventListener("popstate", () => {
    render(new URLSearchParams(location.search).get("section"));
  });

  async function load() {
    try {
      [s, policies] = await Promise.all([api("/settings"), api("/settings/policies")]);
    } catch (e) {
      setError(out, e);
      return;
    }
    const requested = new URLSearchParams(location.search).get("section");
    render(SETTINGS_SECTIONS.includes(requested) ? requested : null);
  }

  document.getElementById("refresh").addEventListener("click", load);
  await load();
}

/* ------------------------------------------------------------------ */
/* Activity — audit trail (who did what) + alerts (what the framework   */
/* itself detected) on two tabs of the same page.                       */
/* ------------------------------------------------------------------ */

// PROMOTED/CREATED read as an outcome worth calling out; REJECTED/DELETED
// as the opposite; everything else (UPDATED, RUN_NOW) is just "something
// happened" — reuses the same three-way badge vocabulary as statusKind()
// without forcing audit actions through its run/model-specific words.
function actionBadge(action) {
  const a = String(action || "");
  if (/PROMOTED|CREATED/.test(a)) return el("span", { class: "badge success" }, a);
  if (/REJECTED|DELETED/.test(a)) return el("span", { class: "badge failed" }, a);
  return el("span", { class: "badge plain" }, a);
}

// GovernanceEvent's own severity, not a run/model status — CRITICAL reads
// as failed (red), WARNING as pending (amber), INFO as plain.
function severityBadge(severity) {
  const s = String(severity || "");
  if (s === "CRITICAL") return el("span", { class: "badge failed" }, s);
  if (s === "WARNING") return el("span", { class: "badge pending" }, s);
  return el("span", { class: "badge plain" }, s);
}

function detailCell(obj) {
  return obj && Object.keys(obj).length
    ? el("details", {},
        el("summary", { class: "faint" }, "detail"),
        el("pre", { class: "log", style: "margin-top:6px" }, JSON.stringify(obj, null, 2)))
    : el("span", { class: "faint" }, "—");
}

async function initActivity() {
  const out = document.getElementById("activity-out");
  const title = document.getElementById("activity-title");
  const desc = document.getElementById("activity-desc");
  const auditBtn = document.getElementById("tab-audit");
  const alertsBtn = document.getElementById("tab-alerts");
  const alertsCount = document.getElementById("alerts-count");

  // Fetched once and cached — refreshed only by refetch() below, called
  // on page load regardless of which tab is open (its count/severity is
  // what makes the Alerts *tab* worth noticing before anyone clicks it;
  // see the badge update at the bottom) and again whenever the Alerts
  // tab is actually opened or "Refresh" is pressed while on it, same as
  // the audit tab has always done. One promise shared by both callers
  // (the eager badge fetch and tab.fetch()) rather than two independent
  // requests racing to fill the same cache.
  let alertsCache = null;
  function refetchAlerts() {
    alertsCache = api("/alerts?limit=200").then((rows) => {
      if (!rows.length) { alertsCount.hidden = true; return rows; }
      const worst = rows.some((r) => r.severity === "CRITICAL") ? "critical"
        : rows.some((r) => r.severity === "WARNING") ? "warning" : "";
      alertsCount.textContent = String(rows.length);
      alertsCount.className = `tab-count ${worst}`.trim();
      alertsCount.hidden = false;
      return rows;
    });
    return alertsCache;
  }

  const TABS = {
    audit: {
      title: "Audit trail",
      desc: 'Who — or what — triggered a schedule or model-promotion decision, and when. ' +
        'There is no login yet, so "who" is whatever the caller sent in an ' +
        '<code>X-Actor</code> header (<code>system</code> otherwise) — an honour-system ' +
        "record, not verified identity.",
      fetch: () => api("/audit?limit=200"),
      columns: ["When", "Actor", "Action", "Entity", "Detail"],
      row: (e) => el("tr", {},
        el("td", { class: "muted nowrap" }, fmt.ago(e.created_at)),
        el("td", { class: "mono" }, e.actor),
        el("td", {}, actionBadge(e.action)),
        el("td", { class: "muted" }, e.entity_type ? `${e.entity_type} #${e.entity_id}` : "—"),
        el("td", {}, detailCell(e.metadata))),
      empty: "No activity recorded yet.",
    },
    alerts: {
      title: "Alerts",
      desc: "Conditions the framework itself detected — a training run failing, a dataset " +
        "drifting, a retrain blocked before it started — not something anyone triggered.",
      fetch: () => alertsCache || refetchAlerts(),
      columns: ["When", "Severity", "Type", "Entity", "Message"],
      row: (e) => el("tr", {},
        el("td", { class: "muted nowrap" }, fmt.ago(e.created_at)),
        el("td", {}, severityBadge(e.severity)),
        el("td", { class: "mono" }, e.event_type),
        el("td", { class: "muted" }, e.entity_type ? `${e.entity_type} #${e.entity_id}` : "—"),
        el("td", {}, e.message, e.payload ? detailCell(e.payload) : null)),
      empty: "No alerts — nothing detected yet.",
    },
  };

  let active = "audit";

  async function load() {
    const tab = TABS[active];
    title.textContent = tab.title;
    desc.innerHTML = tab.desc;
    auditBtn.className = active === "audit" ? "tab active" : "tab";
    auditBtn.setAttribute("aria-selected", String(active === "audit"));
    alertsBtn.className = active === "alerts" ? "tab active" : "tab";
    alertsBtn.setAttribute("aria-selected", String(active === "alerts"));

    let entries;
    try {
      entries = await tab.fetch();
    } catch (e) {
      setError(out, e);
      return;
    }
    const table = el("table", {},
      el("thead", {}, el("tr", {}, ...tab.columns.map((c) => el("th", {}, c)))),
      el("tbody", {}, ...(entries.length ? entries.map(tab.row)
        : [emptyRow(tab.columns.length, tab.empty)])));
    out.replaceChildren(el("div", { class: "table-wrap" }, table));
  }

  auditBtn.addEventListener("click", () => { active = "audit"; load(); });
  alertsBtn.addEventListener("click", () => { active = "alerts"; load(); });
  document.getElementById("refresh").addEventListener("click", () => {
    // The badge has to stay live even while sitting on Audit trail, so
    // it's refetched here independently of whichever tab load() is
    // about to reload — on the Alerts tab that would otherwise be two
    // requests racing for the same cache; forcing the cache clear first
    // makes tab.fetch()'s alertsCache-miss path do that refetch instead.
    if (active === "alerts") alertsCache = null;
    else refetchAlerts();
    load();
  });

  // Regardless of which tab is open at load — the whole point of the
  // badge is earning the Alerts tab a second look from someone sitting
  // on Audit trail who would otherwise have no reason to click it.
  if (active !== "alerts") refetchAlerts();

  await load();
}

/* ------------------------------------------------------------------ */
/* Lineage                                                             */
/* ------------------------------------------------------------------ */

async function renderLineagePicker(out) {
  // The nav's "Lineage" link (and any deep-link with no ?kind=&id=) used
  // to land here with nothing but instructions — technically correct
  // (a graph needs a root to walk from) but useless the moment someone
  // actually wants to see a lineage, since every other page reaches this
  // one only via a link that already carries a node. Listing the current
  // production model versions and latest dataset versions turns this
  // into a real landing page instead of a dead end.
  let models, datasets;
  try {
    [models, datasets] = await Promise.all([api("/models"), api("/datasets")]);
  } catch (e) {
    setError(out, e);
    return;
  }

  const modelRows = models
    .filter((m) => m.production_version)
    .map((m) => el("tr", {},
      el("td", {},
        el("a", { href: `/lineage?kind=model-version&id=${m.production_version.id}` },
          `${m.name} — production v${m.production_version.version_number}`)),
      el("td", { class: "muted" }, "ModelVersion")));

  const datasetRows = datasets
    .filter((d) => d.latest_version)
    .map((d) => el("tr", {},
      el("td", {},
        el("a", { href: `/lineage?kind=dataset&id=${d.id}` },
          // Whole-family view — every version of this dataset side by
          // side, not just the latest. See LineageManager.graph_for_dataset.
          `${d.name} — v${d.latest_version.version_number} and history`)),
      el("td", { class: "muted" }, "Dataset")));

  const rows = [...modelRows, ...datasetRows];
  if (rows.length === 0) {
    out.replaceChildren(banner(
      "Nothing to trace yet — lineage starts once a dataset has a version "
      + "and a model has a production version. Register a dataset and run "
      + "a training pipeline, then come back here."));
    return;
  }

  out.replaceChildren(
    el("p", { class: "muted" }, "Pick a starting point to walk its lineage:"),
    el("div", { class: "table-wrap" },
      el("table", {},
        el("thead", {}, el("tr", {}, el("th", {}, "Start from"), el("th", {}, "Type"))),
        el("tbody", {}, ...rows))));
}

// Fixed column per node *type* — every DatasetVersion lines up in one
// column, every TrainingRun in the next, and so on — rather than a
// column per graph depth. An earlier version computed columns the way
// dagLevels() layers tasks (level(root) = 0, level(n) = 1 +
// max(level(upstream))), which is right for a DAG whose shape *is* the
// question, but wrong here: since the lineage graph now shows a
// dataset's whole version history in parallel (LineageManager.
// graph_for_dataset), a later version sits at a deeper topological
// level than an earlier one purely because it has one more hop of
// derived_from ancestry behind it — so V1's TrainingRun and V2's
// TrainingRun landed in different columns despite being the exact same
// *kind* of thing, which read as misaligned rather than as two
// branches of one story. Same type, same column, fixes that outright.
// RetrainingDecision used to be a node type here — its own column,
// then several rounds of separating its edges' attachment points from
// the dataset's own (dedicated hubs, obstruction detours, wire hops)
// once sharing a column made it a tangle. None of it made the decision
// look like part of the same lineage chain, because it never was one:
// a governance verdict about a node, not a fact that produced or
// consumed one. It's gone as a node entirely now — see
// lineage/manager.py's docstring on why — and shows up instead as a
// status chip in the meta line of whichever node it actually reached
// (lineageNodeMeta), so the graph only has the four kinds of node that
// were ever real lineage facts.
const LINEAGE_TYPE_COLUMN = {
  DatasetVersion: 0,
  TrainingRun: 1,
  ModelVersion: 2,
  ServingInstance: 3,
};

function lineageLevels(nodes, edges) {
  // Row order starts as node-array order — the backend's own discovery
  // order, dataset version 1's whole branch before version 2's — and is
  // then refined against the edges by orderByBarycentre below. An
  // unknown future type falls through to column 0 rather than being
  // dropped.
  const columns = [];
  for (const n of nodes) {
    const col = LINEAGE_TYPE_COLUMN[n.type] ?? 0;
    (columns[col] || (columns[col] = [])).push(n.id);
  }
  // Compact away unoccupied columns before returning. Assigning by type
  // index leaves *holes* whenever a type is absent — a graph with a
  // run that hasn't produced a ModelVersion yet leaves index 2 empty —
  // and a sparse array breaks the caller
  // two ways at once: it lays nodes out at `PAD + ci * COL_W` using the
  // raw index, so the gap either side of the missing column is drawn
  // twice as wide as a real one; and `Math.max(1, ...cols.map(c =>
  // c.length))` spreads the hole as `undefined`, making maxRows NaN and
  // the graph's height with it, which collapses the minimap to a blank
  // box. Filtering keeps the ordering the type map defines while
  // guaranteeing the dense array every consumer already assumed.
  return orderByBarycentre(columns.filter((c) => c && c.length), edges || []);
}

// Every lineage node id ends in ":<its own numeric primary key>" (a
// plain "TrainingRun:5", or a ServingInstance's compound
// "ServingInstance:<slug>:5") — the DB's own insertion/creation order,
// and the one a reader scans for ("where's run 4?") without being told
// to. Used below as the tiebreak that decides a column's order
// whenever grouping by source doesn't (see orderByBarycentre).
function nodeOrdinal(id) {
  const n = Number(id.slice(id.lastIndexOf(":") + 1));
  return Number.isFinite(n) ? n : 0;
}

// Group each column by its own upstream source's row, one left-to-right
// pass — a run sits with the dataset version that trained it, a model
// version with the run that produced it, and so on down the chain —
// then, within a group, order by id. Same-column edges (`derived_from`
// between two dataset versions) are excluded: both ends are in the
// column being placed, so "source" here only ever means something in
// an *earlier* column, never a sibling in the one being sorted.
//
// This one direction is deliberate, not a simplification of something
// bidirectional: an earlier version of this function also swept
// right-to-left, pulling each column toward the average row of its
// *downstream* neighbours too, on the theory that a few alternating
// passes would settle somewhere better than either direction alone.
// Reported against the live graph instead: run 5 sitting between run 3
// and run 4, both dataset version 1's, because run 5's own downstream
// model version had landed in a row that pulled it up out of dataset
// version 2's block entirely — the exact "runs read in a jumbled
// order" a lineage graph exists to not do. A run's row answers "which
// dataset trained it," not "which row did its model happen to end up
// in" — that's what the source-only pass actually encodes, and
// nothing here needs a second direction to be well-defined: column 1
// only ever depends on column 0, already finalised by the time it's
// column 1's turn, all the way across.
function orderByBarycentre(columns, edges) {
  if (columns.length < 2) return columns;
  const colOf = new Map();
  columns.forEach((col, ci) => col.forEach((id) => colOf.set(id, ci)));

  const sources = new Map();
  for (const e of edges) {
    const cs = colOf.get(e.source), ct = colOf.get(e.target);
    if (cs === undefined || ct === undefined || cs === ct) continue;
    (sources.get(e.target) || sources.set(e.target, []).get(e.target)).push(e.source);
  }

  columns[0].sort((a, b) => nodeOrdinal(a) - nodeOrdinal(b));
  const rowOf = new Map();
  columns[0].forEach((id, i) => rowOf.set(id, i));

  for (let ci = 1; ci < columns.length; ci++) {
    const col = columns[ci];
    const key = new Map();
    col.forEach((id, i) => {
      const rows = (sources.get(id) || []).map((n) => rowOf.get(n)).filter((v) => v !== undefined);
      // A node with no source in an earlier column (nothing points at
      // it from the left) keeps its discovery-order position as its
      // key, so it holds place rather than being shoved to row 0 by an
      // average it never earned.
      key.set(id, rows.length ? rows.reduce((a2, b2) => a2 + b2, 0) / rows.length : i);
    });
    col.sort((a, b) => (key.get(a) - key.get(b)) || (nodeOrdinal(a) - nodeOrdinal(b)));
    col.forEach((id, i) => rowOf.set(id, i));
  }
  return columns;
}

// Second line inside a lineage node card: whatever tells the viewer the
// most about that specific node without opening it — a status badge for
// the things that carry framework state (TrainingRun, ModelVersion,
// ServingInstance), a plain count for DatasetVersion, nothing for the
// identity-only nodes (Dataset, Model) whose label already says it all
// — plus, on top of whichever of those applies, a decision chip for any
// node a RetrainingDecision actually reached (see
// LineageManager._expand_decisions): the run it authorised, or the
// dataset version itself when it blocked before a run ever existed.
// Two independent facts about the same card, so both render rather
// than one replacing the other.
function lineageNodeMeta(n) {
  const a = n.attributes || {};
  const parts = [];
  if (n.type === "TrainingRun" && a.status) parts.push(statusBadge(a.status));
  if (n.type === "ModelVersion" && a.state) parts.push(statusBadge(a.state));
  if (n.type === "ServingInstance") {
    parts.push(el("span", { class: `badge ${a.is_active ? "success" : "cancelled"}` },
      a.is_active ? "active" : "inactive"));
  }
  if (n.type === "DatasetVersion" && a.row_count != null) {
    parts.push(el("span", { class: "faint" }, `${fmt.num(a.row_count)} rows`));
  }
  if (a.retraining_decisions && a.retraining_decisions.length) {
    parts.push(decisionChip(a.retraining_decisions));
  }
  if (!parts.length) return null;
  return parts.length === 1 ? parts[0] : el("div", { class: "meta-row" }, ...parts);
}

// The chip itself: "blocked"/"passed" rather than the raw PROMOTED/
// BLOCKED enum — this is a glance-level signal, not the detail view.
// Click opens that detail (showDecisionDetail) rather than following
// whatever link the card itself has — stopPropagation/preventDefault
// keep a click here from also activating the node's own <a> underneath
// it (lineageNodeHref makes most node types real links).
function decisionChip(decisions) {
  const latest = decisions[decisions.length - 1];
  const blocked = latest.outcome === "BLOCKED";
  const text = (blocked ? "blocked" : "passed") + (decisions.length > 1 ? ` ×${decisions.length}` : "");
  const chip = el("span", {
    class: `badge ${blocked ? "failed" : "success"} decision-chip`,
    title: "Retraining decision — click for detail",
  }, el("span", { class: "badge-text" }, text));
  chip.addEventListener("click", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    showDecisionDetail(decisions);
  });
  return chip;
}

// A minimal overlay — this app has no other modal, so no shared
// component to reuse — closed by the backdrop, Escape, or the header's
// own close button. Appended to <body>, well outside the lineage
// graph's own pan/zoom transform, so nothing here has to account for
// the canvas's current scale or scroll position.
function showDecisionDetail(decisions) {
  const close = () => { backdrop.remove(); document.removeEventListener("keydown", onKey); };
  const onKey = (ev) => { if (ev.key === "Escape") close(); };
  document.addEventListener("keydown", onKey);

  const cards = decisions.map((d) => el("div", { class: "lineage-decision-card" },
    el("div", { class: "lineage-decision-card-head" },
      el("span", { class: `badge ${statusKind(d.outcome)}` }, d.outcome),
      el("span", { class: "faint" }, `decision #${d.id}`)),
    el("dl", { class: "kv" },
      el("dt", {}, "Result"), el("dd", {}, d.label),
      ...(d.blocked_reason ? [el("dt", {}, "Reason"), el("dd", {}, d.blocked_reason)] : []),
      el("dt", {}, "Eligible"), el("dd", {}, d.eligible ? "yes" : "no"),
      el("dt", {}, "Approved"),
      el("dd", {}, d.approved == null ? "—" : d.approved ? "yes" : "no"),
      ...(d.approval_responder
        ? [el("dt", {}, "Responder"), el("dd", {}, d.approval_responder)] : []),
      ...(d.training_run_id
        ? [el("dt", {}, "Training run"),
           el("dd", {}, el("a", { href: `/runs/${d.training_run_id}` }, `#${d.training_run_id}`))]
        : []),
      ...(d.model_version_id
        ? [el("dt", {}, "Model version"), el("dd", {}, `#${d.model_version_id}`)] : []))));

  const backdrop = el("div", {
    class: "lineage-decision-backdrop",
    onclick: (ev) => { if (ev.target === backdrop) close(); },
  },
    el("div", { class: "lineage-decision-modal" },
      el("div", { class: "lineage-decision-modal-head" },
        el("h3", {}, "Retraining decision"),
        el("button", { class: "btn", "aria-label": "Close", onclick: close }, "×")),
      ...cards));
  document.body.appendChild(backdrop);
}

// Where clicking a node actually goes — its owning entity's own detail
// page, the way the rest of the console cross-links (the Run column on
// the Models page, the Dataset column on the Runs page, ...). Every node
// type has one except DatasetVersion/ModelVersion, which fold into the
// page for the Dataset/Model they belong to (no standalone
// version-detail route exists — see LineageManager's module docstring on
// why identity and version are one node) and ServingInstance, which has
// no page of its own at all; it links to the ModelVersion that's serving
// it instead, found by walking the one `served_by` edge that always
// points at it rather than trusting anything baked into the id string.
// Returns null — not a broken link — for anything unresolvable, so a
// node with data the graph can't fully explain still renders, just not
// clickable.
function lineageNodeHref(n, byId, edges) {
  const a = n.attributes || {};
  if (n.type === "DatasetVersion" && a.dataset_id != null) return `/datasets/${a.dataset_id}`;
  if (n.type === "ModelVersion" && a.model_id != null) return `/models/${a.model_id}`;
  if (n.type === "TrainingRun") {
    const runId = n.id.split(":")[1];
    return runId ? `/runs/${runId}` : null;
  }
  if (n.type === "ServingInstance") {
    const servedBy = edges.find((e) => e.target === n.id && e.type === "served_by");
    const modelId = servedBy && byId.get(servedBy.source)?.attributes?.model_id;
    return modelId != null ? `/models/${modelId}` : null;
  }
  return null;
}

// The lineage chain's four node types read as three *families* —
// dataset, execution, model+serving — and colouring + iconing by family
// (rather than one hue per type) is what actually makes a branchy graph
// scannable. Every lookup here is total (an unknown type falls through
// to "task"/grey) so a graph never renders a blank icon.
//
// Only four types: a dataset's/model's name lives inside its own
// version node's label (see LineageManager's module docstring on why
// the separate Dataset/Model identity nodes were folded away), so
// there's no second, un-versioned node per family to give its own icon
// or colour — and RetrainingDecision is a chip on a node's meta line
// now (lineageNodeMeta), not a node of its own, so it needs neither.
const LINEAGE_FAMILY = {
  DatasetVersion: "dataset",
  TrainingRun: "task",
  ModelVersion: "model",
  ServingInstance: "serving",
};

// Minimal stroke icons (feather/lucide-style paths, currentColor).
const LINEAGE_ICON_PATHS = {
  DatasetVersion: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.66 3.58 3 8 3s8-1.34 8-3V5"/><path d="M4 11v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6"/>',
  TrainingRun: '<polyline points="3 12 8 12 10 18 14 6 16 12 21 12"/>',
  ModelVersion: '<path d="M21 8l-9-5-9 5 9 5 9-5z"/><path d="M3 8v8l9 5 9-5V8"/><line x1="12" y1="13" x2="12" y2="21"/>',
  ServingInstance: '<rect x="2" y="3" width="20" height="7" rx="2"/><rect x="2" y="14" width="20" height="7" rx="2"/><line x1="6" y1="6.5" x2="6.01" y2="6.5"/><line x1="6" y1="17.5" x2="6.01" y2="17.5"/>',
};

function lineageIcon(type, small) {
  const size = small ? 12 : 15;
  const span = el("span", { class: `icon-chip family-${LINEAGE_FAMILY[type] || "task"}${small ? " sm" : ""}` });
  const svg = svgEl("svg", {
    viewBox: "0 0 24 24", width: String(size), height: String(size),
    fill: "none", stroke: "currentColor", "stroke-width": "2",
    "stroke-linecap": "round", "stroke-linejoin": "round",
  });
  // svgEl() has no innerHTML shortcut (only el() does, for plain HTML) —
  // set it directly. Every path/circle above is a fixed literal from
  // LINEAGE_ICON_PATHS, never node-supplied data, so there's nothing
  // here an API response could inject.
  svg.innerHTML = LINEAGE_ICON_PATHS[type] || LINEAGE_ICON_PATHS.TrainingRun;
  span.appendChild(svg);
  return span;
}

// Orthogonal (Manhattan) routing — horizontal and vertical runs joined
// at right angles, never a curve, the convention EDA schematic tools
// (Quartus, KiCad, ...) draw wires in. A smooth bezier reads fine for
// one edge in isolation but gets genuinely ambiguous in a dense fan —
// which of several softly-curving lines is which stays a real question
// right up until you trace one with a finger. A right-angle joint has
// no such question: a wire is either running along this row or that
// column, never something in between the two.
//
// Returns { points }: a polyline from source to target as [x, y]
// pairs, shared by the path draw and the label-placement pass below so
// a label always sits on the exact path that got drawn. Three shapes:
// - Adjacent columns (the common case): out the source's right edge,
//   one vertical jog at the horizontal midpoint, into the target's
//   left edge.
// - A skip edge (more than one column apart — no current edge type
//   produces one now that RetrainingDecision is a chip rather than a
//   node in between DatasetVersion and TrainingRun, but the shape stays
//   correct for the day a future node type sits between two others
//   again) whose straight path would run through a node sitting in a
//   column in between: the same shape with an extra horizontal leg at
//   a row clear of that node — above or below it, whichever side is
//   already closer to this edge's own path, so the detour stays small
//   rather than every such edge routing to the same side regardless of
//   where it was headed.
// - Same column (`derived_from` between two dataset versions — the
//   only edge connecting two nodes of the same type): a same-rank edge
//   has nowhere to the side to run through, so it exits and re-enters
//   the node's own right edge — a loop out and back, square-cornered
//   instead of a bow.
function lineageEdgeGeometry(e, pos, colOf, NODE_W, NODE_H) {
  const from = pos.get(e.source), to = pos.get(e.target);
  if (colOf.get(e.source) === colOf.get(e.target)) {
    // Two hubs per node, not two per column-pair: `from` is always
    // playing the *outgoing* role for this edge, so it uses the exact
    // same centre-of-right-edge point every other outgoing edge off
    // that node already uses (trained_with, authorized, ...) — a node
    // fanning out to several targets is supposed to share one point.
    // `to` is always playing *incoming*, and gets a second, offset
    // point reserved for that role, so a node that is both a
    // same-column source and a same-column target (a DatasetVersion
    // with an incoming derived_from and an outgoing evaluated_by, say)
    // never puts its own incoming and outgoing edges on the same pixel.
    const y1 = from.y + NODE_H / 2, y2 = to.y + NODE_H * 0.82;
    const x1 = from.x + NODE_W, x2 = to.x + NODE_W;
    const bowX = Math.max(x1, x2) + 64;
    return { points: [[x1, y1], [bowX, y1], [bowX, y2], [x2, y2]] };
  }
  const y1 = from.y + NODE_H / 2, y2 = to.y + NODE_H / 2;
  const x1 = from.x + NODE_W, x2 = to.x;
  const midX = (x1 + x2) / 2;

  // A skip edge can have another node sitting in a column it passes
  // over. The default route here stays within the [y1, y2] band the
  // whole way across, so an obstruction's card — opaque, painted above
  // the SVG — would sit right on top of the straight run through it.
  const lo = Math.min(colOf.get(e.source), colOf.get(e.target));
  const hi = Math.max(colOf.get(e.source), colOf.get(e.target));
  if (hi - lo > 1) {
    const rowLo = Math.min(y1, y2), rowHi = Math.max(y1, y2);
    // The *combined* span of every obstructing node, not just the
    // nearest one — a `promoted`/`rejected` edge from RetrainingDecision
    // straight to a ModelVersion skips the entire TrainingRun column,
    // several rows at once, not just whichever run happens to sit
    // closest to this edge's own midpoint. Routing around only that one
    // still ran straight through the others (confirmed against the
    // live graph: a stray line grazing a training run's card that this
    // edge has nothing to do with). Detouring past the full min-top/
    // max-bottom extent of everything in the way clears all of them in
    // one move, the same way a real schematic routes a signal around an
    // entire block of components rather than weaving between them.
    // NODE_H (64) is the *layout* grid unit ROW_H is spaced around, not
    // the real rendered height of a card once its icon, type, label and
    // meta badge are all stacked — measured against the live graph,
    // that's 91-110px, not 64. Sizing the obstruction box off the
    // layout constant instead of the real one is exactly what let a
    // detour compute as "clear" while still running behind an actual
    // card: 22px past a 64px box landed inside a 94px-tall one.
    // OB_HALF_H is a deliberately generous stand-in for "half the real
    // height of whatever this turns out to be," centred on the same
    // point (p.y + NODE_H / 2) the layout already treats as this node's
    // middle — this function has no access to the DOM to measure the
    // actual box, so a fixed constant with real headroom is the
    // practical fix, not a regression waiting to happen the next time
    // a card grows a line.
    const OB_HALF_H = 60;
    let obTop = null, obBottom = null;
    for (const [id, p] of pos) {
      const c = colOf.get(id);
      if (c === undefined || c <= lo || c >= hi) continue;
      const centerY = p.y + NODE_H / 2;
      const top = centerY - OB_HALF_H, bottom = centerY + OB_HALF_H;
      if (bottom < rowLo || top > rowHi) continue; // this node's row is clear of this edge's span
      obTop = obTop === null ? top : Math.min(obTop, top);
      obBottom = obBottom === null ? bottom : Math.max(obBottom, bottom);
    }
    if (obTop !== null) {
      const clearance = 20;
      const midY = (y1 + y2) / 2;
      const detourY = midY <= (obTop + obBottom) / 2
        ? obTop - clearance
        : obBottom + clearance;
      const xa = x1 + (x2 - x1) / 3, xb = x1 + (2 * (x2 - x1)) / 3;
      return { points: [[x1, y1], [xa, y1], [xa, detourY], [xb, detourY], [xb, y2], [x2, y2]] };
    }
  }

  return { points: [[x1, y1], [midX, y1], [midX, y2], [x2, y2]] };
}

// The node-link graph itself: positions come straight from
// (column, row) on a fixed grid via lineageLevels(), same approach as
// renderDagGraph, with one SVG overlay drawing an arrowed, labelled
// curve per edge. `rootId` gets a highlighted ring so a chain that
// fans out in both directions (a training run reading several dataset
// versions and feeding several model versions) still shows where the
// walk started. Returns `{ el, width, height, positions }` — the panel
// wrapper needs the raw geometry too, to draw the minimap in the same
// coordinate space without recomputing the layout a second time.
function renderLineageGraph(nodes, edges, rootId) {
  // COL_W is double the node width plus a healthy gap (was ~1.2x) — the
  // label-collision fixes above still left column gaps tight enough
  // that a fan-in/fan-out group's curves stayed visually bunched right
  // up against the node edges they left/entered, per user report on
  // model-version=5's 3-way fan-in. Doubling the horizontal gap between
  // columns gives every curve (and its label) real room to separate
  // before the next node's edge starts.
  const COL_W = 496, ROW_H = 112, NODE_W = 208, NODE_H = 64, PAD = 12;
  const levels = lineageLevels(nodes, edges);
  const maxRows = Math.max(1, ...levels.map((c) => c.length));
  const pos = new Map();
  const colOf = new Map();
  levels.forEach((col, ci) => {
    // Centre each column against the tallest one instead of hanging
    // every column from the top. Top-aligning left the single serving
    // instance level with the *first* of five training runs rather than
    // with the model it serves, so short columns pulled all their edges
    // into a diagonal towards row 0 and the graph carried a block of
    // dead space under them. Half-row offsets are fine — nothing here
    // requires a node to sit on an integer row.
    const offset = (maxRows - col.length) / 2;
    col.forEach((id, ri) => {
      pos.set(id, { x: PAD + ci * COL_W, y: PAD + (ri + offset) * ROW_H });
      colOf.set(id, ci);
    });
  });
  const width = PAD * 2 + Math.max(0, levels.length - 1) * COL_W + NODE_W;
  const height = PAD * 2 + (maxRows - 1) * ROW_H + NODE_H;

  const svg = svgEl("svg", {
    style: `position:absolute;inset:0;width:${width}px;height:${height}px`,
    viewBox: `0 0 ${width} ${height}`,
  },
    svgEl("defs", {},
      svgEl("marker", {
        id: "lineage-arrow", viewBox: "0 0 8 8", refX: "7", refY: "4",
        markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse",
      }, svgEl("path", { d: "M0,0 L8,4 L0,8 z", class: "lineage-edge-arrow" }))));

  const validEdges = edges.filter((e) => pos.has(e.source) && pos.has(e.target));
  const geoms = validEdges.map((e) => lineageEdgeGeometry(e, pos, colOf, NODE_W, NODE_H));

  // Two orthogonal wires crossing on the page read as ambiguous —
  // "did these join, or did they just happen to cross?" — the one
  // question this graph never actually has an answer of "joined" for:
  // every edge here is its own independent source-to-target run: none
  // of them share a segment or terminate mid-path into another one.
  // Schematic tools resolve exactly this ambiguity with a small hop —
  // one wire jumps clear over the other rather than drawing through it
  // — and that convention is what's missing here, not a routing bug:
  // the lines are exactly where they should be, they just look like
  // junctions where two happen to cross. Only horizontal legs get the
  // hop (collected from every OTHER edge's vertical legs) — the
  // reverse case, a vertical leg crossing a horizontal one, never
  // actually arises in this layout (verticals only ever run inside a
  // column's own jog; two edges' horizontals only ever meet at a
  // shared row, which the row-hub work three commits back already
  // keeps from happening on the same pixel).
  const verticals = [];
  geoms.forEach((g, ei) => {
    for (let i = 1; i < g.points.length; i++) {
      const [ax, ay] = g.points[i - 1], [bx, by] = g.points[i];
      if (Math.abs(ax - bx) < 0.5 && Math.abs(ay - by) > 0.5) {
        verticals.push({ x: ax, y1: Math.min(ay, by), y2: Math.max(ay, by), ei });
      }
    }
  });

  const HOP_R = 6;
  function horizontalWithHops(x1, y, x2, ei) {
    const dir = x2 >= x1 ? 1 : -1;
    const lo = Math.min(x1, x2), hi = Math.max(x1, x2);
    const hops = verticals
      .filter((v) => v.ei !== ei && v.x > lo + HOP_R * 2 && v.x < hi - HOP_R * 2
        && v.y1 < y - 1 && v.y2 > y + 1)
      .map((v) => v.x)
      .sort((a, b) => (a - b) * dir);
    let d = "", cur = x1;
    for (const hx of hops) {
      const before = hx - dir * HOP_R, after = hx + dir * HOP_R;
      d += `L${before},${y} A${HOP_R},${HOP_R} 0 0 ${dir > 0 ? 1 : 0} ${after},${y} `;
      cur = after;
    }
    return d + `L${x2},${y}`;
  }

  geoms.forEach((g, ei) => {
    let d = `M${g.points[0][0]},${g.points[0][1]}`;
    for (let i = 1; i < g.points.length; i++) {
      const [ax, ay] = g.points[i - 1], [bx, by] = g.points[i];
      d += " " + (Math.abs(ay - by) < 0.5 && Math.abs(ax - bx) > 0.5
        ? horizontalWithHops(ax, ay, bx, ei)
        : `L${bx},${by}`);
    }
    svg.appendChild(svgEl("path", {
      class: "lineage-edge", d, fill: "none", "marker-end": "url(#lineage-arrow)",
    }));
  });

  // Labels, drawn in a second pass and in two stages:
  //
  // 1. Anchor each label to its own edge's last horizontal run — the
  //    leg closest to the target, after the path has actually turned
  //    toward wherever *this* edge is going. A fan of edges sharing a
  //    source
  //    (one dataset version trained_with four different runs) shares
  //    that source's whole first leg too, right down to the pixel, so
  //    anchoring anywhere on it piles every one of those labels into
  //    the same small cluster regardless of how the group as a whole
  //    gets spread out — confirmed against the live graph: four
  //    "trained with" tags stacked edge-to-edge a few px apart, right
  //    where the lines hadn't yet diverged. The last horizontal leg —
  //    the short run directly into the target's own left edge — has no
  //    such problem: every edge has its own, sitting right where the
  //    label reads most naturally, immediately before the node it
  //    names a relationship to, rather than out in the middle of the
  //    canvas on the vertical jog that got it there.
  // 2. That still leaves the rarer case of two edges whose last legs
  //    land close regardless (two targets one row apart, say) — so
  //    every placed label is checked against every earlier one by
  //    simple box overlap and nudged vertically until clear. An opaque
  //    pill behind the text (not just a stroke halo) keeps whatever
  //    overlap survives both passes from reading as a smear.
  function edgeLabelAnchor(points) {
    for (let i = points.length - 1; i > 0; i--) {
      const [ax, ay] = points[i - 1], [bx, by] = points[i];
      if (Math.abs(ay - by) < 0.5 && Math.abs(ax - bx) > 0.5) {
        return { x: (ax + bx) / 2, y: ay };
      }
    }
    // No horizontal leg at all (shouldn't happen — every shape this
    // function draws ends with one — but a same-row edge with a
    // zero-length final leg falls back to its own midpoint).
    const [ax, ay] = points[points.length - 2], [bx, by] = points[points.length - 1];
    return { x: (ax + bx) / 2, y: (ay + by) / 2 };
  }
  const labels = [];
  geoms.forEach((g, ei) => {
    const { x, y } = edgeLabelAnchor(g.points);
    const text = validEdges[ei].type.replace(/_/g, " ");
    labels.push({ x, y, w: text.length * 5.6 + 10, h: 14, text });
  });
  const placedBoxes = [];
  for (const lb of labels) {
    let y = lb.y, dir = 1, step = 0, box;
    for (let tries = 0; tries < 14; tries++) {
      box = { left: lb.x - lb.w / 2 - 4, right: lb.x + lb.w / 2 + 4, top: y - lb.h - 1, bottom: y + 1 };
      const hit = placedBoxes.some((p) =>
        box.left < p.right && box.right > p.left && box.top < p.bottom && box.bottom > p.top);
      if (!hit) break;
      step += 8;
      y = lb.y + dir * step;
      dir *= -1;
    }
    // Registered even on the rare exhausted-retries case (a box still
    // overlapping something) — later labels nudging away from a known
    // occupied spot beats them not knowing about it at all.
    placedBoxes.push(box);
    svg.appendChild(svgEl("rect", {
      class: "lineage-edge-label-bg",
      x: String(lb.x - lb.w / 2), y: String(y - lb.h - 1), width: String(lb.w), height: String(lb.h), rx: "3",
    }));
    svg.appendChild(svgEl("text", {
      class: "lineage-edge-label", x: String(lb.x), y: String(y - 5), "text-anchor": "middle",
    }, lb.text));
  }

  const byId = new Map(nodes.map((n) => [n.id, n]));
  const nodeEls = nodes.map((n) => {
    const p = pos.get(n.id);
    if (!p) return null;
    const meta = lineageNodeMeta(n);
    const href = lineageNodeHref(n, byId, edges);
    return el(href ? "a" : "div", {
      class: `lineage-node ${n.type}${n.id === rootId ? " root" : ""}`,
      style: `position:absolute;left:${p.x}px;top:${p.y}px;width:${NODE_W}px`,
      title: href ? `${n.label || n.id} — open detail page` : (n.label || n.id),
      ...(href ? { href } : {}),
    },
      lineageIcon(n.type),
      el("div", { class: "text" },
        el("div", { class: "type" }, n.type.replace(/([a-z])([A-Z])/g, "$1 $2")),
        el("div", { class: "label" }, n.label || n.id),
        meta && el("div", { class: "meta" }, meta)));
  }).filter(Boolean);

  return { el: el("div",
    { class: "lineage-graph", style: `position:relative;width:${width}px;height:${height}px` },
    svg, ...nodeEls), width, height, pos, nodeW: NODE_W, nodeH: NODE_H };
}

// Colour key above the graph: one icon chip per node type actually
// present, in chain order, using the exact same family icon/colour the
// nodes themselves use so the legend is a real key and not a second
// vocabulary to cross-reference.
function renderLineageLegend(nodes) {
  const order = ["DatasetVersion", "TrainingRun", "ModelVersion", "ServingInstance"];
  const present = order.filter((t) => nodes.some((n) => n.type === t));
  return el("div", { class: "legend lineage-legend" },
    el("span", { class: "legend-label" }, "Legend"),
    ...present.map((t) => el("span", { class: "legend-item" },
      lineageIcon(t, true), t.replace(/([a-z])([A-Z])/g, "$1 $2"))));
}

// A small overview in the corner of the viewport: every node scaled
// down to a coloured dot in the same relative layout as the full graph,
// plus a rectangle tracking the viewport's current scroll/zoom window.
// Clicking anywhere on it re-centres the main viewport there — the
// point of a minimap on a graph wide enough to need one is jumping
// around without scrolling blind.
function renderLineageMinimap(nodes, graph, viewport) {
  const MW = 168, MH = 104, PAD = 6;
  const scale = Math.min((MW - PAD * 2) / graph.width, (MH - PAD * 2) / graph.height);
  const ox = (MW - graph.width * scale) / 2, oy = (MH - graph.height * scale) / 2;

  const dots = nodes.map((n) => {
    const p = graph.pos.get(n.id);
    if (!p) return null;
    return svgEl("rect", {
      x: String(ox + p.x * scale), y: String(oy + p.y * scale),
      width: String(Math.max(3, graph.nodeW * scale)), height: String(Math.max(3, graph.nodeH * scale)),
      rx: "1.5", class: `lineage-minimap-node family-${LINEAGE_FAMILY[n.type] || "task"}`,
    });
  }).filter(Boolean);

  const svg = svgEl("svg", { viewBox: `0 0 ${MW} ${MH}`, width: String(MW), height: String(MH) }, ...dots);
  const viewRect = el("div", { class: "lineage-minimap-viewport" });
  const mini = el("div", { class: "lineage-minimap" }, svg, viewRect);

  function syncRect() {
    const s = graph.el.style.transform.match(/scale\(([\d.]+)\)/);
    const zoom = s ? +s[1] : 1;
    const vw = Math.min(graph.width, viewport.clientWidth / zoom);
    const vh = Math.min(graph.height, viewport.clientHeight / zoom);
    const vx = viewport.scrollLeft / zoom;
    const vy = viewport.scrollTop / zoom;
    viewRect.style.left = `${ox + vx * scale}px`;
    viewRect.style.top = `${oy + vy * scale}px`;
    viewRect.style.width = `${Math.max(6, vw * scale)}px`;
    viewRect.style.height = `${Math.max(6, vh * scale)}px`;
  }
  viewport.addEventListener("scroll", syncRect);
  new ResizeObserver(syncRect).observe(viewport);
  syncRect();

  mini.addEventListener("click", (e) => {
    const rect = mini.getBoundingClientRect();
    const mx = (e.clientX - rect.left - ox) / scale;
    const my = (e.clientY - rect.top - oy) / scale;
    const s = graph.el.style.transform.match(/scale\(([\d.]+)\)/);
    const zoom = s ? +s[1] : 1;
    viewport.scrollTo({
      left: Math.max(0, mx * zoom - viewport.clientWidth / 2),
      top: Math.max(0, my * zoom - viewport.clientHeight / 2),
      behavior: "smooth",
    });
  });

  return { el: mini, syncRect };
}

// Graph + icon legend + a floating zoom/fullscreen cluster + minimap,
// all inside one card — the same pieces the graph screenshots that
// prompted this design carry, built out of Gateflow's own tokens rather
// than copied wholesale. Zoom scales the graph via CSS transform (cheap,
// keeps the SVG edges in sync for free); panning is native scroll on
// the bordered, dot-grid viewport; the zoom/fullscreen controls float
// over that viewport's corner (outside the scrolling element) so they
// stay put as you pan, the way a map's own zoom controls do.
function renderLineageGraphPanel(nodes, edges, rootId) {
  const graph = renderLineageGraph(nodes, edges, rootId);
  const viewport = el("div", { class: "lineage-graph-viewport" }, graph.el);
  const minimap = renderLineageMinimap(nodes, graph, viewport);

  const ZOOM_MIN = 0.15, ZOOM_MAX = 1.6;
  let scale = 1;
  const applyScale = () => { graph.el.style.transform = `scale(${scale})`; minimap.syncRect(); };
  // Scales down (never up past 1:1) to whatever fits the viewport's
  // current size, width and height both — the number a "fit to view"
  // control computes everywhere, and also what should greet someone
  // landing on the page rather than a 1:1 view of just its top-left
  // corner. Columns are double-spaced now (see COL_W above), so most
  // real graphs are wider than the page; this is what makes that
  // still open showing the whole shape.
  const fitToView = () => {
    const fit = Math.min(1,
      (viewport.clientWidth - 16) / graph.width,
      (viewport.clientHeight - 16) / graph.height);
    scale = Math.max(ZOOM_MIN, +fit.toFixed(2));
    applyScale();
  };
  const zoomOut = el("button", {
    class: "btn", type: "button", "aria-label": "Zoom out",
    onclick: () => { scale = Math.max(ZOOM_MIN, +(scale - 0.15).toFixed(2)); applyScale(); },
  }, "−");
  const zoomIn = el("button", {
    class: "btn", type: "button", "aria-label": "Zoom in",
    onclick: () => { scale = Math.min(ZOOM_MAX, +(scale + 0.15).toFixed(2)); applyScale(); },
  }, "+");
  const fitBtn = el("button", {
    class: "btn", type: "button", "aria-label": "Fit to view", title: "Fit to view",
    onclick: fitToView,
  }, "⤢");
  const fullscreen = el("button", {
    class: "btn", type: "button", "aria-label": "Toggle fullscreen",
    onclick: () => {
      if (document.fullscreenElement) document.exitFullscreen();
      else viewport.requestFullscreen?.().catch(() => {});
    },
  }, "⛶");
  const controls = el("div", { class: "lineage-zoom-controls" }, zoomIn, fitBtn, zoomOut, fullscreen);

  // Deferred a frame: clientWidth/clientHeight read 0 until the panel
  // this function returns is actually attached to the document, which
  // happens synchronously in the caller right after this returns — by
  // the next frame it's laid out and these are real numbers.
  requestAnimationFrame(fitToView);

  return el("div", { class: "card lineage-graph-card" },
    renderLineageLegend(nodes),
    el("div", { class: "lineage-canvas" }, viewport, controls, minimap.el));
}

async function initLineage() {
  const params = new URLSearchParams(location.search);
  const kind = params.get("kind");
  const id = params.get("id");
  const out = document.getElementById("lineage-out");

  if (!kind || !id) {
    await renderLineagePicker(out);
    return;
  }
  try {
    const g = await api(`/lineage/${kind}/${id}`);
    if (!g.nodes.length) {
      out.replaceChildren(banner("No lineage found for this starting point."));
      return;
    }
    const byId = new Map(g.nodes.map((n) => [n.id, n]));

    out.replaceChildren(
      renderLineageGraphPanel(g.nodes, g.edges, g.root_id),
      el("details", { class: "lineage-edges-detail" },
        el("summary", {}, `Edges (${g.edges.length})`),
        el("div", { class: "table-wrap", style: "margin-top:10px" },
          el("table", {},
            el("thead", {}, el("tr", {}, el("th", {}, "From"), el("th", {}, "Relation"), el("th", {}, "To"))),
            el("tbody", {}, ...g.edges.map((e) =>
              el("tr", {},
                el("td", { class: "mono" }, byId.get(e.source)?.label || e.source),
                el("td", { class: "muted" }, e.type),
                el("td", { class: "mono" }, byId.get(e.target)?.label || e.target))))))));
  } catch (e) {
    setError(out, e);
  }
}


/* ------------------------------------------------------------------ */
/* MLflow model registry reconciliation                                */
/* ------------------------------------------------------------------ */

// The framework promotes versions in its own table; since CP1/CP2 that
// promotion also pushes into MLflow's own registry (see
// mlops_framework.tracking.mlflow_registry) — this panel is where any
// remaining disagreement between the two would still show up (MLflow
// down at promote time, someone changing state by hand on either side).
// Per-version columns (MLflow version/stage/alias, and a "disagrees"
// flag) are merged directly into the model detail page's own Versions
// table via ``onVersions`` — showing the two side by side in one table
// is the point, not a second table repeating the same version numbers.
// This function renders only the top-of-page summary banner and any
// registry entries the framework doesn't know about at all.
// The framework's own ModelState is the badge a reader needs on every
// row (see renderVersionsTable) — MLflow's registry ("stage", its own
// version numbering, aliases) is a second vocabulary for the same
// question, worth showing only when it actually disagrees or when
// someone deliberately goes looking, not as a permanent fixture next to
// every version. So: a disagreement stays an open, red banner (that is
// the one case this panel exists to catch); everything else — the
// "agrees" confirmation, the registry name, anything registered in
// MLflow the framework doesn't know about — lives behind a closed
// <details>, collapsed by default.
function renderRegistrySummary(host, modelId, onVersions) {
  api(`/models/${modelId}/registry-reconciliation`).then((p) => {
    if (!p.available) {
      host.replaceChildren(banner(`MLflow registry: ${p.reason}`, "warn"));
      return;
    }
    const d = p.data;
    const rows = d.versions || [];
    onVersions(new Map(rows.map((v) => [v.framework_version_id, v])));

    const drifted = rows.filter((v) => v.drift);
    const registryLine = el("span", { class: "faint" },
      d.registry_names?.length
        ? `registered as ${d.registry_names.join(", ")}`
        : "no matching registered model");

    const mlflowOnlyTable = (d.mlflow_only || []).length
      ? el("div", { style: "margin-top:16px" },
          el("div", { class: "chart-title", style: "margin-bottom:8px" },
            "Registered in MLflow but unknown to this framework"),
          el("div", { class: "table-wrap" },
            el("table", {},
              el("thead", {}, el("tr", {},
                el("th", {}, "Name"), el("th", {}, "Version"),
                el("th", {}, "Run"), el("th", {}, "Aliases"))),
              el("tbody", {}, ...d.mlflow_only.map((m) =>
                el("tr", {},
                  el("td", { class: "mono" }, m.mlflow_name),
                  el("td", { class: "mono" }, m.mlflow_version),
                  el("td", { class: "mono truncate", title: m.run_id }, m.run_id),
                  el("td", { class: "mono" }, (m.aliases || []).join(", ") || "—")))))))
      : null;

    if (d.drift_count) {
      mount(host,
        el("div", { class: "section-head" }, el("h3", {}, "MLflow registry"), registryLine),
        banner(
          `${d.drift_count} version${d.drift_count > 1 ? "s" : ""} disagree ` +
          `between this framework and MLflow: ` +
          drifted.map((v) => `v${v.framework_version_number} — ${v.drift_reason}`).join("; "),
          "err"),
        mlflowOnlyTable);
      return;
    }
    mount(host,
      el("details", {},
        el("summary", { class: "faint" }, "MLflow registry — agrees, ", registryLine),
        el("div", { style: "margin-top:10px" }, mlflowOnlyTable)));
  }).catch(() => {});
}

/* ------------------------------------------------------------------ */
/* Pipelines (Airflow)                                                 */
/* ------------------------------------------------------------------ */

function airflowHealthCard(data) {
  const health = data.health || {};
  const importErrors = data.import_errors || [];
  const pools = data.pools || [];

  const components = Object.entries(health).map(([name, info]) => {
    const status = (info && info.status) || null;
    const kind = status === "healthy" ? "ok" : status ? "err" : "";
    return el("div", { class: `kpi ${kind}` },
      el("div", { class: "label" }, name.replace(/_/g, " ")),
      el("div", { class: "value" }, status || "n/a"));
  });

  const poolRows = pools.map((p) =>
    el("div", { class: "kpi" },
      el("div", { class: "label" }, p.name),
      el("div", { class: "value" }, `${p.running_slots}/${p.slots}`)));

  return el("div", {},
    components.length || poolRows.length
      ? el("div", { class: "kpi-grid", style: "margin-bottom:12px" }, ...components, ...poolRows)
      : null,
    importErrors.length
      ? el("div", { class: "banner err", style: "margin-bottom:12px" },
          el("strong", {}, `${importErrors.length} DAG file${importErrors.length > 1 ? "s" : ""} failed to parse: `),
          importErrors.map((e) => e.filename).join(", "))
      : null);
}

async function initPipelines() {
  const healthHost = document.getElementById("pipelines-health");
  const out = document.getElementById("pipelines-out");

  api("/airflow/health").then((p) => {
    if (!p.available) { healthHost.replaceChildren(banner(p.reason, "warn")); return; }
    healthHost.replaceChildren(airflowHealthCard(p.data));
  }).catch(() => {});

  let p;
  try {
    p = await api("/airflow/dags");
  } catch (e) {
    setError(out, e);
    return;
  }
  if (!p.available) {
    out.replaceChildren(banner(p.reason, "warn"));
    return;
  }

  const dags = p.data.dags || [];
  const table = el("table", {}, el("thead", {}, el("tr", {})), el("tbody", {}));
  out.replaceChildren(el("div", { class: "table-wrap" }, table));

  makeSortable(table, dags,
    [
      { label: "DAG", sort: (d) => d.dag_id },
      { label: "Status" },
      { label: "Schedule" },
      { label: "Next run", sort: (d) => d.next_dagrun },
      { label: "Owners" },
      { label: "Tags" },
    ],
    (d) => el("tr", {},
      el("td", {}, el("a", { href: `/pipelines/${encodeURIComponent(d.dag_id)}` }, d.dag_id)),
      el("td", {}, el("span", { class: `badge ${d.is_paused ? "cancelled" : "success"}` },
        d.is_paused ? "Paused" : "Active")),
      el("td", { class: "mono faint" }, d.schedule_interval || "manual / external only"),
      el("td", { class: "muted nowrap" }, d.next_dagrun ? fmt.time(d.next_dagrun) : "—"),
      el("td", { class: "muted" }, (d.owners || []).join(", ") || "—"),
      el("td", { class: "muted" }, (d.tags || []).join(", ") || "—")));
}

// Layers `tasks` into columns for a general DAG graph view: level(root)
// = 0, level(n) = 1 + max(level(upstream)) over every incoming edge —
// Kahn's topological order, so a node is only placed once every one of
// its upstreams already has a level (a plain BFS-from-roots would
// under-place a join node fed by branches of different lengths).
// A single unbranched chain is just the special case where every
// column holds exactly one task, so this replaces the old
// linear-only chain renderer rather than sitting beside it.
//
// Returns an array of columns (each an array of task_id), or null if
// an edge points at an unknown task_id or a cycle leaves some node
// unplaceable — either means the structure can't be trusted enough to
// draw, and the caller falls back to the plain task table.
function dagLevels(tasks) {
  const byId = new Map(tasks.map((t) => [t.task_id, t]));
  const indegree = new Map(tasks.map((t) => [t.task_id, 0]));
  for (const t of tasks) {
    for (const d of t.downstream_task_ids || []) {
      if (!byId.has(d)) return null;
      indegree.set(d, (indegree.get(d) || 0) + 1);
    }
  }

  const level = new Map();
  const remaining = new Map(indegree);
  const placed = new Set();
  let frontier = tasks.filter((t) => (indegree.get(t.task_id) || 0) === 0).map((t) => t.task_id);
  if (frontier.length === 0 && tasks.length > 0) return null;
  frontier.forEach((id) => { level.set(id, 0); placed.add(id); });

  while (frontier.length) {
    const next = [];
    for (const id of frontier) {
      for (const d of byId.get(id).downstream_task_ids || []) {
        level.set(d, Math.max(level.get(d) || 0, level.get(id) + 1));
        remaining.set(d, remaining.get(d) - 1);
        if (remaining.get(d) === 0 && !placed.has(d)) {
          placed.add(d);
          next.push(d);
        }
      }
    }
    frontier = next;
  }
  if (placed.size !== tasks.length) return null;

  const columns = [];
  for (const t of tasks) {
    const lvl = level.get(t.task_id) || 0;
    (columns[lvl] || (columns[lvl] = [])).push(t.task_id);
  }
  return columns;
}

// Renders `tasks` as a layered graph: node positions come straight from
// (column, row) on a fixed grid — computed directly rather than
// measured from the DOM after paint, the same approach lineChart/
// barChart already use below — with a single SVG overlay drawing one
// curve per downstream edge between those same computed points, so
// edges can never drift out of sync with the nodes they connect.
// `stateByTaskId` (task_id -> Airflow state string) is optional; when
// given, nodes are coloured by it (the console passes the latest run's
// states), otherwise nodes render neutral.
function renderDagGraph(tasks, levels, stateByTaskId) {
  const COL_W = 210, ROW_H = 72, NODE_W = 180, NODE_H = 44, PAD = 16;
  const byId = new Map(tasks.map((t) => [t.task_id, t]));
  const pos = new Map();
  levels.forEach((col, ci) => {
    col.forEach((tid, ri) => pos.set(tid, { x: PAD + ci * COL_W, y: PAD + ri * ROW_H }));
  });
  const maxRows = Math.max(1, ...levels.map((c) => c.length));
  const width = PAD * 2 + (levels.length - 1) * COL_W + NODE_W;
  const height = PAD * 2 + (maxRows - 1) * ROW_H + NODE_H;

  const svg = svgEl("svg", {
    style: `position:absolute;inset:0;width:${width}px;height:${height}px`,
    viewBox: `0 0 ${width} ${height}`,
  });
  for (const t of tasks) {
    const from = pos.get(t.task_id);
    for (const d of t.downstream_task_ids || []) {
      const to = pos.get(d);
      if (!from || !to) continue;
      const x1 = from.x + NODE_W, y1 = from.y + NODE_H / 2;
      const x2 = to.x, y2 = to.y + NODE_H / 2;
      const midX = (x1 + x2) / 2;
      svg.appendChild(svgEl("path", {
        class: "dag-edge",
        d: `M${x1},${y1} C${midX},${y1} ${midX},${y2} ${x2},${y2}`,
        fill: "none",
      }));
    }
  }

  const nodes = tasks.map((t) => {
    const p = pos.get(t.task_id);
    const state = stateByTaskId && stateByTaskId.get(t.task_id);
    const kind = state ? statusKind(state) : "";
    return el("div", {
      class: `lineage-node Task${kind ? ` state-${kind}` : ""}`,
      style: `position:absolute;left:${p.x}px;top:${p.y}px;width:${NODE_W}px`,
      title: state ? `${t.task_id} — ${state}` : t.task_id,
    },
      el("div", { class: "type" }, t.operator_name || "task"),
      el("div", { class: "label" }, t.task_id));
  });

  return el("div",
    { class: "dag-graph", style: `position:relative;width:${width}px;height:${height}px` },
    svg, ...nodes);
}

// Airflow-Tree-View-style grid: one row per task (declaration order),
// one column per run (newest first, matching the "Recent runs" table
// below it). Cells come from the task-instance data `/airflow/dags/
// {id}` already expanded server-side (`grid_cells`) — no per-cell
// request. A cell only links to the framework's own run page when
// `byExecutionId` resolves one; a scheduler-triggered run has no
// framework-side row to link to, same distinction "Recent runs" draws.
function renderTaskHistoryGrid(tasks, gridRunIds, gridCells, byExecutionId, dagId) {
  if (!gridRunIds.length) {
    return banner("No run history yet for this DAG.");
  }
  const byTaskThenRun = new Map();
  for (const c of gridCells) {
    if (!byTaskThenRun.has(c.task_id)) byTaskThenRun.set(c.task_id, new Map());
    byTaskThenRun.get(c.task_id).set(c.dag_run_id, c);
  }

  const header = el("tr", {},
    el("th", {}, "Task"),
    ...gridRunIds.map((rid) => el("th", { class: "mono" }, rid.replace(/^mlops-/, ""))));

  const rows = (tasks.length ? tasks : [...byTaskThenRun.keys()].map((task_id) => ({ task_id })))
    .map((t) => {
      const byRun = byTaskThenRun.get(t.task_id) || new Map();
      return el("tr", {},
        el("td", { class: "mono" }, t.task_id),
        ...gridRunIds.map((rid) => {
          const cell = byRun.get(rid);
          if (!cell) return el("td", {}, el("span", { class: "tree-cell" }));
          const kind = statusKind(cell.state);
          const fwId = byExecutionId.get(`${dagId}/${rid}`);
          const title = `${cell.state}`
            + (cell.duration != null ? ` · ${cell.duration.toFixed(1)}s` : "")
            + ` · ${rid}`;
          const swatch = el("span", { class: `tree-cell ${kind}`, title });
          return el("td", {}, fwId
            ? el("a", { class: "tree-cell-link", href: `/runs/${fwId}`, "aria-label": title }, swatch)
            : swatch);
        }));
    });

  return el("div", { class: "table-wrap" },
    el("table", { class: "tree-grid" },
      el("thead", {}, header),
      el("tbody", {}, ...rows)));
}

async function initPipelineDetail(dagId) {
  const head = document.getElementById("pipeline-head");
  const body = document.getElementById("pipeline-body");

  mount(head,
    el("div", { class: "breadcrumb" }, el("a", { href: "/pipelines" }, "Pipelines"), " / ", dagId),
    el("h2", { class: "mono" }, dagId));

  // Runs this framework itself started carry the composite execution id
  // "dag_id/dag_run_id" — cross-linking back to them is what tells "a
  // scheduler-triggered run" and "a run this console already knows about"
  // apart in the history table below.
  let byExecutionId = new Map();
  try {
    const runs = await api("/training-runs?limit=500");
    byExecutionId = new Map(
      runs.filter((r) => r.execution_id).map((r) => [r.execution_id, r.id]));
  } catch { /* the cross-link is a bonus, not a requirement */ }

  let p;
  try {
    p = await api(`/airflow/dags/${encodeURIComponent(dagId)}`);
  } catch (e) {
    setError(body, e);
    return;
  }
  if (!p.available) {
    body.replaceChildren(banner(p.reason, "warn"));
    return;
  }

  const tasks = p.data.tasks || [];
  const dagRuns = p.data.dag_runs || [];
  const gridRunIds = p.data.grid_run_ids || [];
  const gridCells = p.data.grid_cells || [];

  // One lookup per run, built once — switching the run selector below is
  // then just a Map read, no re-scan of gridCells.
  const statesByRun = new Map();
  for (const c of gridCells) {
    if (!statesByRun.has(c.dag_run_id)) statesByRun.set(c.dag_run_id, new Map());
    statesByRun.get(c.dag_run_id).set(c.task_id, c.state);
  }
  const levels = tasks.length ? dagLevels(tasks) : null;

  // Newest run first, matching the grid header and the run picker below.
  const graphHost = el("div", { class: "table-wrap" });
  function paintGraph(runId) {
    const stateByTask = statesByRun.get(runId) || new Map();
    const graph = levels ? renderDagGraph(tasks, levels, stateByTask) : null;
    mount(graphHost, graph || banner("No task structure to draw yet."));
  }
  if (gridRunIds.length) paintGraph(gridRunIds[0]);
  else mount(graphHost, banner("This DAG has no run history yet — nothing to colour."));

  const runPicker = gridRunIds.length
    ? el("label", { class: "run-picker" },
        "Run: ",
        el("select", {
          onchange: (e) => paintGraph(e.target.value),
        }, ...gridRunIds.map((rid, i) =>
          el("option", { value: rid }, rid.replace(/^mlops-/, "") + (i === 0 ? "  (latest)" : "")))))
    : null;

  // Built from every state actually seen across this DAG's history, not
  // a fixed universal list — a DAG that has never retried or skipped a
  // task doesn't get legend entries for states that can't appear.
  const seenStates = [...new Set(gridCells.map((c) => c.state).filter(Boolean))];
  const legendOrder = ["success", "running", "failed", "upstream_failed", "up_for_retry", "queued", "scheduled", "skipped", "removed"];
  seenStates.sort((a, b) => {
    const ia = legendOrder.indexOf(a), ib = legendOrder.indexOf(b);
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib) || a.localeCompare(b);
  });
  const legend = seenStates.length
    ? el("div", { class: "legend" },
        ...seenStates.map((s) => el("span", { class: `legend-item ${statusKind(s)}` },
          el("span", { class: "dot" }), s)))
    : null;

  const taskTable = el("table", {},
    el("thead", {}, el("tr", {},
      el("th", {}, "Task"), el("th", {}, "Operator"),
      el("th", {}, "Trigger rule"), el("th", {}, "Downstream"))),
    el("tbody", {}, ...(tasks.length ? tasks.map((t) =>
      el("tr", {},
        el("td", { class: "mono" }, t.task_id),
        el("td", { class: "muted" }, t.operator_name || "—"),
        el("td", { class: "muted" }, t.trigger_rule || "—"),
        el("td", {}, (t.downstream_task_ids || []).join(", ") || "—")))
      : [emptyRow(4, "No tasks — the DAG may have failed to parse.")])));

  const runsTable = el("table", {},
    el("thead", {}, el("tr", {},
      el("th", {}, "Run"), el("th", {}, "Status"), el("th", {}, "Type"),
      el("th", {}, "Training run"), el("th", {}, "Started"), el("th", {}, "Ended"))),
    el("tbody", {}, ...(dagRuns.length ? dagRuns.map((r) => {
      const fwId = byExecutionId.get(`${dagId}/${r.dag_run_id}`);
      return el("tr", {},
        el("td", { class: "mono" }, r.dag_run_id),
        el("td", {}, statusBadge(r.state)),
        el("td", { class: "muted" }, r.run_type || "—"),
        el("td", {}, fwId ? el("a", { href: `/runs/${fwId}` }, `#${fwId}`)
                          : el("span", { class: "faint" }, "scheduler")),
        el("td", { class: "muted nowrap" }, fmt.ago(r.start_date || r.execution_date)),
        el("td", { class: "muted nowrap" }, fmt.time(r.end_date)));
    }) : [emptyRow(6, "No runs yet.")])));

  mount(body,
    el("h3", {}, "Tasks"),
    el("div", { class: "card", style: "margin-bottom:16px" },
      (runPicker || legend)
        ? el("div", { class: "graph-toolbar" }, runPicker, legend)
        : null,
      levels ? el("div", { style: "margin-bottom:16px" }, graphHost) : null,
      el("div", { class: "table-wrap" }, taskTable)),
    el("h3", {}, "Task history"),
    el("p", { class: "muted", style: "margin:0 0 10px" },
      `Most recent ${gridRunIds.length || 0} run${gridRunIds.length === 1 ? "" : "s"}, newest first.`),
    el("div", { class: "card", style: "margin-bottom:16px" },
      renderTaskHistoryGrid(tasks, gridRunIds, gridCells, byExecutionId, dagId)),
    el("h3", {}, "Recent runs"),
    el("p", { class: "muted", style: "margin:0 0 10px" },
      "Includes runs the scheduler triggered on its own, not only ones started from this console."),
    el("div", { class: "table-wrap" }, runsTable));
}
