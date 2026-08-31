"""The shared HTML/CSS/JS shell for both dashboard surfaces - one
`render_page(data, static)` used by dashboard.py (live local server) and
static_dashboard.py (GitHub Pages generator), so the two never drift into
different-looking pages for the same data.

PRICE CHARTS use TradingView's lightweight-charts (github.com/tradingview/
lightweight-charts, Apache 2.0) loaded from a CDN (unpkg) - the one
external, browser-side dependency in this whole project, chosen because
hand-rolling a real candlestick chart would either be worse than a proven,
purpose-built, 35KB library or a large, unjustified amount of new code.
Its license requires attribution; see the page footer. The Python backend
remains stdlib-only - nothing new was added to requirements.txt for this.

STATIC vs LIVE: `static=True` (GitHub Pages) removes the "Track a ticker"
network action (no backend exists there to run it) and replaces it with
instructions to edit watchlist.txt in the repo; `static=False` (the local
server) keeps the live POST /api/track control. Every other panel, chart,
and filter behaves identically in both modes - they render off the exact
same embedded JSON shape (dashboard_data.collect_dashboard_data).
"""
from __future__ import annotations

import json

LIGHTWEIGHT_CHARTS_CDN = "https://unpkg.com/lightweight-charts@5/dist/lightweight-charts.standalone.production.js"

STYLE = r"""
:root{
  --bg:#0A0D10; --panel:#10151A; --panel2:#141A21; --border:#1F2830; --border-soft:#182028;
  --text:#DDE6EC; --text-dim:#7C8A97; --text-faint:#4E5964;
  --accent:#38D9C4; --accent-dim:#1B4A45;
  --pos:#1FCF7A; --pos-bg:#0F2A1E; --neg:#F0495A; --neg-bg:#2E1218; --warn:#E0A93E; --warn-bg:#332608;
  --mono:"IBM Plex Mono","SF Mono",Consolas,monospace; --sans:-apple-system,"Segoe UI",sans-serif;
}
*{box-sizing:border-box;} body{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);font-size:13px;}
::selection{background:var(--accent-dim);color:var(--accent);}
.topbar{display:flex;align-items:center;gap:16px;padding:10px 20px;border-bottom:1px solid var(--border);background:var(--panel);position:sticky;top:0;z-index:10;flex-wrap:wrap;}
.topbar h1{font-size:15px;margin:0;font-weight:700;letter-spacing:.02em;}
.topbar h1 span{color:var(--accent);}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--pos);box-shadow:0 0 6px var(--pos);animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.35;}}
.topbar .sub{color:var(--text-dim);font-size:11.5px;font-family:var(--mono);}
.topbar .spacer{flex:1;}
.claim-warning{background:var(--warn-bg);color:var(--warn);border:1px solid #4A390F;padding:4px 10px;font-size:11px;font-weight:600;border-radius:3px;}
.layout{display:grid;grid-template-columns:250px 1fr;min-height:calc(100vh - 45px);}
@media (max-width:900px){.layout{grid-template-columns:1fr;}}
.sidebar{border-right:1px solid var(--border);background:var(--panel);padding:16px;overflow-y:auto;}
.sidebar h3{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--text-faint);margin:20px 0 8px;}
.sidebar h3:first-child{margin-top:0;}
.track-row{display:flex;gap:6px;}
.track-row input{flex:1;background:var(--bg);border:1px solid var(--border);color:var(--text);padding:7px 8px;font-family:var(--mono);font-size:12px;border-radius:3px;text-transform:uppercase;}
.track-row input:focus{outline:none;border-color:var(--accent);}
.btn{background:var(--accent-dim);color:var(--accent);border:1px solid var(--accent);padding:7px 12px;font-size:11.5px;font-weight:600;border-radius:3px;cursor:pointer;font-family:var(--sans);}
.btn:hover{background:var(--accent);color:#00201C;}
.btn:disabled{opacity:.4;cursor:wait;}
.btn.small{padding:4px 9px;font-size:10.5px;}
.track-status{font-size:11px;color:var(--text-dim);margin-top:6px;min-height:14px;}
.static-note{font-size:11px;color:var(--text-dim);line-height:1.5;background:var(--panel2);padding:8px 9px;border-radius:3px;}
.static-note code{color:var(--accent);}
.check-list{display:flex;flex-direction:column;gap:5px;}
.check-list label{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--text-dim);cursor:pointer;}
.check-list label:hover{color:var(--text);}
.check-list input{accent-color:var(--accent);}
.horizon-tabs{display:flex;flex-wrap:wrap;gap:5px;}
.htab{background:var(--panel2);border:1px solid var(--border);color:var(--text-dim);padding:5px 10px;font-size:11.5px;font-family:var(--mono);border-radius:3px;cursor:pointer;}
.htab.active{background:var(--accent-dim);border-color:var(--accent);color:var(--accent);}
.main{padding:18px 22px;overflow-x:hidden;}
.stat-row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px;}
.stat{background:var(--panel);border:1px solid var(--border);border-radius:4px;padding:10px 16px;min-width:110px;}
.stat .n{font-family:var(--mono);font-size:20px;font-weight:700;}
.stat .l{font-size:10px;color:var(--text-faint);text-transform:uppercase;letter-spacing:.05em;margin-top:2px;}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:5px;margin-bottom:18px;overflow:hidden;}
.panel-head{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;border-bottom:1px solid var(--border-soft);gap:10px;flex-wrap:wrap;}
.panel-head h2{font-size:12px;margin:0;text-transform:uppercase;letter-spacing:.05em;color:var(--text-dim);}
.panel-head select{background:var(--panel2);color:var(--text);border:1px solid var(--border);border-radius:3px;padding:4px 8px;font-family:var(--mono);font-size:11.5px;}
.panel-body{padding:14px;}
.charts-row{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
@media (max-width:1100px){.charts-row{grid-template-columns:1fr;}}
svg text{font-family:var(--mono);}
.empty-state{color:var(--text-faint);font-size:12px;padding:26px 10px;text-align:center;font-style:italic;}
table{border-collapse:collapse;width:100%;font-size:12px;}
thead th{position:sticky;top:0;background:var(--panel2);text-align:left;padding:7px 10px;font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:var(--text-faint);border-bottom:1px solid var(--border);cursor:pointer;white-space:nowrap;user-select:none;}
thead th:hover{color:var(--accent);}
thead th.sorted::after{content:" \25BE";color:var(--accent);}
tbody td{padding:7px 10px;border-bottom:1px solid var(--border-soft);font-family:var(--mono);vertical-align:top;}
tbody tr.row:hover{background:var(--panel2);cursor:pointer;}
.pos{color:var(--pos);font-weight:600;} .neg{color:var(--neg);font-weight:600;} .zero{color:var(--text-faint);}
.mode-chip{display:inline-block;padding:2px 7px;border-radius:3px;font-size:10px;font-weight:700;letter-spacing:.03em;}
.mode-chip.A_NO_NEWS{background:#1A2733;color:#6EA8D8;} .mode-chip.B_BLENDED{background:#231A33;color:#B48EDB;} .mode-chip.C_NEWS_ONLY{background:#132B24;color:#4CC9A8;}
.entity-pill{font-weight:700;color:var(--text);}
.detail-row td{background:var(--bg);border-bottom:1px solid var(--border);}
.detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:10px 4px;}
.detail-grid h4{font-size:10.5px;text-transform:uppercase;color:var(--text-faint);margin:0 0 8px;letter-spacing:.05em;}
.axis-bar-row{display:flex;align-items:center;gap:8px;font-size:11px;margin-bottom:4px;}
.axis-bar-row .name{width:130px;color:var(--text-dim);}
.axis-bar-track{flex:1;height:12px;background:var(--panel2);border-radius:2px;position:relative;}
.axis-bar-fill{position:absolute;top:0;bottom:0;border-radius:2px;}
.axis-bar-val{width:44px;text-align:right;font-family:var(--mono);}
.reasoning{font-size:11.5px;color:var(--text-dim);font-family:var(--sans);line-height:1.5;background:var(--panel2);padding:8px 10px;border-radius:3px;}
.metrics-table td, .metrics-table th{font-size:11.5px;}
#priceChart{width:100%;height:320px;}
footer.credits{padding:14px 22px 30px;color:var(--text-faint);font-size:10.5px;border-top:1px solid var(--border);}
footer.credits a{color:var(--text-dim);}
"""


def render_page(data: dict, static: bool) -> str:
    data_json = json.dumps(data)
    topbar_control = (
        '<span class="sub" id="genAt"></span>'
        if static else
        '<span class="live-dot"></span><span class="sub" id="dbpath"></span>'
    )
    refresh_control = "" if static else '<button class="btn small" onclick="refresh()">Refresh</button>'
    track_control = (
        f'<div class="static-note">Static build (GitHub Pages) — no live backend here. '
        f'To track a new ticker: add it to <code>watchlist.txt</code> in the repo (edit directly on '
        f'GitHub, or push a commit) and wait for the next scheduled run.</div>'
        if static else
        '<div class="track-row">'
        '<input id="tickerInput" placeholder="e.g. AMD" maxlength="10" onkeydown="if(event.key===\'Enter\')trackTicker()">'
        '<button class="btn" id="trackBtn" onclick="trackTicker()">Track</button></div>'
        '<div class="track-status" id="trackStatus"></div>'
    )
    track_js = "" if static else r"""
async function trackTicker(){
  const input = document.getElementById("tickerInput");
  const ticker = input.value.trim().toUpperCase();
  if(!ticker) return;
  const btn = document.getElementById("trackBtn");
  const status = document.getElementById("trackStatus");
  btn.disabled = true; status.textContent = `Running live research pass for ${ticker}... (real network calls, several seconds)`;
  try{
    const resp = await fetch("/api/track", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ticker})});
    const result = await resp.json();
    if(result.ok){ status.textContent = `${ticker} tracked - ${result.n_logged} predictions logged.`; await loadData(); input.value=""; }
    else{ status.textContent = `Failed: ${result.error}`; }
  } catch(e){ status.textContent = `Failed: ${e}`; }
  btn.disabled = false;
}
async function loadData(){
  const resp = await fetch("/api/data");
  window.__DATA__ = await resp.json();
  document.getElementById("dbpath").textContent = window.__DATA__.db_path;
  buildFilters(); renderAll();
}
function refresh(){ loadData(); }
"""
    init_js = (
        'document.getElementById("genAt").textContent = "Generated " + window.__DATA__.generated_at.slice(0,19) + " UTC";'
        if static else
        'document.getElementById("dbpath").textContent = window.__DATA__.db_path;'
    )
    lightweight_charts_script = f'<script src="{LIGHTWEIGHT_CHARTS_CDN}"></script>'

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Research Evaluation Terminal</title>
{lightweight_charts_script}
<style>{STYLE}</style></head><body>
<div class="topbar">
  <h1>RESEARCH <span>EVAL</span> TERMINAL</h1>
  {topbar_control}
  <div class="spacer"></div>
  <span class="claim-warning">NO PREDICTIVE-VALIDITY CLAIM MADE UNTIL OBSERVATIONS ACCUMULATE</span>
  {refresh_control}
</div>
<div class="layout">
  <div class="sidebar">
    <h3>Track a ticker</h3>
    {track_control}

    <h3>Entities</h3>
    <div class="check-list" id="entityFilters"></div>

    <h3>Modes</h3>
    <div class="check-list" id="modeFilters"></div>

    <h3>Horizon (for metrics/charts)</h3>
    <div class="horizon-tabs" id="horizonTabs"></div>
  </div>
  <div class="main">
    <div class="stat-row" id="statRow"></div>

    <div class="panel">
      <div class="panel-head">
        <h2>Spot price — <span id="priceEntityLabel"></span></h2>
        <select id="priceEntitySelect" onchange="onPriceEntityChange(this.value)"></select>
      </div>
      <div class="panel-body"><div id="priceChart"></div><div id="priceChartEmpty"></div></div>
    </div>

    <div class="panel">
      <div class="panel-head"><h2>Predicted Impact &amp; Confidence — current log</h2></div>
      <div class="panel-body"><div id="chartImpact"></div></div>
    </div>

    <div class="charts-row">
      <div class="panel">
        <div class="panel-head"><h2>News-state axes (click a row to select an entity)</h2></div>
        <div class="panel-body"><div id="chartAxes"></div></div>
      </div>
      <div class="panel">
        <div class="panel-head"><h2>Equity curve — <span id="eqHorizonLabel"></span></h2></div>
        <div class="panel-body"><div id="chartEquity"></div></div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-head"><h2>Mode comparison — <span id="metricsHorizonLabel"></span></h2></div>
      <div class="panel-body"><div id="metricsTable"></div></div>
    </div>

    <div class="panel">
      <div class="panel-head"><h2>Prediction log</h2></div>
      <div class="panel-body"><div id="logTable"></div></div>
    </div>
  </div>
</div>
<footer class="credits">Price charts rendered with <a href="https://github.com/tradingview/lightweight-charts" target="_blank" rel="noopener">TradingView Lightweight Charts</a> (Apache 2.0).
No investment advice; no predictive-validity claim is made until real prospective observations accumulate.</footer>

<script>
window.__DATA__ = {data_json};
let STATE = {{ entityFilter: new Set(), entitiesSeen: new Set(), modeFilter: new Set(window.__DATA__.modes),
              horizon: "5", sortCol: "triggered_at", sortDir: -1, selectedEntity: null, expanded: new Set(),
              priceEntity: null, priceChartHandle: null }};

function fmtNum(x, d){{ return x===null||x===undefined ? "—" : Number(x).toFixed(d===undefined?4:d); }}
function fmtPct(x){{ return x===null||x===undefined ? "—" : (x*100).toFixed(1)+"%"; }}
function fmtSigned(x, d){{ if(x===null||x===undefined) return "—"; const v=Number(x); return (v>=0?"+":"")+v.toFixed(d===undefined?4:d); }}
function signClass(x){{ if(x===null||x===undefined||x===0) return "zero"; return x>0 ? "pos":"neg"; }}
function esc(s){{ return (s===null||s===undefined) ? "" : String(s).replace(/[&<>"]/g, c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}}[c])); }}

function allEntities(){{ return [...new Set(window.__DATA__.predictions.map(p=>p.entity))].sort(); }}

function buildFilters(){{
  const entities = allEntities();
  entities.forEach(e => {{ if(!STATE.entitiesSeen.has(e)){{ STATE.entityFilter.add(e); STATE.entitiesSeen.add(e); }} }});
  document.getElementById("entityFilters").innerHTML = entities.map(e =>
    `<label><input type="checkbox" ${{STATE.entityFilter.has(e)?"checked":""}} onchange="toggleEntity('${{e}}')">${{e}}</label>`
  ).join("");
  document.getElementById("modeFilters").innerHTML = window.__DATA__.modes.map(m =>
    `<label><input type="checkbox" ${{STATE.modeFilter.has(m)?"checked":""}} onchange="toggleMode('${{m}}')"><span class="mode-chip ${{m}}">${{m}}</span></label>`
  ).join("");
  document.getElementById("horizonTabs").innerHTML = ["1","5","20","60"].map(h =>
    `<div class="htab ${{STATE.horizon===h?"active":""}}" onclick="setHorizon('${{h}}')">${{h}}d</div>`
  ).join("");
  const priceEntities = Object.keys(window.__DATA__.price_series||{{}}).sort();
  if(!STATE.priceEntity || !priceEntities.includes(STATE.priceEntity)) STATE.priceEntity = priceEntities[0]||null;
  document.getElementById("priceEntitySelect").innerHTML = priceEntities.map(e =>
    `<option value="${{e}}" ${{e===STATE.priceEntity?"selected":""}}>${{e}}</option>`).join("");
}}
function toggleEntity(e){{ STATE.entityFilter.has(e) ? STATE.entityFilter.delete(e) : STATE.entityFilter.add(e); renderAll(); }}
function toggleMode(m){{ STATE.modeFilter.has(m) ? STATE.modeFilter.delete(m) : STATE.modeFilter.add(m); renderAll(); }}
function setHorizon(h){{ STATE.horizon = h; buildFilters(); renderAll(); }}
function onPriceEntityChange(e){{ STATE.priceEntity = e; renderPriceChart(); }}

function filteredPredictions(){{
  return window.__DATA__.predictions.filter(p => STATE.entityFilter.has(p.entity) && STATE.modeFilter.has(p.mode));
}}

function renderStats(){{
  const preds = filteredPredictions();
  const resolvedAny = preds.filter(p => p.realized_return_1d!==null || p.realized_return_5d!==null ||
                                         p.realized_return_20d!==null || p.realized_return_60d!==null).length;
  const stats = [
    [preds.length, "Predictions"], [resolvedAny, "Resolved (any horizon)"],
    [new Set(preds.map(p=>p.entity)).size, "Entities shown"], [window.__DATA__.predictions.length, "Total in log"],
  ];
  document.getElementById("statRow").innerHTML = stats.map(([n,l]) =>
    `<div class="stat"><div class="n">${{n}}</div><div class="l">${{l}}</div></div>`).join("");
}}

function renderPriceChart(){{
  document.getElementById("priceEntityLabel").textContent = STATE.priceEntity || "(none tracked)";
  const container = document.getElementById("priceChart");
  const emptyEl = document.getElementById("priceChartEmpty");
  container.innerHTML = ""; emptyEl.innerHTML = "";
  const series = STATE.priceEntity ? (window.__DATA__.price_series||{{}})[STATE.priceEntity] : null;
  if(!series || series.length===0 || typeof LightweightCharts==="undefined"){{
    emptyEl.innerHTML = '<div class="empty-state">' +
      (typeof LightweightCharts==="undefined" ? "Chart library did not load (offline?)." : "No price history available for this entity.") +
      '</div>';
    return;
  }}
  const chart = LightweightCharts.createChart(container, {{
    layout: {{ background: {{color:"#10151A"}}, textColor:"#7C8A97" }},
    grid: {{ vertLines:{{color:"#1F2830"}}, horzLines:{{color:"#1F2830"}} }},
    timeScale: {{ borderColor:"#1F2830" }}, rightPriceScale: {{ borderColor:"#1F2830" }},
    autoSize: true,
  }});
  const candles = chart.addSeries(LightweightCharts.CandlestickSeries, {{
    upColor:"#1FCF7A", downColor:"#F0495A", borderVisible:false, wickUpColor:"#1FCF7A", wickDownColor:"#F0495A",
  }});
  candles.setData(series);
  chart.timeScale().fitContent();
}}

// --- SVG chart helpers (no library - plain SVG strings, used for everything except the price chart) ---
function svgBarChart(groups, opts){{
  const W = opts.width||760, H = opts.height||220, padL=44, padB=28, padT=10, padR=10;
  const plotW = W-padL-padR, plotH = H-padT-padB;
  const allVals = groups.flatMap(g=>g.bars.map(b=>b.value));
  const maxAbs = Math.max(1e-6, ...allVals.map(v=>Math.abs(v)));
  const zeroY = padT + plotH/2;
  const groupW = plotW/Math.max(1,groups.length);
  let bars = "", labels = "";
  groups.forEach((g,gi)=>{{
    const gx = padL + gi*groupW;
    const barW = Math.min(18, (groupW-10)/Math.max(1,g.bars.length));
    g.bars.forEach((b,bi)=>{{
      const bx = gx + 5 + bi*barW;
      const h = (Math.abs(b.value)/maxAbs) * (plotH/2 - 4);
      const y = b.value>=0 ? zeroY-h : zeroY;
      bars += `<rect x="${{bx.toFixed(1)}}" y="${{y.toFixed(1)}}" width="${{(barW-2).toFixed(1)}}" height="${{Math.max(h,0.5).toFixed(1)}}" fill="${{b.value>=0?"#1FCF7A":"#F0495A"}}" opacity="0.9"><title>${{esc(g.label)}} ${{esc(b.name)}}: ${{b.value.toFixed(3)}}</title></rect>`;
    }});
    labels += `<text x="${{(gx+groupW/2).toFixed(1)}}" y="${{H-8}}" font-size="10" fill="#7C8A97" text-anchor="middle">${{esc(g.label)}}</text>`;
  }});
  return `<svg viewBox="0 0 ${{W}} ${{H}}" width="100%" height="${{H}}">
    <line x1="${{padL}}" y1="${{zeroY}}" x2="${{W-padR}}" y2="${{zeroY}}" stroke="#1F2830"/>
    <text x="${{padL-6}}" y="${{zeroY+3}}" font-size="9" fill="#4E5964" text-anchor="end">0</text>
    ${{bars}}${{labels}}
  </svg>`;
}}

function svgHBarChart(items, opts){{
  const W = opts.width||700, rowH = 22, padL=140, padR=50, padT=6;
  const H = padT*2 + items.length*rowH;
  const maxAbs = Math.max(1e-6, ...items.map(i=>Math.abs(i.value)));
  const midX = padL + (W-padL-padR)/2;
  const halfW = (W-padL-padR)/2;
  let rows = "";
  items.forEach((it,i)=>{{
    const y = padT + i*rowH;
    const w = (Math.abs(it.value)/maxAbs) * halfW;
    const x = it.value>=0 ? midX : midX-w;
    rows += `<text x="${{padL-8}}" y="${{y+rowH/2+4}}" font-size="10.5" fill="#AEBAB3" text-anchor="end">${{esc(it.name)}}</text>
      <rect x="${{x.toFixed(1)}}" y="${{y+4}}" width="${{Math.max(w,0.5).toFixed(1)}}" height="${{rowH-8}}" fill="${{it.value>=0?"#1FCF7A":"#F0495A"}}" opacity="0.85"/>
      <text x="${{midX + (it.value>=0? w+6 : -w-6)}}" y="${{y+rowH/2+4}}" font-size="10" fill="#7C8A97" text-anchor="${{it.value>=0?"start":"end"}}">${{it.value.toFixed(2)}}</text>`;
  }});
  return `<svg viewBox="0 0 ${{W}} ${{H}}" width="100%" height="${{H}}">
    <line x1="${{midX}}" y1="0" x2="${{midX}}" y2="${{H}}" stroke="#1F2830"/>${{rows}}
  </svg>`;
}}

function svgLineChart(series, opts){{
  const W = opts.width||760, H = opts.height||220, padL=44, padB=24, padT=10, padR=14;
  if(series.length===0) return null;
  const plotW=W-padL-padR, plotH=H-padT-padB;
  const ys = series.map(p=>p.y);
  const minY = Math.min(0,...ys), maxY = Math.max(0,...ys);
  const range = (maxY-minY)||1;
  const x = i => padL + (series.length<=1?0:(i/(series.length-1))*plotW);
  const y = v => padT + plotH - ((v-minY)/range)*plotH;
  const zeroY = y(0);
  let path = series.map((p,i)=>`${{i===0?"M":"L"}}${{x(i).toFixed(1)}},${{y(p.y).toFixed(1)}}`).join(" ");
  const last = series[series.length-1];
  return `<svg viewBox="0 0 ${{W}} ${{H}}" width="100%" height="${{H}}">
    <line x1="${{padL}}" y1="${{zeroY.toFixed(1)}}" x2="${{W-padR}}" y2="${{zeroY.toFixed(1)}}" stroke="#1F2830"/>
    <path d="${{path}}" fill="none" stroke="#38D9C4" stroke-width="2"/>
    <circle cx="${{x(series.length-1).toFixed(1)}}" cy="${{y(last.y).toFixed(1)}}" r="3.5" fill="${{last.y>=0?"#1FCF7A":"#F0495A"}}"/>
    <text x="${{W-padR}}" y="${{y(last.y)-8}}" font-size="10" fill="#DDE6EC" text-anchor="end">${{last.y.toFixed(3)}}</text>
  </svg>`;
}}

function renderImpactChart(){{
  const preds = filteredPredictions();
  const entities = [...new Set(preds.map(p=>p.entity))].sort();
  const groups = entities.map(e => ({{ label:e, bars: window.__DATA__.modes
    .filter(m=>STATE.modeFilter.has(m))
    .map(m => {{ const p = preds.find(pp=>pp.entity===e && pp.mode===m);
                return {{name:m, value: p && p.predicted_impact!==null ? p.predicted_impact : 0}}; }}) }}));
  document.getElementById("chartImpact").innerHTML = groups.length ?
    svgBarChart(groups, {{}}) + `<div style="font-size:10.5px;color:var(--text-faint);margin-top:4px">Bars: ${{[...STATE.modeFilter].join(" · ")}} (left→right per ticker) · green = positive predicted_impact, red = negative</div>`
    : '<div class="empty-state">No predictions match the current filters.</div>';
}}

function renderAxesChart(){{
  const preds = filteredPredictions();
  let entity = STATE.selectedEntity && preds.some(p=>p.entity===STATE.selectedEntity) ? STATE.selectedEntity : (preds[0] && preds[0].entity);
  const withNews = preds.find(p => p.entity===entity && p.news_state && p.news_state.dimensions);
  if(!withNews){{ document.getElementById("chartAxes").innerHTML = '<div class="empty-state">No news_state axes available for the selected/filtered entities.</div>'; return; }}
  const items = window.__DATA__.axes.map(a => ({{name:a, value: withNews.news_state.dimensions[a]}}))
    .filter(i => i.value !== null && i.value !== undefined);
  document.getElementById("chartAxes").innerHTML = items.length ?
    `<div style="font-size:11px;color:var(--text-dim);margin-bottom:6px">Entity: <b style="color:var(--accent)">${{esc(entity)}}</b></div>` + svgHBarChart(items, {{}})
    : `<div class="empty-state">${{esc(entity||"")}} carried no signal on any axis.</div>`;
}}

function renderEquityChart(){{
  document.getElementById("eqHorizonLabel").textContent = STATE.horizon + "d, mode B_BLENDED";
  const preds = filteredPredictions().filter(p=>p.mode==="B_BLENDED" && p.predicted_impact!==null);
  const col = "realized_return_"+STATE.horizon+"d";
  const resolved = preds.filter(p=>p[col]!==null).sort((a,b)=>a.triggered_at.localeCompare(b.triggered_at));
  if(resolved.length===0){{
    document.getElementById("chartEquity").innerHTML = '<div class="empty-state">No resolved trades yet at this horizon — chart will populate once outcome_resolution.py fills in real outcomes. Not a placeholder: this is the actual query, currently returning zero rows.</div>';
    return;
  }}
  let cum=0; const series = resolved.map(p=>{{ const dir = p.predicted_impact>0?1:(p.predicted_impact<0?-1:0); cum += dir*p[col]; return {{y:cum}}; }});
  document.getElementById("chartEquity").innerHTML = svgLineChart(series, {{}});
}}

function renderMetricsTable(){{
  document.getElementById("metricsHorizonLabel").textContent = STATE.horizon + "d";
  const byMode = window.__DATA__.metrics[STATE.horizon];
  const cols = [["n","n"],["direction_accuracy","Hit rate",true],["brier_score","Brier"],
                ["sharpe_per_trade","Sharpe"],["sortino_per_trade","Sortino"],
                ["max_drawdown","Max DD",true],["turnover_trades_per_year","Turnover/yr"]];
  let rows = "";
  window.__DATA__.modes.forEach(m=>{{
    const d = byMode[m];
    rows += `<tr><td><span class="mode-chip ${{m}}">${{m}}</span></td>` + cols.map(([k,_,pct])=>{{
      const v = d[k];
      const disp = pct ? fmtPct(v) : fmtNum(v,3);
      return `<td class="${{k==='n'?'':signClass(v)}}">${{v===null||v===undefined?"—":disp}}</td>`;
    }}).join("") + "</tr>";
  }});
  const anyData = window.__DATA__.modes.some(m=>byMode[m].n>0);
  document.getElementById("metricsTable").innerHTML =
    (anyData?"":'<div class="empty-state" style="margin-bottom:8px">No resolved observations at this horizon for any mode — every value below is honestly "—", not a claim.</div>') +
    `<table class="metrics-table"><thead><tr><th>Mode</th>${{cols.map(c=>`<th>${{c[1]}}</th>`).join("")}}</tr></thead><tbody>${{rows}}</tbody></table>`;
}}

function sortIcon(col){{ return STATE.sortCol===col ? "sorted" : ""; }}
function setSort(col){{ if(STATE.sortCol===col) STATE.sortDir*=-1; else {{STATE.sortCol=col; STATE.sortDir=-1;}} renderLogTable(); }}

function renderLogTable(){{
  let preds = filteredPredictions().slice();
  preds.sort((a,b)=>{{
    let av=a[STATE.sortCol], bv=b[STATE.sortCol];
    if(av===null||av===undefined) av = -Infinity; if(bv===null||bv===undefined) bv = -Infinity;
    if(typeof av==="string") return STATE.sortDir*String(av).localeCompare(String(bv));
    return STATE.sortDir*((av>bv)?1:(av<bv?-1:0));
  }});
  const cols = [["entity","Entity"],["mode","Mode"],["decision_label","Decision"],["predicted_impact","Impact"],
                ["predicted_confidence","Conf."],["realized_return_1d","1d"],["realized_return_5d","5d"],
                ["realized_return_20d","20d"],["realized_return_60d","60d"],["triggered_at","Triggered"]];
  let head = cols.map(([k,l])=>`<th class="${{sortIcon(k)}}" onclick="setSort('${{k}}')">${{l}}</th>`).join("");
  let body = "";
  if(preds.length===0) body = `<tr><td colspan="${{cols.length}}"><div class="empty-state">No predictions match the current filters.</div></td></tr>`;
  preds.forEach(p=>{{
    const expanded = STATE.expanded.has(p.id);
    body += `<tr class="row" onclick="toggleExpand('${{p.id}}')">
      <td class="entity-pill">${{esc(p.entity)}}</td>
      <td><span class="mode-chip ${{p.mode}}">${{p.mode}}</span></td>
      <td>${{esc(p.decision_label)}}</td>
      <td class="${{signClass(p.predicted_impact)}}">${{fmtSigned(p.predicted_impact,2)}}</td>
      <td>${{fmtPct(p.predicted_confidence)}}</td>
      <td class="${{signClass(p.realized_return_1d)}}">${{fmtSigned(p.realized_return_1d,3)}}</td>
      <td class="${{signClass(p.realized_return_5d)}}">${{fmtSigned(p.realized_return_5d,3)}}</td>
      <td class="${{signClass(p.realized_return_20d)}}">${{fmtSigned(p.realized_return_20d,3)}}</td>
      <td class="${{signClass(p.realized_return_60d)}}">${{fmtSigned(p.realized_return_60d,3)}}</td>
      <td style="color:var(--text-faint)">${{esc(p.triggered_at.slice(0,19))}}</td>
    </tr>`;
    if(expanded){{
      const dims = (p.news_state && p.news_state.dimensions) || {{}};
      const dimItems = window.__DATA__.axes.filter(a=>dims[a]!==null && dims[a]!==undefined);
      body += `<tr class="detail-row"><td colspan="${{cols.length}}"><div class="detail-grid">
        <div><h4>Mode reasoning</h4><div class="reasoning">${{esc(p.mode_reasoning||"—")}}</div>
          <h4 style="margin-top:12px">Context</h4>
          <div style="font-size:11.5px;color:var(--text-dim)">assessment_confidence=${{fmtPct(p.assessment_confidence)}} ·
          narratives=${{p.narrative_count??"—"}} · risks=${{p.risk_count??"—"}} · llm_status=${{esc(p.llm_status||"—")}} ·
          contradiction_axes=${{esc((p.news_state&&p.news_state.contradiction_axes||[]).join(", ")||"none")}}</div></div>
        <div><h4>News-state axes</h4>${{dimItems.length ? dimItems.map(a=>{{
            const v=dims[a]; const pctW=Math.min(100,Math.abs(v)*100);
            return `<div class="axis-bar-row"><div class="name">${{a}}</div><div class="axis-bar-track"><div class="axis-bar-fill" style="${{v>=0?'left:50%':'right:50%'}};width:${{pctW/2}}%;background:${{v>=0?'#1FCF7A':'#F0495A'}}"></div></div><div class="axis-bar-val">${{v.toFixed(2)}}</div></div>`;
          }}).join("") : '<div class="empty-state">No axes carried a signal.</div>'}}</div>
      </div></td></tr>`;
    }}
  }});
  document.getElementById("logTable").innerHTML = `<table><thead><tr>${{head}}</tr></thead><tbody>${{body}}</tbody></table>`;
}}
function toggleExpand(id){{ STATE.expanded.has(id) ? STATE.expanded.delete(id) : STATE.expanded.add(id); renderLogTable(); }}

function renderAll(){{
  renderStats(); renderPriceChart(); renderImpactChart(); renderAxesChart(); renderEquityChart(); renderMetricsTable(); renderLogTable();
}}
{track_js}
{init_js}
buildFilters(); renderAll();
</script>
</body></html>"""
