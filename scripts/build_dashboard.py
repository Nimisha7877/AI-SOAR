#!/usr/bin/env python
"""Build the single-file demo dashboard: docs/demo/dashboard.html.

Run:
    python scripts/build_dashboard.py
    then open docs/demo/dashboard.html in any browser (no server, no internet)

Two views inside ONE self-contained file (no external deps, works from file://):
- Dashboard view: KPIs, severity pie, verdict funnel, alerts-per-day chart,
  automation policy, confusion matrices. Charts always show the FULL dataset -
  they never mutate when you filter.
- Alerts view: the alert log with day / severity / family filters; clicking a
  pie slice or a day bar jumps here with that filter pre-applied. Row click
  expands the full response trail + explanation.

Deliberately NOT shown: internal evaluation tiers, score tables, dataset names.
Those live in artifacts/reports/ and the thesis, not on a public demo screen.
"""

from __future__ import annotations

import base64
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_soar.config import get_settings  # noqa: E402

OUT_REL = Path("docs") / "demo" / "dashboard.html"

# family, policy, mitre, evidence (no raw scores on the public demo)
POLICY_ROWS = [
    ("BruteForce", "auto_response", "T1110", "hardened-audit proven"),
    ("DDoS", "auto_response", "T1498", "hardened-audit proven"),
    ("DoS", "auto_response", "T1499", "hardened-audit proven"),
    ("PortScan", "auto_response", "T1046", "hardened-audit proven"),
    ("WebAttack", "human_approval", "T1190", "rare in training data"),
    ("Botnet", "human_approval", "T1071", "drift-prone in training data"),
    ("Infiltration", "human_approval", "T1566/T1046", "near-absent in training data"),
]

# "MachineLearningCVE-Friday-07-07-2017-6" -> "Friday-07-07-2017"
_DAY_RE = re.compile(r"(?:MachineLearningCVE-)?([A-Za-z]+day-\d{2}-\d{2}-\d{4})")


def extract_day(record: dict) -> str:
    """Traffic day (the day the captured flow happened), not the demo run day."""
    src = str(record.get("source_flow_id") or "")
    match = _DAY_RE.search(src)
    if match:
        return match.group(1)
    return (record.get("created_at") or "")[:10] or "unknown"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001 - never let one bad line kill the demo
                continue
    return out


def image_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def build_data(settings) -> dict:
    reports = Path(settings.paths.reports)
    incidents_dir = Path(settings.paths.incidents)

    incidents = load_jsonl(incidents_dir / "incidents.jsonl")
    latest: dict[str, dict] = {}
    for rec in incidents:
        latest[rec["incident_id"]] = rec
    alerts = sorted(latest.values(), key=lambda r: r.get("created_at", ""), reverse=True)
    for alert in alerts:
        alert["day"] = extract_day(alert)

    explanations = load_jsonl(incidents_dir / "explanations.jsonl")
    expl_map: dict[str, dict] = {}
    for rec in explanations:
        expl_map[rec["incident_id"]] = rec

    demo = (
        json.loads((reports / "response_demo_report.json").read_text(encoding="utf-8"))
        if (reports / "response_demo_report.json").exists()
        else {}
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "alerts": alerts,
        "explanations": expl_map,
        "demo": {
            "flows_per_second": demo.get("flows_per_second"),
            "latency_ms": demo.get("latency_ms", {}),
            "decisions": demo.get("decisions", {}),
        },
        "policy": POLICY_ROWS,
        "images": {
            "system_cm": image_uri(reports / "system_test_cm.png"),
            "hardened_cm": image_uri(reports / "hardened_system_cm.png"),
        },
    }


# --------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI SOAR — Demo Dashboard</title>
<style>
  :root{
    --bg:#0b1220; --panel:#121a2b; --panel2:#0f1626; --ink:#e6edf7; --mut:#8ea0b8;
    --crit:#ff5470; --high:#ff9f43; --med:#feca57; --low:#48dbfb;
    --ok:#2ecc71; --warn:#f1c40f; --line:#22304a;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:14px/1.5 "Segoe UI",system-ui,-apple-system,sans-serif}
  header{padding:18px 28px;border-bottom:1px solid var(--line);
         display:flex;flex-wrap:wrap;gap:14px;align-items:center}
  header h1{margin:0;font-size:21px;letter-spacing:.4px}
  header .sub{color:var(--mut);font-size:12px}
  nav{margin-left:auto;display:flex;gap:8px}
  .tab{background:var(--panel2);color:var(--mut);border:1px solid var(--line);
       border-radius:999px;padding:7px 16px;font-size:13px;cursor:pointer}
  .tab:hover{color:var(--ink)}
  .tab.active{background:#1d3a5f;color:#dff1ff;border-color:#2f5f96}
  .tab .badge{background:rgba(255,255,255,.12);border-radius:999px;padding:1px 7px;
              font-size:11px;margin-left:6px}
  main{padding:22px 28px;display:grid;gap:20px}
  .cards{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
  .card .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.8px}
  .card .v{font-size:26px;font-weight:650;margin-top:4px}
  .card .s{color:var(--mut);font-size:11px;margin-top:2px}
  .row{display:grid;gap:20px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
  .panel h2{margin:0 0 10px;font-size:14px;text-transform:uppercase;letter-spacing:.8px;color:var(--mut)}
  .legend{display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;font-size:12px;color:var(--mut)}
  .dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:5px}
  .sq{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:5px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
  th{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.6px;
     position:sticky;top:0;background:var(--panel);z-index:2}
  tr.alert{cursor:pointer}
  tr.alert:hover{background:#182338}
  tr.detail td{background:var(--panel2);padding:14px 18px}
  .chip{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600}
  .sev-critical{background:rgba(255,84,112,.16);color:var(--crit)}
  .sev-high{background:rgba(255,159,67,.16);color:var(--high)}
  .sev-medium{background:rgba(254,202,87,.16);color:var(--med)}
  .sev-low{background:rgba(72,219,251,.16);color:var(--low)}
  .st-auto_resolved{background:rgba(46,204,113,.15);color:var(--ok)}
  .st-pending_approval{background:rgba(241,196,64,.15);color:var(--warn)}
  .st-dismissed{background:rgba(142,160,184,.15);color:var(--mut)}
  .v-TP{color:var(--ok);font-weight:700}
  .v-FP{color:var(--crit);font-weight:700}
  .v-MIS{color:var(--high);font-weight:700}
  .filters{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:10px}
  .filters label{color:var(--mut);font-size:12px}
  select{background:var(--panel2);color:var(--ink);border:1px solid var(--line);
         border-radius:8px;padding:6px 10px;font-size:13px}
  .btn{background:var(--panel2);color:var(--mut);border:1px solid var(--line);
       border-radius:8px;padding:6px 12px;font-size:12px;cursor:pointer}
  .btn:hover{color:var(--ink)}
  .muted{color:var(--mut);font-size:12px}
  .expl{margin-top:10px;border-left:3px solid var(--low);padding:8px 12px;
        background:#101a2c;border-radius:0 8px 8px 0}
  .expl .cite{color:var(--mut);font-size:11px;margin-top:6px}
  .acts{margin:6px 0 0;padding-left:18px}
  .acts li{margin:3px 0}
  img.cm{max-width:100%;border-radius:10px;border:1px solid var(--line)}
  svg [data-sev], svg [data-day]{cursor:pointer}
  svg [data-sev]:hover, svg [data-day]:hover{filter:brightness(1.18)}
  footer{padding:18px 28px;color:var(--mut);font-size:12px;border-top:1px solid var(--line)}
  .hidden{display:none !important}
</style>
</head>
<body>
<header>
  <h1>AI SOAR</h1>
  <span class="sub">two-stage ML cascade · evidence-gated playbooks · human-in-the-loop</span>
  <nav>
    <button class="tab active" id="tabDash">Dashboard</button>
    <button class="tab" id="tabAlerts">Alerts<span class="badge" id="alertBadge">0</span></button>
  </nav>
</header>

<!-- ======================= DASHBOARD VIEW ======================= -->
<main id="viewDash">
  <section class="cards" id="kpis"></section>

  <section class="row">
    <div class="panel">
      <h2>Alerts by severity</h2>
      <svg id="pie" viewBox="0 0 200 200" width="100%" height="240"></svg>
      <div class="legend" id="pieLegend"></div>
      <div class="muted" style="margin-top:6px">Click a slice to open those alerts.</div>
    </div>
    <div class="panel">
      <h2>Verdict funnel</h2>
      <svg id="funnel" viewBox="0 0 320 250" width="100%" height="250"></svg>
      <div class="muted" id="funnelNote"></div>
    </div>
  </section>

  <section class="panel">
    <h2>Alerts by day</h2>
    <svg id="days" viewBox="0 0 900 260" width="100%" height="260"></svg>
    <div class="legend" id="dayLegend"></div>
    <div class="muted" style="margin-top:6px">Click a bar to open that day's alerts.</div>
  </section>

  <section class="row">
    <div class="panel">
      <h2>Automation policy</h2>
      <table id="policy"></table>
      <div class="muted" style="margin-top:8px">
        Destructive actions (host isolation, firewall block, DNS null-route) always wait for a
        human approver, even for auto-response families.
      </div>
    </div>
    <div class="panel">
      <h2>Confusion matrices</h2>
      <div id="cms" class="muted"></div>
    </div>
  </section>
</main>

<!-- ======================= ALERTS VIEW ======================= -->
<main id="viewAlerts" class="hidden">
  <section class="panel">
    <h2>Alert log</h2>
    <div class="filters">
      <label>Day</label><select id="fDay"></select>
      <label>Severity</label><select id="fSev"></select>
      <label>Family</label><select id="fFam"></select>
      <label>Status</label><select id="fStatus"></select>
      <button class="btn" id="fReset">reset</button>
      <span class="muted" id="fCount"></span>
      <button class="btn" id="backDash" style="margin-left:auto">← dashboard</button>
    </div>
    <div style="max-height:640px;overflow:auto">
      <table>
        <thead><tr>
          <th>Time</th><th>Traffic day</th><th>Incident</th><th>Family</th><th>Severity</th>
          <th>Decision</th><th>Status</th><th>Verdict</th>
        </tr></thead>
        <tbody id="alerts"></tbody>
      </table>
    </div>
    <div class="muted" style="margin-top:8px">Click any alert row to expand its response trail and explanation.</div>
  </section>
</main>

<footer>
  All responses shown here are <b>simulated</b>; destructive actions are approval-gated by policy.
  Incident and explanation logs live in <code>artifacts/incidents/</code>. Regenerate with
  <code>python scripts/build_dashboard.py</code>.
</footer>

<script id="data" type="application/json">@@DATA@@</script>
<script>
const DATA  = JSON.parse(document.getElementById('data').textContent);
const ALL   = DATA.alerts;                 // charts always use this - never filtered
const $     = (id) => document.getElementById(id);
const SEV   = ['critical','high','medium','low'];
const SEVC  = {critical:'#ff5470', high:'#ff9f43', medium:'#feca57', low:'#48dbfb'};
const state = {day:'all', sev:'all', fam:'all', status:'all'};

/* ======================= shared helpers ======================= */
function verdictOf(a){
  const t = a.ground_truth;
  if (t == null) return '—';
  if (t === a.family) return 'TP';
  if (t === 'BENIGN') return 'FP';
  return 'MIS';
}
function dayStamp(d){
  const m = String(d).match(/(\d{2})-(\d{2})-(\d{4})/);
  return m ? new Date(+m[3], +m[2]-1, +m[1]).getTime() : 0;
}
function dayShort(d){
  const parts = String(d).split('-');
  const m = String(d).match(/(\d{2})-(\d{2})/);
  return parts[0].slice(0,3) + (m ? ' ' + m[0] : '');
}

/* ======================= view switching ======================= */
function showView(name){
  const dash = name !== 'alerts';
  $('viewDash').classList.toggle('hidden', !dash);
  $('viewAlerts').classList.toggle('hidden', dash);
  $('tabDash').classList.toggle('active', dash);
  $('tabAlerts').classList.toggle('active', !dash);
  window.scrollTo({top:0, behavior:'instant'});
}
function gotoAlerts(params){
  Object.assign(state, params);
  syncSelects();
  const q = Object.entries(state).filter(([,v]) => v !== 'all')
                .map(([k,v]) => `${k}=${encodeURIComponent(v)}`).join('&');
  location.hash = 'alerts' + (q ? '?' + q : '');
  renderAlerts();
  showView('alerts');
}
function applyHash(){
  const h = location.hash.replace(/^#/, '');
  if (!h.startsWith('alerts')) { showView('dashboard'); return; }
  const q = h.split('?')[1] || '';
  state.day = state.sev = state.fam = state.status = 'all';
  new URLSearchParams(q).forEach((v, k) => { if (k in state) state[k] = v; });
  syncSelects(); renderAlerts(); showView('alerts');
}
$('tabDash').addEventListener('click', () => { location.hash = ''; });
$('tabAlerts').addEventListener('click', () => { location.hash = 'alerts'; });
$('backDash').addEventListener('click', () => { location.hash = ''; });
window.addEventListener('hashchange', applyHash);

/* ======================= DASHBOARD (static, full data) ======================= */
(function kpis(){
  const tp = ALL.filter(a => verdictOf(a)==='TP').length;
  const fp = ALL.filter(a => verdictOf(a)==='FP').length;
  const mis = ALL.filter(a => verdictOf(a)==='MIS').length;
  const pending = ALL.filter(a => a.status==='pending_approval').length;
  const d = DATA.demo || {};
  const lat = (d.latency_ms || {}).p95;
  const cards = [
    ['Total alerts', ALL.length, 'incidents in audit log'],
    ['True positives', tp, 'correct family'],
    ['False positives', fp, 'benign flagged'],
    ['Mis-family', mis, 'wrong family'],
    ['Pending approval', pending, 'human work queue'],
    ['Throughput', d.flows_per_second != null ? d.flows_per_second : '—', 'flows/s'],
    ['p95 latency', lat != null ? lat + ' ms' : '—', 'per flow'],
  ];
  $('kpis').innerHTML = cards.map(c =>
    `<div class="card"><div class="k">${c[0]}</div><div class="v">${c[1]}</div><div class="s">${c[2]}</div></div>`
  ).join('');
  $('alertBadge').textContent = ALL.length;
})();

(function pie(){
  const counts = SEV.map(s => ALL.filter(a => a.severity===s).length);
  const total = counts.reduce((x,y)=>x+y,0) || 1;
  const R = 70, C = 2*Math.PI*R;
  let off = 0;
  let svg = `<circle cx="100" cy="100" r="${R}" fill="none" stroke="#1b2740" stroke-width="34"/>`;
  SEV.forEach((s,i)=>{
    const frac = counts[i]/total;
    if (frac <= 0) return;
    svg += `<circle data-sev="${s}" cx="100" cy="100" r="${R}" fill="none" stroke="${SEVC[s]}" stroke-width="34"
            stroke-dasharray="${(frac*C).toFixed(2)} ${(C-frac*C).toFixed(2)}"
            stroke-dashoffset="${(-off*C).toFixed(2)}" transform="rotate(-90 100 100)"/>`;
    off += frac;
  });
  svg += `<text x="100" y="96" text-anchor="middle" fill="#e6edf7" font-size="26" font-weight="700">${ALL.length}</text>
          <text x="100" y="114" text-anchor="middle" fill="#8ea0b8" font-size="11">alerts</text>`;
  $('pie').innerHTML = svg;
  $('pieLegend').innerHTML = SEV.map((s,i)=>
    `<span><span class="dot" style="background:${SEVC[s]}"></span>${s}: ${counts[i]}</span>`).join('');
  $('pie').querySelectorAll('[data-sev]').forEach(el =>
    el.addEventListener('click', () => gotoAlerts({sev: el.dataset.sev})));
})();

(function funnel(){
  const tp = ALL.filter(a => verdictOf(a)==='TP').length;
  const fp = ALL.filter(a => verdictOf(a)==='FP').length;
  const mis = ALL.filter(a => verdictOf(a)==='MIS').length;
  const yTop = 15, yApex = 235, halfTop = 145, cx = 160;
  const hw = (y) => halfTop * (yApex - y) / (yApex - yTop);
  const y1 = 95, y2 = 165;
  const band = (ya, yb, fill, label, value, ly) => `
    <polygon points="${cx-hw(ya)},${ya} ${cx+hw(ya)},${ya} ${cx+hw(yb)},${yb} ${cx-hw(yb)},${yb}"
             fill="${fill}" fill-opacity="0.82" stroke="#0b1220" stroke-width="2"/>
    <text x="${cx}" y="${ly}" text-anchor="middle" fill="#0b1220" font-size="17" font-weight="800">${value}</text>
    <text x="${cx}" y="${ly+16}" text-anchor="middle" fill="#0b1220" font-size="10" font-weight="700">${label}</text>`;
  $('funnel').innerHTML =
    band(yTop, y1, '#ff5470', 'TOTAL ALERTS', ALL.length, 52) +
    band(y1, y2, '#48dbfb', 'TRUE POSITIVES', tp, 126) +
    band(y2, yApex, '#feca57', 'FALSE POSITIVES', fp, 192);
  $('funnelNote').textContent = mis > 0
    ? `Plus ${mis} mis-family verdict(s) - visible per alert in the Alerts tab.`
    : 'Every flagged alert matched its true family.';
})();

(function dayChart(){
  const W = 900, H = 260, base = H - 46, top = 26;
  const byDay = {};
  ALL.forEach(a => {
    const d = a.day || 'unknown';
    byDay[d] = byDay[d] || {critical:0, high:0, medium:0, low:0, total:0};
    if (byDay[d][a.severity] != null) byDay[d][a.severity]++;
    byDay[d].total++;
  });
  const list = Object.keys(byDay).sort((x,y) => dayStamp(x)-dayStamp(y));
  if (!list.length){ $('days').innerHTML = '<text x="20" y="40" fill="#8ea0b8">no data</text>'; return; }
  const maxTotal = Math.max(...list.map(d => byDay[d].total)) || 1;
  const slot = (W - 60) / list.length;
  const barW = Math.min(90, slot * 0.62);
  const stackOrder = ['low','medium','high','critical'];
  let svg = `<line x1="30" y1="${base}" x2="${W-20}" y2="${base}" stroke="#22304a" stroke-width="1"/>`;
  list.forEach((d, i) => {
    const cx = 40 + slot*i + slot/2;
    const x = cx - barW/2;
    let y = base;
    stackOrder.forEach(s => {
      const v = byDay[d][s];
      if (!v) return;
      const h = (v/maxTotal) * (base - top);
      y -= h;
      svg += `<rect data-day="${d}" x="${x.toFixed(1)}" y="${y.toFixed(1)}"
               width="${barW.toFixed(1)}" height="${Math.max(h,1).toFixed(1)}" fill="${SEVC[s]}" rx="2"/>`;
    });
    svg += `<text x="${cx.toFixed(1)}" y="${(y-8).toFixed(1)}" text-anchor="middle" fill="#e6edf7"
             font-size="13" font-weight="700">${byDay[d].total}</text>`;
    svg += `<text x="${cx.toFixed(1)}" y="${base+20}" text-anchor="middle" fill="#8ea0b8" font-size="11">${dayShort(d)}</text>`;
    svg += `<text x="${cx.toFixed(1)}" y="${base+34}" text-anchor="middle" fill="#5d6f8a" font-size="9">${d}</text>`;
  });
  $('days').innerHTML = svg;
  $('dayLegend').innerHTML = SEV.map(s =>
    `<span><span class="sq" style="background:${SEVC[s]}"></span>${s}</span>`).join('');
  $('days').querySelectorAll('[data-day]').forEach(el =>
    el.addEventListener('click', () => gotoAlerts({day: el.dataset.day})));
})();

$('policy').innerHTML = '<tr><th>Family</th><th>Policy</th><th>MITRE</th><th>Why</th></tr>' +
  DATA.policy.map(r => `<tr><td>${r[0]}</td>
    <td><span class="chip ${r[1]==='auto_response'?'st-auto_resolved':'st-pending_approval'}">${r[1]}</span></td>
    <td class="muted">${r[2]}</td><td class="muted">${r[3]}</td></tr>`).join('');

const cms = Object.entries(DATA.images||{}).filter(([,v]) => v);
$('cms').innerHTML = cms.length
  ? cms.map(([k,v]) => `<div style="margin-bottom:12px"><img class="cm" src="${v}" alt="${k}"></div>`).join('')
  : 'confusion-matrix PNGs not found in artifacts/reports/';

/* ======================= ALERTS VIEW (filtered) ======================= */
function fillSelect(sel, values, allLabel){
  sel.innerHTML = `<option value="all">${allLabel}</option>` +
    values.map(v => `<option value="${v}">${v}</option>`).join('');
}
fillSelect($('fDay'), [...new Set(ALL.map(a => a.day))].sort((x,y) => dayStamp(y)-dayStamp(x)), 'all days');
fillSelect($('fSev'), SEV, 'all severities');
fillSelect($('fFam'), [...new Set(ALL.map(a => a.family))].sort(), 'all families');
fillSelect($('fStatus'), [...new Set(ALL.map(a => a.status))].sort(), 'all statuses');

function syncSelects(){
  $('fDay').value = state.day; $('fSev').value = state.sev;
  $('fFam').value = state.fam; $('fStatus').value = state.status;
}
[['fDay','day'],['fSev','sev'],['fFam','fam'],['fStatus','status']].forEach(([id,key]) =>
  $(id).addEventListener('change', e => { state[key] = e.target.value; renderAlerts(); }));
$('fReset').addEventListener('click', () => {
  state.day = state.sev = state.fam = state.status = 'all';
  syncSelects(); renderAlerts();
});

function filtered(){
  return ALL.filter(a =>
    (state.day==='all'    || a.day===state.day) &&
    (state.sev==='all'    || a.severity===state.sev) &&
    (state.fam==='all'    || a.family===state.fam) &&
    (state.status==='all' || a.status===state.status));
}

function detailRow(a){
  const acts = (a.actions||[]).map(x =>
    `<li><b>${x.action}</b> — ${x.status}${x.approved_by?` (approved by ${x.approved_by})`:''}<br>
     <span class="muted">${x.message||''}</span></li>`).join('') ||
    '<li>playbook held in full pending human approval</li>';
  const notes = (a.notes||[]).map(n => `<li class="muted">${n}</li>`).join('');
  const ex = DATA.explanations[a.incident_id];
  const expl = ex ? `<div class="expl"><b>Explanation</b>
      <span class="chip ${ex.is_fallback?'st-pending_approval':'st-auto_resolved'}">${ex.provider}</span>
      <div style="margin-top:6px;white-space:pre-wrap">${ex.text}</div>
      <div class="cite">citations: ${ex.citations.join(' · ')}</div></div>` : '';
  return `<tr class="detail"><td colspan="8">
      <div><b>Decision:</b> ${a.decision} — <span class="muted">${a.decision_reason||''}</span></div>
      <div style="margin-top:4px"><b>Confidence:</b> ${a.confidence} · <b>gate p:</b> ${a.gate_probability}
        ${a.ground_truth?` · <b>ground truth:</b> ${a.ground_truth}`:''}</div>
      <ul class="acts">${acts}</ul>
      ${notes?`<ul class="acts">${notes}</ul>`:''}
      ${expl}
    </td></tr>`;
}

function renderAlerts(){
  const list = filtered();
  const tp = list.filter(a => verdictOf(a)==='TP').length;
  const fp = list.filter(a => verdictOf(a)==='FP').length;
  $('fCount').textContent = `${list.length} of ${ALL.length} alerts · TP ${tp} · FP ${fp}`;
  $('alerts').innerHTML = list.map(a => `
    <tr class="alert" data-id="${a.incident_id}">
      <td class="muted">${(a.created_at||'').replace('T',' ').slice(11,19)}</td>
      <td class="muted">${a.day||'—'}</td>
      <td><b>${a.incident_id}</b></td>
      <td>${a.family}</td>
      <td><span class="chip sev-${a.severity}">${a.severity}</span></td>
      <td>${a.decision}</td>
      <td><span class="chip st-${a.status}">${a.status}</span></td>
      <td class="v-${verdictOf(a)}">${verdictOf(a)}</td>
    </tr>`).join('') || '<tr><td colspan="8" class="muted">no alerts match the filters</td></tr>';
  document.querySelectorAll('tr.alert').forEach(tr => tr.addEventListener('click', () => {
    const id = tr.dataset.id;
    const existing = tr.nextElementSibling;
    if (existing && existing.classList.contains('detail')) { existing.remove(); return; }
    tr.insertAdjacentHTML('afterend', detailRow(ALL.find(a => a.incident_id===id)));
  }));
}

/* ======================= boot ======================= */
syncSelects();
renderAlerts();
applyHash();
</script>
</body>
</html>
"""


def main() -> int:
    settings = get_settings()
    data = build_data(settings)
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = HTML.replace("@@DATA@@", payload)

    out = ROOT / OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    days = sorted({a.get("day", "?") for a in data["alerts"]})
    print(f"dashboard written : {out}")
    print(f"alerts embedded   : {len(data['alerts'])}")
    print(f"traffic days      : {', '.join(days)}")
    print(f"explanations      : {len(data['explanations'])}")
    print(f"file size         : {out.stat().st_size / 1024:.0f} KB (single file, no external deps)")
    print("\nopen it with:  start docs\\demo\\dashboard.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())