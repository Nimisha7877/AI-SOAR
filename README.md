# AI SOAR

**ML-driven Security Orchestration, Automation & Response.**
A SOAR platform that replaces signature-based detection and first-line triage with a calibrated
two-stage classifier, keeps response declarative and auditable, and explains every action with an
RAG-grounded LLM.

> **Scope note:** research/demo build on CICIDS2017 network flow data. Detection, response
> orchestration and explanation are real and measured; **actuators are simulated** (no live
> firewall/EDR/DNS is touched) and n8n + live pcap ingestion are future work.

---

## What it does

1. **Detects** malicious flows with a binary gate, then **classifies** the attack family
   (7 families) with a confidence score.
2. **Decides** the response from a policy catalog: `auto_response`, `human_approval` or
   `unknown_queue` — the automation scope per family is derived from *measured* reliability,
   not from guesswork.
3. **Executes** playbooks through simulated actuators (isolate host, block source, null-route DNS,
   disable account…) behind **two human approval gates** for destructive actions.
4. **Explains** each incident: RAG retrieval over a security knowledge base + local LLM, with
   citations, MITRE ATT&CK mapping, and a deterministic template fallback if no LLM is reachable.
5. **Audits** everything to an append-only, event-sourced log, and renders it as a
   **single-file HTML dashboard**.

---

## Traditional SOAR vs AI SOAR

| Aspect | Traditional SOAR | **AI SOAR** |
|---|---|---|
| Detection | signatures, IOC rules, static thresholds | learned two-stage classifier over 80 flow features |
| Triage | analyst reads the alert queue | automatic family + severity + MITRE mapping |
| Prioritisation | fixed severity mapping | calibrated confidence + evidence-based policy |
| Response | same playbook steps, always | policy catalog; automation granted only where evaluation proved reliability |
| Explanation | analyst's free-text notes | RAG-grounded LLM narrative with citations + fallback |
| Unknown threats | rule gap → silently missed | confidence gate → `unknown_queue` → explicit human review |
| New threat type | write a new rule/playbook | retrain + update policy (minutes, not days) |
| Audit trail | ticket comments | append-only event-sourced JSONL (actor, approver, timestamp) |
| Alert volume | analyst is the bottleneck → alert fatigue | auto-triage; only ambiguous / destructive cases reach a human |
| Mean time to triage | minutes to hours per alert | **4.26 ms** p95 per flow (2,000-flow replay @ 362 flows/s) |
| False positives | static thresholds, hand-tuned per environment | calibrated probabilities + a documented threshold sweep |
| Consistency | depends on the analyst on shift | same input → same decision, same written justification |
| Knowledge retention | playbooks + tribal knowledge | versioned knowledge base; explanations cite it |
| Compliance evidence | assembled manually from tickets | audit log already holds who decided what, when, and why |
| Skill required | rule/playbook engineers | data + retraining discipline |
| Brand-new IOC | a rule can be written in minutes | needs retraining (rules still win here) |

**Where it is better:** it triages at machine speed (362 flows/s, p95 4.3 ms/flow in the demo),
it does not need a human to write a rule for every variant, and it *quantifies its own reliability*
per attack family — so automation is granted on evidence, and weakly-measured families are routed
to a human instead of being automated blindly. Every action arrives with a written justification an
analyst can challenge.

**Where it is not better (honest):** rules are more interpretable and changeable instantly; a
classifier needs labelled data, drifts when traffic changes, and is closed-set — it cannot name an
attack it has never seen. That is exactly why this build keeps human approval gates and an
`unknown_queue` rather than claiming full autonomy.

**Net effect, in one line each:**

1. **Speed** — triage drops from "as fast as an analyst can read" to milliseconds per flow.
2. **Evidence-based automation** — automation is granted per family only where evaluation proved
   reliability (DDoS 0.999 → auto; Botnet 0.000 → always human), instead of an all-or-nothing switch.
3. **Explainability by default** — every incident ships with a cited, MITRE-mapped justification,
   so an analyst can challenge the machine's reasoning instead of guessing it.
4. **Auditability** — approvals, dismissals and actuators are event-sourced, which is what a
   compliance reviewer actually asks for.
5. **Measured honesty** — the project publishes its own inflation (0.9986 upper bound vs 0.6836
   hardened), which is the difference between a demo and a defensible result.

---

## Architecture

```
        ┌───────────────────────────────────────────────────────────────┐
        │  INGESTION   pcap / CSV flows  →  clean, dedup, label-collapse │
        └───────────────────────────────┬───────────────────────────────┘
                                        │  80 → 70 usable features
                                        ▼
        ┌───────────────────────────────────────────────────────────────┐
        │  STAGE 1 — binary gate (LightGBM)     benign / malicious       │
        └───────────────┬───────────────────────────────────┬───────────┘
                  benign│                                   │malicious
                        ▼                                   ▼
                     DROP / log            ┌────────────────────────────┐
                                           │ STAGE 2 — family classifier│
                                           │ BruteForce, DDoS, DoS,     │
                                           │ PortScan, WebAttack,       │
                                           │ Botnet, Infiltration       │
                                           └───────────────┬────────────┘
                                                           │ family + confidence
                        confidence < floor ────────────────┤
                                ▼                          ▼
                      ┌──────────────────┐   ┌───────────────────────────────┐
                      │  UNKNOWN QUEUE   │   │  POLICY CATALOG (per family)  │
                      │  → human review  │   │  auto_response | human_approval│
                      └──────────────────┘   └───────────────┬───────────────┘
                                                             ▼
                             ┌───────────────────────────────────────────────┐
                             │  RESPONSE ENGINE — playbooks + 2 approval gates│
                             │  simulated actuators, event-sourced audit log  │
                             └───────────────┬───────────────────────────────┘
                                             ▼
                    ┌────────────────────────────────────────────────────────┐
                    │  EXPLAIN — RAG (TF-IDF over knowledge base) + LLM       │
                    │  cited justification, MITRE mapping, template fallback  │
                    └───────────────┬────────────────────────────────────────┘
                                    ▼
              ┌──────────────────────────────────────────────┐
              │  SURFACE — FastAPI (/health /predict /batch) │
              │  single-file HTML dashboard  ·  n8n (roadmap)│
              └──────────────────────────────────────────────┘
```

| Layer | Module | Responsibility |
|---|---|---|
| Ingestion | `src/ai_soar/data/` | schema, chunked loading, cleaning, label collapse, splits |
| Models | `src/ai_soar/models/` | binary gate + family classifier (LightGBM), registry/metadata |
| Evaluation | `src/ai_soar/evaluation/` | macro/per-class F1, confusion matrices, leakage audit |
| Inference | `src/ai_soar/inference/` | FastAPI service, cascade, confidence gate, decision policy |
| Response | `src/ai_soar/response/` | policy catalog, playbooks, actuators, approval gates, audit store |
| Explain | `src/ai_soar/explain/` | RAG store/retrieval, LLM client (4 providers), explainer + fallback |
| Config | `src/ai_soar/config.py` | typed settings; precedence **`.env` > `settings.yaml` > code defaults** |

---

## Features

- **Two-stage cascade** — cheap binary gate first, expensive 7-way classifier only on flagged flows.
- **Evidence-gated automation** — each family's automation level comes from its burst-hardened F1.
- **Two human approval gates** for destructive actions; every approval is event-sourced with actor id.
- **Append-only audit trail** (JSONL) — incidents, actions, approvals, dismissals, notes.
- **Grounded LLM explanations** — retrieval-augmented, citations per explanation, MITRE ids;
  deterministic template fallback so explanation never blocks or crashes the pipeline.
- **Offline-first LLM** — Ollama local model; OpenAI/Anthropic/Gemini also supported via `.env`.
- **Single-file dashboard** — no server, no CDN, no build step; opens from `file://`.
- **Strict input contract** — API rejects feature drift with HTTP 422 instead of guessing.
- **Reproducible** — pinned dependencies, fixed seeds, `bootstrap.ps1` / `Dockerfile`.

---

## Tech stack

| Area | Used |
|---|---|
| Language / runtime | Python 3.12 |
| ML | LightGBM 4.5, scikit-learn 1.5 |
| Data | pandas 2.2, pyarrow 17 (parquet), numpy 1.26 |
| API | FastAPI 0.115, Uvicorn 0.30, pydantic 2.8 |
| LLM / RAG | Ollama (qwen2.5-coder:3b), TF-IDF retrieval (scikit-learn), httpx |
| Reporting | matplotlib 3.9, seaborn 0.13 |
| Config / secrets | PyYAML + python-dotenv, typed pydantic settings |
| Packaging | `bootstrap.ps1` (Windows), Docker (python:3.12-slim + libgomp1) |
| Dataset | CICIDS2017 (2.8M flows, 80 features, 15 labels); CSE-CIC-IDS2018 for cross-dataset work |

---

## Results & metrics

**Data pipeline:** 2,830,743 raw rows → 2,616,379 clean rows (4,376 `Inf` cells fixed, 2,867 NaN
rows dropped, **211,497 duplicate flows removed**), 15 raw labels collapsed to 8 families.

**Evaluation — read in this order.** CICIDS2017 contains near-duplicate flows, so a random split
inflates every score. Each tier answers a different question:

| Tier | Question | Result |
|---|---|---|
| A. Stratified test | upper bound (duplicates present) | stage-1 AUC **1.000** / F1 0.9974; stage-2 macro-F1 **0.9986**; cascaded 0.9554 |
| B. Burst-hardened split | honest discrimination | macro-F1 **0.6836** → measured **inflation 0.315** |
| C. Temporal Mon–Thu → Friday | unseen day | gate AUC 0.8527, recall 0.3457 @ 0.5; system macro-F1 0.2091 |
| D. Cross-dataset (CSE-CIC-IDS2018) | unseen environment | in progress — 5.45 GB downloaded & verified |

**Per-family hardened F1 → the automation policy it produced:**

| Family | Hardened F1 | Policy | MITRE |
|---|---|---|---|
| BruteForce | 0.999 | `auto_response` | T1110 |
| DDoS | 0.999 | `auto_response` | T1498 |
| DoS | 0.985 | `auto_response` | T1499 |
| PortScan | 0.980 | `auto_response` | T1046 |
| WebAttack | 0.790 | `human_approval` | T1190 |
| Botnet | 0.000 | `human_approval` | T1071 |
| Infiltration | 0.034 | `human_approval` | T1566 / T1046 |

**Operational numbers (2,000-flow replay):** 324 incidents raised, 362 flows/s, p95 **4.26 ms**/flow;
7 human approvals exercised through the audit log; verdict check on the replay — 324 TP / 0 FP
(stratified split ⇒ upper bound; realistic FP rates come from tiers B–D).

**Known limitations:** Botnet (0.000) and Infiltration (n = 36 samples in the whole dataset) are
effectively unmeasurable on CICIDS2017 — they are automated *never*, and are the main reason the
2018 dataset is next. The classifier is closed-set: a threshold sweep showed no operating point that
recalls unseen-family flows without a 100% false-positive rate, hence the confidence gate and
`unknown_queue`.

---

## Problems faced (and what I did)

1. **Extreme class imbalance** — BENIGN 2.27M vs Heartbleed 11 instances (>200,000:1).
   *Fix:* two-stage cascade + class weighting. Rejected SMOTE — synthetic rows duplicate existing
   flows and would inflate the leakage problem below.
2. **Dirty dataset** — `Inf`/NaN in Flow IAT, en-dash label typos, known CICFlowMeter
   flow-termination bugs. *Fix:* explicit cleaning rules + `normalize_label`, all counted and
   reported (`dataset_report.json`) instead of silently dropped.
3. **Label leakage from near-duplicate flows** — random splits scored 0.9986 while the model had
   effectively memorised duplicates. *Fix:* built a burst-hardened split and published the gap
   (inflation **0.315**) rather than quoting the flattering number.
4. **Classes are day-segregated** — each attack family appears on one weekday only, so stratified
   splits cannot measure generalisation. *Fix:* added a temporal Mon–Thu → Friday tier; it exposed
   recall collapse (0.346), which reshaped the policy design.
5. **Closed-set model cannot say "I don't know"** — unseen families were forced into known classes
   at ~1.0 confidence. *Fix:* confidence floor → `unknown_queue` → human review; quantified with a
   threshold sweep showing no usable operating point.
6. **Local LLM too slow / hallucination risk** — an 8B model timed out at 90 s and 240 s on CPU.
   *Fix:* smaller local model + `max_tokens: 512`, retrieval-grounded prompt ("use ONLY this
   context"), mandatory citations, and a deterministic template fallback that always returns
   (verified live against both 404 and timeout).
7. **Policy-config drift** — settings listed `firewall_block` while the playbook action is
   `firewall_block_source`, so a destructive action had silently bypassed the human gate.
   *Fix:* caught by verifying the audit log against the config; added the vocabulary check to the
   demo run. Lesson: never trust config, trust the emitted events.

---

## Output

**`docs/demo/dashboard.html`** — one self-contained file (embedded data, inline SVG charts, vanilla
JS, base64 images). Two views:

- **Dashboard:** KPI cards (total alerts, TP, FP, pending approvals, throughput, p95 latency),
  severity pie, verdict funnel (total → TP → FP), alerts-per-day stacked bars, automation policy,
  confusion matrices.
- **Alerts:** the incident log with day / severity / family / status filters; clicking a chart
  deep-links here (`#alerts?sev=critical`); clicking a row expands the full response trail —
  actions executed, approval gates, approver, audit notes, and the cited LLM explanation.

**API:** `GET /health`, `POST /predict`, `POST /predict/batch` (+ Swagger UI at `/docs`).

**Logs:** `artifacts/incidents/incidents.jsonl`, `explanations.jsonl`, `artifacts/reports/*.json`.

<!-- add screenshots when ready:
![Dashboard](docs/demo/dashboard.png)
![Alert detail](docs/demo/alert_detail.png)
-->

---

## Quick start

```powershell
cd "C:\Users\tiwar\Desktop\AI SOAR"      # or wherever you cloned the repo
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
```

Creates `venv\`, installs pinned dependencies, writes `.env` from `.env.example`, verifies models
and artifacts, then prints the demo commands. Trained models (~8 MB) and demo artifacts are
committed, so **no dataset download and no retraining** are needed. Docker works too:
`docker build -t ai-soar:latest .` → `docker run --rm -p 8000:8000 ai-soar:latest`.

```powershell
venv\Scripts\python.exe scripts\serve_api.py                                        # API + /docs
venv\Scripts\python.exe scripts\demo_response.py --rows 2000 --fresh --auto-approve  # incidents
venv\Scripts\python.exe scripts\explain_incidents.py --family DDoS --limit 1         # LLM explain
venv\Scripts\python.exe scripts\build_dashboard.py                                   # dashboard
start docs\demo\dashboard.html                                                       # open it
```

Full pipeline from raw CSVs (needs the dataset in `data/raw/`): `build_dataset.py` →
`profile_dataset.py` → `make_stratified_splits.py` → `train_models.py` → `eval_leakage_audit.py` →
`eval_temporal.py` → `eval_threshold_sweep.py`.

---

## Future scope

1. **Cross-dataset validation on CSE-CIC-IDS2018** (data downloaded) — schema/label mapping,
   tier-D report, and a merged-training A/B to recover Botnet and Infiltration.
2. **Open-set / novelty detection** — autoencoder or energy-based scoring so genuinely new attacks
   are labelled `UNKNOWN` instead of being forced into a known family.
3. **Drift monitoring + scheduled retraining** — feature-distribution alarms, model registry with
   versioned rollback, champion/challenger evaluation.
4. **API-level orchestration** — `POST /incidents`, `/incidents/{id}/approve`, `/dismiss` so the
   engine is drivable over HTTP, plus a live dashboard (WebSocket) instead of a static build.
5. **n8n workflow integration** — webhook client + exported workflow JSON: alert → enrich → notify
   (Slack/email) → approval button → callback → close incident.
6. **Real-time ingestion** — streaming replay at 1x/10x/100x, then true pcap → features via
   CICFlowMeter.
7. **Real actuator connectors** — firewall/EDR/DNS/IAM adapters behind the same policy catalog,
   with dry-run mode and rollback.
8. **Alert correlation & incident merging** — group related flows into one campaign/incident
   (source, time window, kill-chain stage) instead of raising one incident per flow.
9. **Threat-intel enrichment** — MISP / VirusTotal / OTX lookups fed into the RAG context so
   explanations carry external corroboration, not just internal policy.
10. **Active-learning feedback loop** — analyst dismissals and approvals become new training labels,
    with champion/challenger evaluation before a model is promoted.
11. **Compliance & evidence export** — map audit events to NIST CSF / ISO 27001 controls and export
    a reviewer-ready evidence pack per incident.
12. **Deployment at scale** — Kubernetes/Helm chart, IaC (Terraform), autoscaling inference workers,
    and production hardening: authn/authz + TLS, rate limiting, multi-tenancy, HA, CI/CD, plus
    per-flow SHAP explanations surfaced in the dashboard.

---

Design decisions, build order and the full evaluation methodology:
[`PROJECT_BRIEF.md`](PROJECT_BRIEF.md).