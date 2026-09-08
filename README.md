# AI SOAR

**ML-driven Security Orchestration, Automation & Response.**
A SOAR platform that replaces signature-based detection and first-line triage with a calibrated
two-stage classifier, keeps response declarative and auditable, and explains every action with an
RAG-grounded LLM.

> **Scope note:** research/demo build. Models are trained on CICIDS2017 network flow data and
> evaluated on four progressively harder tiers, ending with **10,749,490 flows from
> CSE-CIC-IDS2018** — an environment the models never saw. Detection, response orchestration and
> explanation are real and measured; **actuators are simulated** (no live firewall/EDR/DNS is
> touched). **n8n orchestration is implemented and validated** (webhook client +
> importable workflow, tested end-to-end against n8n 2.38 self-hosted); live pcap
> ingestion remains future work.

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
| False positives | static thresholds, hand-tuned per environment | calibrated probabilities + a documented threshold sweep; **0.0374 % FP measured on 9.65 M benign flows from an unseen environment** |
| Behaviour in a new network | rules need re-tuning per site, silently | measured, published, and root-caused (tier D) instead of assumed |
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
attack it has never seen. Tier D proved the drift point on 10.75 M unseen flows: detection recall
fell to 8.16 % even though the model still *ranked* attacks correctly (AUC 0.8789). That is exactly
why this build keeps human approval gates and an `unknown_queue` rather than claiming full
autonomy.

**Net effect, in one line each:**

1. **Speed** — triage drops from "as fast as an analyst can read" to milliseconds per flow.
2. **Evidence-based automation** — automation is granted per family only where evaluation proved
   reliability (DDoS 0.999 → auto; Botnet 0.000 → always human), instead of an all-or-nothing switch.
3. **Explainability by default** — every incident ships with a cited, MITRE-mapped justification,
   so an analyst can challenge the machine's reasoning instead of guessing it.
4. **Auditability** — approvals, dismissals and actuators are event-sourced, which is what a
   compliance reviewer actually asks for.
5. **Measured honesty** — the project publishes its own inflation *and* its own worst case:
   0.9986 (stratified upper bound) → 0.6836 (burst-hardened) → 0.2091 (unseen day) → **0.1544**
   (unseen environment, 10.75 M flows). That gradient is the difference between a demo and a
   defensible result.

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
              │  single-file HTML dashboard  ·  n8n workflow │
              └──────────────────────────────────────────────┘
```

| Layer | Module | Responsibility |
|---|---|---|
| Ingestion | `src/ai_soar/data/` | schema, chunked loading, cleaning, label collapse, splits, **2018 schema adapter** |
| Models | `src/ai_soar/models/` | binary gate + family classifier (LightGBM), registry/metadata |
| Evaluation | `src/ai_soar/evaluation/` | macro/per-class F1, confusion matrices, leakage audit, **four-tier protocol** |
| Inference | `src/ai_soar/inference/` | FastAPI service, cascade, confidence gate, decision policy |
| Response | `src/ai_soar/response/` | policy catalog, playbooks, actuators, approval gates, audit store |
| Explain | `src/ai_soar/explain/` | RAG store/retrieval, LLM client (4 providers), explainer + fallback |
| Config | `src/ai_soar/config.py` | typed settings; precedence **`.env` > `settings.yaml` > code defaults** |

---

## Features

- **Two-stage cascade** — cheap binary gate first, expensive 7-way classifier only on flagged flows.
- **Evidence-gated automation** — each family's automation level comes from its burst-hardened F1.
- **Four-tier evaluation protocol** — stratified upper bound → burst-hardened → temporal → cross-dataset,
  each tier removing exactly one flattering assumption, all four published.
- **Schema adapter at the ingestion boundary** — adding CSE-CIC-IDS2018 (18 renamed columns, one
  duplicated column, four extra identity columns, a different label vocabulary) required **zero
  changes to the inference path**: models, predictor, response engine, explainer and dashboard were
  untouched. The adapter converts 2018 flows into the 2017 canonical shape at the door.
- **Two human approval gates** for destructive actions; every approval is event-sourced with actor id.
- **Append-only audit trail** (JSONL) — incidents, actions, approvals, dismissals, notes.
- **Grounded LLM explanations** — retrieval-augmented, citations per explanation, MITRE ids;
  deterministic template fallback so explanation never blocks or crashes the pipeline.
- **Offline-first LLM** — Ollama local model; OpenAI/Anthropic/Gemini also supported via `.env`.
- **Single-file dashboard** — no server, no CDN, no build step; opens from `file://`.
- **Strict input contract** — API rejects feature drift with HTTP 422 instead of guessing.
- **n8n orchestration, fail-open** — incident events push to an n8n webhook (one attempt,
  3 s timeout, disabled by default); an importable workflow routes human approvals back
  through the API. n8n being down never blocks detection, response or audit.
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
| Orchestration | n8n 2.x — webhook client (`ai_soar.orchestration.n8n`) + importable workflow (`n8n_automation/workflows/`); the notification/approval plane only, never the detection plane |
| Storage | **no external database** — parquet for data, append-only JSONL for incidents/audit, an in-memory TF-IDF index for RAG, JSON for reports. Benign flows never reach the store, so write volume is *incident* volume, not flow volume. |
| Packaging | `bootstrap.ps1` (Windows), Docker (python:3.12-slim + libgomp1) |
| Dataset (train) | CICIDS2017 — 2,830,743 flows, 80 features, 15 labels collapsed to 8 families |
| Dataset (evaluate) | CSE-CIC-IDS2018 — 13,191,623 raw flows over 6 days → 10,749,490 clean. Used for tier D **only**: never for training, validation, threshold selection or feature choice. |

---

## Results & metrics

**Data pipeline — CICIDS2017 (training):** 2,830,743 raw rows → 2,616,379 clean rows (4,376 `Inf`
cells fixed, 2,867 NaN rows dropped, **211,497 duplicate flows removed**), 15 raw labels collapsed
to 8 families.

**Data pipeline — CSE-CIC-IDS2018 (evaluation):** 13,191,623 raw rows → **10,749,490** clean rows
(1 junk header row, 74,595 rows dropped for `Inf`/NaN, **2,367,537 duplicates removed**). Duplicate
rates were wildly family-dependent — 75.2 % of BruteForce rows and 66.7 % of DoS rows were repeats,
versus 11.2 % of benign — so dedup was attributed per family rather than reported as one number.
Built by streaming chunks with a sliding-window dedup, because these CSVs are grouped by victim
machine rather than ordered by time.

**Evaluation — read in this order.** CICIDS2017 contains near-duplicate flows, so a random split
inflates every score. Each tier answers a different question:

| Tier | Question | Result |
|---|---|---|
| A. Stratified test | upper bound (duplicates present) | stage-1 AUC **1.000** / F1 0.9974; stage-2 macro-F1 **0.9986**; cascaded 0.9554 |
| B. Burst-hardened split | honest discrimination | macro-F1 **0.6836** → measured **inflation 0.315** |
| C. Temporal Mon–Thu → Friday | unseen day | gate AUC 0.8527, recall 0.3457 @ 0.5; system macro-F1 0.2091 |
| D. Cross-dataset → CSE-CIC-IDS2018 | unseen environment, 10,749,490 flows | gate AUC **0.8789**; recall **8.16 %** @ 0.5; **FP 0.0374 %**; macro-F1 **0.1544** |

### Tier D in detail — the models meet a network they never saw

10,749,490 flows scored in **250 s (~43,000 flows/s)**, **0 non-finite cells**.

**Gate:** AUC **0.8789** · detection recall **8.1572 %** (CI95 8.1062–8.2084) · **false-positive
rate 0.0374 %** (CI95 0.0362–0.0386) on **9,648,065 benign flows** · specificity **99.9626 %**.

| Family | Support | Recall @0.5 | Precision | F1 | Passed gate | Stage-2 correct \| given gate |
|---|---|---|---|---|---|---|
| DoS | 200,455 | **44.69 %** | **99.09 %** | 0.6160 | 44.74 % | **99.91 %** |
| BruteForce | 94,711 | 0.09 % | 3.25 % | 0.0017 | 0.16 % | 54.55 % |
| DDoS | 805,945 | **0.00 %** | 0.00 % | 0.0000 | 0.00 % | — |
| WebAttack | 314 | 0.00 % | 0.00 % | 0.0000 | 0.00 % | — (LOW SUPPORT) |
| PortScan / Botnet / Infiltration | 0 | not present in the six downloaded days — **excluded from the macro average, never scored as zero** | | | | |

Macro over the 4 covered attack families: P 0.2558 · R 0.1120 · **F1 0.1544**.

**What the SOAR would actually have done:**

| Routing decision | Flows | Attack | Benign |
|---|---|---|---|
| `log_only` | 10,656,039 (99.13 %) | 1,011,580 | 9,644,459 |
| `auto_response` | 93,383 (0.87 %) | 89,842 | 3,541 |
| `human_approval` | 5 | 2 | 3 |
| `unknown_queue` | 63 | 1 | 62 |

→ **when the system acts alone it is right 96.2 % of the time** (89,842 / 93,383). The
cross-environment failure mode is *missed attacks*, not runaway automation: 3,541 benign flows
(0.0367 %) would have been auto-isolated across six days of traffic.

**Root cause — isolated, not guessed.** `scripts/diagnose_tier_d.py` scores the 2017 test split as a
control, sweeps nine thresholds, and tests every high-gain feature for a unit/scale mismatch under
three simultaneous conditions (IQR moved >5×, p5–p95 windows non-overlapping, and one factor
explaining every quantile), per file as well as pooled — because pooling six days can hide a
single-day mapping error. It cleared feature misalignment (booster metadata == `FEATURE_COLUMNS`)
and cleared unit mismatches, then localised the real cause: **2018 attacks carry roughly 5× smaller
packet-length statistics than 2017 attacks** (Bwd Packet Length Std 1,755 → 268.8; Average Packet
Size 795.7 → 109.7), and those are precisely the gate's highest-gain features.

**Calibration ≠ discrimination.** Tier D *ranks* better than tier C (AUC 0.8789 vs 0.8527, per-day
0.83–0.99) yet recalls far less at the deployed cut (8.16 % vs 34.57 %). The deficit is in the
probability scale, not in the model's ability to separate attack from benign — which is why the
numbers below are reported as *recall at a threshold calibrated on 2017*, always next to the AUC.

**Internal consistency:** per-day family counts reconcile exactly with the build census —
DDoS = Tuesday-20 (575,471) + Wednesday-21 (230,474) = 805,945; DoS = Friday-16 (200,455);
BruteForce = Wednesday-14 (94,143) + 568 = 94,711; and the four routing decisions sum to
10,749,490.

**Deliberate choice:** neither the 0.5 gate cut nor the per-family automation allowlist was re-tuned
on 2018 data — that would be fitting the evaluation set. Re-deriving both from cross-dataset
evidence is future work (merged-training A/B), with its own split.

**Operational numbers (2,000-flow replay):** 324 incidents raised, 362 flows/s, p95 **4.26 ms**/flow;
7 human approvals exercised through the audit log; n8n alert-webhook round-trip measured at
**60 ms** per execution (n8n 2.38 self-hosted); 324 TP / 0 FP on that replay — the zero is a
stratified-split artifact, and the realistic figure is tier D's **0.0374 %**.

**Known limitations:**

- **DDoS is blind cross-environment** — 0 of 805,945 flows, and DDoS is the family the hardened
  evidence had automated *most* confidently (F1 0.999). An allowlist derived from same-distribution
  evidence does not transfer between capture environments.
- **The confidence floor is not a novelty detector** — only 63 of 10,749,490 flows reached
  `unknown_queue`. An unseen-family attack is classified into a *known* family at ~1.0 confidence,
  so containment comes from the human-approval route and per-family allowlisting, not from the gate.
- Botnet (0.000) and Infiltration (n = 36 in the whole of CICIDS2017) remain unmeasurable, and the
  six downloaded 2018 days contain **no PortScan, Botnet or Infiltration at all** — CSE-CIC-IDS2018
  has no PortScan label, its nmap scan being part of the Infiltration scenario — so tier D can score
  only 4 of 7 families.
- 5 of the 6 2018 CSVs contain exactly 1,048,575 rows (2²⁰−1, Excel's row limit), so those days are
  capped subsets; and since the files are machine-grouped rather than time-ordered, no temporal tier
  is possible on 2018.
- WebAttack support is 314 flows, flagged LOW SUPPORT, so its interval is wide.

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
8. **A cross-dataset smoke test that looked exactly like a bug** — the first 2018 run reported gate
   AUC 0.856 with detection recall of *precisely* 0.0000 %, which is either a serious finding or a
   broken feature mapping, and the two need opposite responses. *Fix:* wrote a diagnostic that
   scores the 2017 test split as a control, sweeps thresholds, and tests each high-gain feature for
   a unit/scale mismatch only when three conditions hold together (>5× IQR move, non-overlapping
   p5–p95 windows, one factor explaining every quantile) — checked per file as well as pooled, since
   pooling six days hides a single-day mapping error. It cleared alignment and units and localised a
   genuine packet-length scale shift. Two lessons: never judge a dataset from a prefix when rows are
   grouped by machine (that slice was the single worst-case family), and always publish AUC next to
   recall@threshold, because a threshold calibrated in one environment does not transfer.
9. **The orchestrator fought back (n8n 2.x)** — the imported workflow failed at runtime twice with
   the same cryptic error (`compareOperationFunctions[...] is not a function`): first because the
   legacy IF node's boolean operation names changed across n8n majors, then because the Webhook
   node delivers a JSON payload *nested under `.body`*, so every `$json.field` expression silently
   read `undefined` and routed to the wrong branch — a **green execution that was quietly wrong**.
   *Fix:* stopped guessing formats, exported the workflow **from n8n itself** and treated that as
   ground truth, switched conditions to string equality, and made every expression
   `($json.body || $json).field` so it survives both payload shapes. Lesson: when a third-party
   engine rejects your config, let the engine author the config and patch the minimum.

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

**API:** `GET /health`, `POST /predict`, `POST /predict/batch`, `POST /ingest` (score *and*
respond), `GET /incidents` (+ `/incidents/summary`, `/incidents/{id}`),
`POST /incidents/{id}/approve`, `POST /incidents/{id}/dismiss` (+ Swagger UI at `/docs`).

**Logs:** `artifacts/incidents/incidents.jsonl`, `explanations.jsonl`.

**Orchestration:** `n8n_automation/workflows/soar_alert_pipeline.json` — import-ready workflow:
alert webhook → human-approval routing → SOC notification slot, plus a decision webhook that
calls the approve/dismiss API routes back. Validated end-to-end on n8n 2.38 self-hosted
(60 ms execution; alert and approval round-trips green).

**Reports (`artifacts/reports/`):** `dataset_report.json`, `profile_report.json`,
`leakage_audit_report.json`, `temporal_eval_report.json`, `threshold_sweep_report.json`,
`cicids2018_verify_report.json`, `cicids2018_build_report.json`, `tier_d_diagnostic.json`,
**`tier_d_cross_dataset.json`** + `tier_d_confusion_2018.png`.

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
start docs\demo\dashboard.html                                                          # open it
npx n8n                                                                              # orchestration plane
venv\Scripts\python.exe -m ai_soar.orchestration.n8n --test --force                     # webhook wiring
```

Full pipeline from raw CSVs (needs the dataset in `data/raw/`): `build_dataset.py` →
`profile_dataset.py` → `make_stratified_splits.py` → `train_models.py` → `eval_leakage_audit.py` →
`eval_temporal.py` → `eval_threshold_sweep.py`.

Cross-dataset tier D (needs the CSE-CIC-IDS2018 CSVs in `data/external/cicids2018/`, ~5.45 GB, not
committed): `verify_cicids2018.py --full` → `build_dataset2018.py` → `diagnose_tier_d.py` →
`eval_cross_dataset.py` (~4 min for 10.75 M flows, ≲1 GB RAM).

---

## Future scope

1. **Merged-training A/B + recalibration (Step 9b)** — retrain on CICIDS2017 *and* CSE-CIC-IDS2018
   with a split that respects machine/day grouping, then re-derive both the 0.5 gate cut and the
   per-family automation allowlist from **cross-dataset** evidence instead of same-distribution
   evidence. Adding the 01-03 and 02-03-2018 captures would also restore PortScan / Botnet /
   Infiltration coverage, which tier D could not score at all.
2. **Open-set / novelty detection** — autoencoder or energy-based scoring so genuinely new attacks
   are labelled `UNKNOWN` instead of being forced into a known family. Tier D showed the confidence
   floor cannot do this job: 63 of 10.75 M flows reached the unknown queue.
3. **Drift monitoring + scheduled retraining** — feature-distribution alarms (the tier-D diagnostic
   already computes per-feature shift statistics, so it is the natural probe), model registry with
   versioned rollback, champion/challenger evaluation.
4. **Live operations surface** — WebSocket-driven dashboard instead of a static build, plus
   alert-correlation views over the incident log. (The HTTP incident/approve/dismiss API
   shipped together with the n8n work.)
5. **n8n channel integrations** — wire the workflow's notification slot to Slack/Gmail with a
   one-click approval button posting to `/webhook/ai-soar-decision`. The client, routing and
   API callbacks are already in place and validated; only the channel credential is missing.
6. **Real-time ingestion** — streaming replay at 1x/10x/100x driven by real flow timestamps (which
   means preserving `Timestamp` through the builder as a non-feature column), then true pcap →
   features via CICFlowMeter. On the storage side: hourly-rotated JSONL plus an in-memory index for
   `latest()`, moving to SQLite (WAL) only if query volume demands it — benign flows never reach the
   store, so write volume stays at incident volume.
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
