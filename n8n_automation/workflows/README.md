# n8n workflow — AI SOAR alert pipeline

`soar_alert_pipeline.json` is an import-ready n8n workflow. It is the **human side**
of the SOAR loop: notifications and approval decisions. Detection, policy and
response stay in Python — this workflow never decides whether something is an attack.

```
                       ┌─────────────────────────── ALERT SIDE ───────────────────────────┐
 Python (n8n.py) ────> │ Webhook /webhook/ai-soar ─> IF needs_human ─┬─ true ─> Code       │
                       │                                             │         "Format SOC │
                       │                                             │          Alert" ───>│ NoOp (attach Slack/Gmail)
                       │                                             └─ false ─> NoOp      │
                       └────────────────────────────────────────────────"Auto-Response     │
                                                                        Logged"            │
                       ┌────────────────────────── DECISION SIDE ──────────────────────────┐
 Slack button /  ────> │ Webhook /webhook/ai-soar-decision ─> IF decision == "approve"      │
 dashboard / curl      │        ├─ true ─> HTTP POST /incidents/{id}/approve ─┐             │
                       │        └─ false ─> HTTP POST /incidents/{id}/dismiss ┴─> NoOp      │
                       └──────────────────────────────────────────────"Decision Applied"   │
```

Both sides are independent: the alert side works with no decision traffic, and the
decision side can be driven by `curl` alone if you never wire up a chat button.

---

## 1. Run n8n

Pick one. Local npm is the fastest for a demo.

```powershell
# A) npm (needs Node 18/20/22)
npx n8n

# B) Docker
docker run -it --rm -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n
```

Open <http://localhost:5678> and create the local owner account when prompted.

> If n8n runs in Docker and the SOAR API runs on your Windows host, `localhost`
> inside the container is NOT your machine. Use `http://host.docker.internal:8000`
> in the two HTTP Request nodes.

## 2. Import the workflow

n8n UI → **Workflows** → **⋯** (top right) → **Import from File…** →
select `n8n_automation/workflows/soar_alert_pipeline.json`.

Then click **Active** (top right toggle) so the production webhook paths
(`/webhook/...`) are registered. While the workflow is inactive only the
`/webhook-test/...` paths answer, and only when you press **Execute Workflow**.

## 3. Point the SOAR at n8n

`config/settings.yaml`:

```yaml
n8n:
  enabled: true                     # was false
  base_url: http://localhost:5678
  webhook_path: /webhook/ai-soar    # must match the Webhook node's path
```

Restart the API if it is running (`enabled` is read at first use).

## 4. Test the wiring

```powershell
cd "C:\Users\tiwar\Desktop\AI SOAR"
$env:PYTHONPATH="src"

# send one synthetic alert event (works with n8n.enabled=false too)
.\venv\Scripts\python.exe -m ai_soar.orchestration.n8n --test --force

# see the exact JSON payload without sending it
.\venv\Scripts\python.exe -m ai_soar.orchestration.n8n --show
```

`delivered: True` + an execution in the n8n UI = wiring is good.

### Which code paths actually push to n8n

Only the **HTTP API** calls `notify_incident()`: `POST /ingest`, and
`POST /incidents/{id}/approve` / `/dismiss`. `scripts/demo_response.py` runs the
predictor and the response engine **in-process** (no HTTP), so it writes incidents to
the JSONL log but does *not* notify n8n — by design, since a benchmarking driver should
not depend on an orchestrator being up.

So for a live end-to-end demo, generate incidents with the script and then drive the
**decision** through the API:

```powershell
# terminal 1 - create incidents (in-process, fast: ~362 flows/s)
.\venv\Scripts\python.exe scripts\demo_response.py --rows 2000 --fresh

# terminal 2 - serve the API over the same incident log
.\venv\Scripts\python.exe scripts\serve_api.py

# terminal 3 - find something pending, then approve it through the API.
# This is the call that fires the n8n webhook with event=approved.
Invoke-RestMethod http://localhost:8000/incidents/summary
Invoke-RestMethod -Method Post -Uri http://localhost:8000/incidents/INC-XXXXXXXX/approve `
  -ContentType "application/json" -Body '{"approver":"nimisha"}'
```

To fire the **alert** side (`event=approval_required`) instead, either use the
`--test --force` self-test above, or score a flow through `POST /ingest`.

## 5. Close the loop from the decision side

```powershell
# list what is waiting for a human
Invoke-RestMethod http://localhost:8000/incidents/summary

# approve through n8n (this exercises the decision webhook + the HTTP callback)
Invoke-RestMethod -Method Post -Uri http://localhost:5678/webhook/ai-soar-decision `
  -ContentType "application/json" `
  -Body '{"incident_id":"INC-XXXXXXXX","decision":"approve","approver":"nimisha"}'

# or dismiss
Invoke-RestMethod -Method Post -Uri http://localhost:5678/webhook/ai-soar-decision `
  -ContentType "application/json" `
  -Body '{"incident_id":"INC-XXXXXXXX","decision":"dismiss","approver":"nimisha","reason":"false positive"}'
```

**Decision webhook body:**

| Field | Required | Meaning |
|---|---|---|
| `incident_id` | yes | `INC-XXXXXXXX` from the alert or from `/incidents/summary` |
| `decision` | yes | `approve` or `dismiss` (anything else takes the dismiss branch) |
| `approver` | no | written to the audit log; defaults to `n8n-approver` |
| `reason` | no | recorded on dismiss; defaults to `dismissed via n8n` |

You can also call the SOAR API directly and skip n8n entirely — the workflow is a
convenience layer, not a dependency:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/incidents/INC-XXXXXXXX/approve `
  -ContentType "application/json" -Body '{"approver":"nimisha"}'
```

---

## Alert payload (what the Webhook node receives)

Emitted by `src/ai_soar/orchestration/n8n.py::build_payload()`. Keys are hand-built
and stable on purpose, so workflow expressions keep working when the incident model grows.

| Key | Example | Notes |
|---|---|---|
| `event` | `approval_required` | `approval_required` \| `approved` \| `dismissed` \| `resolved` \| `opened` |
| `incident_id` | `INC-1A2B3C4D` | |
| `family` | `DDoS` | one of the 7 attack families |
| `severity` | `critical` | `low` \| `medium` \| `high` \| `critical` |
| `status` | `pending_approval` | incident status after this event |
| `decision` | `human_approval` | policy decision from the cascade |
| `decision_reason` | `family DDoS requires approval…` | |
| `confidence` | `0.999998` | max stage-2 probability |
| `gate_probability` | `0.999995` | stage-1 P(malicious) |
| `needs_human` | `true` | **the field the IF node routes on** |
| `pending_actions` | `["firewall_block_source"]` | held destructive actions |
| `playbook` | `["notify_soc", "firewall_block_source"]` | planned steps |
| `actions[]` | `{action, actuator, status, simulated, approved_by, message}` | full response trail |
| `notes[]` | `["approved by nimisha: …"]` | last 5 audit notes |
| `callback.detail` / `.approve` / `.dismiss` | `http://localhost:8000/incidents/INC-…/approve` | ready-made URLs |
| `model_version`, `request_id`, `created_at`, `sent_at`, `source` | | provenance |

Override the advertised host with `AI_SOAR_API_BASE_URL` (e.g. behind a reverse proxy)
so the `callback.*` URLs stay clickable from n8n.

---

## Add a real notification channel

The workflow imports with **zero credentials** — the notification step is a labelled
no-op. To make it real:

1. Open **Notify SOC (attach Slack or Gmail)**.
2. Add a Slack / Gmail / Telegram node after it (Credentials → New).
3. Map fields from the Code node: `{{ $json.headline }}`, `{{ $json.summary }}`,
   `{{ $json.pending_actions }}`, `{{ $json.approve_url }}`, `{{ $json.dismiss_url }}`.
4. For a one-click approval button, point the button at
   `http://<n8n-host>:5678/webhook/ai-soar-decision` with a JSON body built from
   `{{ $json.incident_id }}`.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `delivered: False`, n8n shows nothing | Workflow not **Active** → production path returns 404. Activate it, or use `--url http://localhost:5678/webhook-test/ai-soar` while editing. |
| `delivered: False`, "Connection refused" | n8n not running, or `n8n.base_url` points at the wrong host/port (`AI_SOAR_N8N_BASE_URL`). |
| Payload arrives but the IF node always takes the false branch | `$json.needs_human` missing → you are on an older `n8n.py`; re-extract the update zip. |
| HTTP Request node returns 404 | Wrong incident id, or the API is not on `localhost:8000` (Docker: use `host.docker.internal`). |
| HTTP Request node returns 409 | Nothing was pending approval — the incident was already approved/dismissed. |
| HTTP Request node returns 503 | The SOAR response engine failed to start (actuator/store config); `/health` will say so. |
| Notifications are slow | They cannot be: one attempt, 3 s timeout (`AI_SOAR_N8N_TIMEOUT`), fail-open. The SOAR never waits on n8n. |

**By design:** if n8n is down, disabled, or unreachable, `notify_incident()` returns
`False`, logs a warning, and the incident is still detected, responded to and audited.
Orchestration is an enhancement, never a dependency.
