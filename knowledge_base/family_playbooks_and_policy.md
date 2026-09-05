# AI SOAR Knowledge Base — Attack Families, Playbooks & Response Policy

Source of truth for the LLM explainer (RAG retrieval) and for human reviewers.
Every statement here is backed by the project's own evaluation artifacts; where
the system is weak, this document says so plainly, because an explainer that
cannot admit uncertainty is a liability in a SOC.

## Global response policy

- Detection/triage is performed by a two-stage LightGBM cascade trained on
  CICIDS2017 (binary gate -> 7-family classifier). Responses are executed by
  declarative family playbooks through a SIMULATED actuator backend.
- Automation allowlist (families that may auto-respond) is derived from the
  burst-hardened leakage audit, not from vendor defaults:
  BruteForce 0.999, DDoS 0.999, DoS 0.985, PortScan 0.980 -> automation allowed.
  WebAttack 0.790, Botnet 0.000, Infiltration 0.034 -> human approval required.
- Destructive actions (`host_isolation`, `firewall_block_source`) always wait
  for a human approver, even inside an auto_response incident
  (`settings.response.require_approval_for`).
- Known systemic limitation: a closed-set classifier cannot say "unknown".
  The temporal evaluation showed never-seen families being forced into known
  classes at ~1.0 confidence. Low-confidence and weakly-measured families are
  therefore routed to humans; novelty detection is future work.
- Metric hygiene: stratified-split scores are an UPPER BOUND (near-duplicate
  flows). Honest discrimination numbers come from the burst-hardened split
  (macro-F1 0.6836), the temporal Friday test (gate AUC 0.8527, recall 0.3457)
  and (pending) the CSE-CIC-IDS2018 cross-dataset tier.

## BruteForce

- What it is: repeated credential guessing against SSH/FTP/web logins.
- CICIDS2017 tools: Patator (SSH, FTP, web login variants).
- MITRE ATT&CK: T1110 Brute Force.
- Flow indicators: many short same-size flows to one destination port, high
  flow count per source, regular inter-arrival timing.
- Playbook: rate_limit_source -> firewall_block_source (approval) ->
  create_ticket -> notify_soc.
- Rationale: rate-limiting is reversible and immediately reduces guess rate;
  a full block is destructive (could hide a misconfigured service) so it needs
  a human; ticket drives credential reset follow-up.
- Model note: hardened F1 0.999 — safest family to automate.

## DoS

- What it is: single-source flood exhausting a service's capacity.
- CICIDS2017 tools: Hulk, GoldenEye, Slowloris, Slowhttptest.
- MITRE ATT&CK: T1499 Endpoint Denial of Service.
- Flow indicators: extreme packets/bytes per flow, tiny inter-arrival times,
  one source to one destination.
- Playbook: rate_limit_source -> null_route_target (approval) -> create_ticket
  -> notify_soc.
- Rationale: shedding volume is the fastest reversible relief; null-routing the
  destination also drops legitimate traffic, hence approval-gated.
- Model note: hardened F1 0.985.

## DDoS

- What it is: coordinated multi-source flood (LOIC variants in CICIDS2017).
- MITRE ATT&CK: T1498 Network Denial of Service.
- Flow indicators: many sources to one destination, synchronized bursts,
  homogeneous flow sizes.
- Playbook: null_route_target (approval) -> rate_limit_source -> create_ticket
  -> notify_soc.
- Rationale: multi-source floods defeat per-source rate limits; upstream
  relief comes first but is destructive, so a human approves.
- Model note: hardened F1 0.999; note DDoS and DoS are fingerprint-similar —
  confusion between them is expected and operationally tolerable because both
  playbooks start with volume shedding.

## PortScan

- What it is: reconnaissance — enumerating open ports/services.
- CICIDS2017 tools: nmap variants.
- MITRE ATT&CK: T1046 Network Service Discovery.
- Flow indicators: many tiny flows to many destination ports from one source,
  mostly unanswered SYNs.
- Playbook: add_to_watchlist -> create_ticket.
- Rationale: scanning alone is not an outage; disrupting a scanner can tip off
  an attacker. Watch + log is the proportionate response.
- Model note: hardened F1 0.980; severity LOW by policy.

## WebAttack

- What it is: application-layer abuse — SQL injection, XSS, brute-force logins
  against a web app (CICIDS2017 uses DVWA + selenium harness).
- MITRE ATT&CK: T1190 Exploit Public-Facing Application (SQLi/XSS payloads).
- Flow indicators: anomalous URI payload sizes, irregular request timing,
  error-rate bursts (visible as response-size variance in flow features).
- Playbook: waf_block (approval) -> capture_forensics -> create_ticket ->
  notify_soc.
- Rationale: virtual patching changes application behaviour, so approval first;
  evidence capture matters because injection attempts precede real breaches.
- Model note: hardened F1 0.790 and only 2,143 rows in CICIDS2017 — automation
  NOT allowed; every WebAttack incident waits for a human.

## Botnet

- What it is: compromised hosts taking commands from a controller (C2).
- CICIDS2017 tools: Ares bot framework.
- MITRE ATT&CK: T1071 Application Layer Protocol (C2), T1573 Encrypted Channel.
- Flow indicators: periodic beaconing, small uniform payloads to external IPs.
- Playbook: host_isolation (approval) -> capture_forensics -> create_ticket ->
  notify_soc.
- Rationale: isolation is the only containment that stops C2, but it takes a
  host offline — always a human decision; forensics before cleanup.
- Model note: hardened F1 0.000 (within-day drift in CICIDS2017) — the model's
  weakest family; incidents are held for humans by policy.

## Infiltration

- What it is: an insider-side foothold pivoting internally (email-delivered
  backdoor, then internal scanning).
- MITRE ATT&CK: T1566 Phishing (initial), T1046/T1021 internal recon & movement.
- Flow indicators: internal-to-internal scan patterns after an external drop.
- Playbook: host_isolation (approval) -> capture_forensics -> create_ticket ->
  notify_soc.
- Rationale: lateral movement escalates fast; contain first, preserve evidence.
- Model note: only 36 rows in CICIDS2017 — statistically unmeasurable
  (hardened F1 0.034). CSE-CIC-IDS2018 (~160k rows reported) is the planned
  remedy; until then every Infiltration verdict is human-gated.

## BENIGN

- Not an incident. The binary gate logs and drops benign flows; no playbook
  runs. False positives on benign traffic are the gate's cost and are measured
  in the temporal evaluation (89 FP at threshold 0.5 on Friday).