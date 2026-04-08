---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
  - email
  - triage
  - benchmark
pinned: false
license: mit
---

# 📧 Email Triage OpenEnv

An OpenEnv benchmark environment where AI agents learn to **prioritize, categorize, and route emails** using contextual understanding. Agents interact with realistic email scenarios and receive reward signals that encourage accurate triage decisions.

---

## 🧩 Environment Description

Email triage is a high-value real-world task performed by operations teams, customer support, legal departments, and executives every day. Poor email triage leads to missed SLA deadlines, legal exposure, revenue loss, and operational inefficiency.

This environment challenges agents to:
- Detect urgency and priority signals (sender authority, keywords, deadlines)
- Classify emails into business categories (support, billing, legal, etc.)
- Route emails to the correct team queue
- Extract concrete action items
- Identify compliance and legal risks (GDPR, HIPAA, DMCA, financial)
- Assign appropriate SLA deadlines

---

## 🏗️ Architecture

```
├── inference.py          # Baseline inference script (required at root by OpenEnv spec)
├── validate.py           # Pre-submission validation script
├── openenv.yaml          # OpenEnv metadata and spec
├── Dockerfile            # Container definition for HF Spaces
├── requirements.txt
├── test_environment.py   # Self-contained test suite
├── run_local.sh          # Local run script (Linux/Mac)
├── run_local.bat         # Local run script (Windows)
├── .env.example          # Sample environment variable config
├── server/
│   ├── __init__.py
│   ├── app.py            # FastAPI HTTP server (OpenEnv API)
│   ├── environment.py    # Core env logic: reset/step/state
│   ├── models.py         # Pydantic typed models
│   └── graders.py        # Task-specific grading functions
└── data/
    ├── __init__.py
    ├── emails.py         # Data loader (loads JSON files dynamically)
    ├── easy.json         # 8 ambiguous emails for priority-classification
    ├── medium.json       # 6 emails for category-routing
    └── hard.json         # 5 high-stakes emails for full-triage-pipeline
```

---

## 📊 Observation Space

Each observation contains:

| Field | Type | Description |
|---|---|---|
| `email_id` | string | Unique email identifier |
| `subject` | string | Email subject line |
| `sender` | string | Sender email address |
| `sender_domain` | string | Sender's domain (credibility signal) |
| `body` | string | Full email body text |
| `timestamp` | string | ISO 8601 timestamp |
| `thread_length` | int | Number of emails in thread |
| `has_attachments` | bool | Whether attachments are present |
| `step` | int | Current step number |
| `previous_actions` | list | Last 3 actions for context |
| `feedback` | string | Feedback from previous step |
| `task_name` | string | Active task name |

---

## 🎯 Action Space

Agents submit a JSON object with any/all of these fields:

| Field | Type | Values |
|---|---|---|
| `priority` | enum | `critical`, `high`, `medium`, `low` |
| `category` | enum | `support`, `billing`, `sales`, `legal`, `hr`, `engineering`, `spam`, `other` |
| `route_to` | string | Team queue name (e.g., `"legal-team"`, `"engineering-oncall"`) |
| `action_items` | list[str] | Concrete actions to take |
| `sla_hours` | int | SLA deadline in hours |
| `sentiment` | enum | `positive`, `neutral`, `negative`, `urgent` |
| `flags` | list[str] | From: `pii`, `legal_risk`, `financial`, `escalate` |
| `reasoning` | string | Agent's explanation |

---

## 📋 Tasks

### Task 1: `priority-classification` 🟢 Easy
**Goal:** Classify each email's urgency level and sentiment.

**Graded fields:** `priority` (85%), `sentiment` (15%)

**Scoring:**
- Priority uses an adjacency matrix — adjacent priorities get partial credit (e.g., `critical`→`high` = 0.5)
- No penalty for missing optional fields

**Dataset:** 8 emails ranging from production outages to newsletter subscriptions

**Expected baseline score:** 0.60–0.80

---

### Task 2: `category-routing` 🟡 Medium
**Goal:** Classify priority + category, route to correct team, extract action items, identify risk flags.

**Graded fields:** `priority` (25%), `category` (30%), `route_to` (20%), `action_items` (15%), `flags` (10%)

**Scoring:**
- Category is exact match only
- Routing has partial credit for keyword overlap
- Action items scored by keyword coverage (≥50% match per item)
- Missing `escalate` or `legal_risk` flags incur penalty

**Dataset:** 6 emails including API issues, billing disputes, partnership inquiries, DMCA notices

**Expected baseline score:** 0.35–0.60

---

### Task 3: `full-triage-pipeline` 🔴 Hard
**Goal:** Complete end-to-end triage across all dimensions.

**Graded fields:** `priority` (20%), `category` (20%), `route_to` (15%), `action_items` (20%), `sla_hours` (10%), `sentiment` (5%), `flags` (10%)

**Scoring:**
- All 7 fields contribute to score
- Extra penalty (-0.15) for missing critical priority on critical-grade emails
- SLA tolerance varies by priority tier (critical: ±2h, high: ±8h, etc.)

**Dataset:** 5 high-stakes emails including HIPAA violations, M&A interest, ransomware attacks, GDPR requests

**Expected baseline score:** 0.20–0.45

---

## 🏆 Reward Function

Rewards are provided after **every step** (not just at episode end), enabling agents to learn from trajectory:

- Each email is an independent triage decision
- Reward = weighted sum of dimension scores − penalties
- Penalties: false flag escalations, missing critical flags, misclassifying critical emails
- Episode score = mean reward across all steps (normalized to [0, 1])

**Partial progress signals:**
- Getting priority right earns reward even if category is wrong
- Correct action items earn reward even with wrong routing
- Adjacent priorities get partial credit

---

## 🚀 Setup & Usage

### Local Development

```bash
# Clone the repo
git clone https://huggingface.co/spaces/your-team/email-triage-env
cd email-triage-env

# Install dependencies
pip install -r requirements.txt

# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# In another terminal, run the baseline inference script
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Docker

```bash
# Build
docker build -t email-triage-env .

# Run the environment
docker run -p 7860:7860 email-triage-env

# Run inference against it
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

### Run a specific task

```bash
export EMAIL_TRIAGE_TASK=priority-classification   # easy
export EMAIL_TRIAGE_TASK=category-routing          # medium
export EMAIL_TRIAGE_TASK=full-triage-pipeline      # hard
export EMAIL_TRIAGE_TASK=all                       # all three
python inference.py
```

---

## 🌐 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Submit triage action |
| `/state` | GET | Current environment state |
| `/tasks` | GET | List all tasks |

### Example: Reset

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "priority-classification", "session_id": "my-agent"}'
```

### Example: Step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "priority": "critical",
      "category": "engineering",
      "route_to": "engineering-oncall",
      "action_items": ["page on-call engineer", "check monitoring dashboard"],
      "sla_hours": 1,
      "sentiment": "urgent",
      "flags": ["escalate"],
      "reasoning": "Production outage affecting all users"
    },
    "session_id": "my-agent"
  }'
```

---

## 📈 Baseline Scores

Tested with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Difficulty | Baseline Score |
|---|---|---|
| `priority-classification` | Easy | ~0.58 |
| `category-routing` | Medium | ~0.51 |
| `full-triage-pipeline` | Hard | ~0.58 |

---

## ✅ OpenEnv Compliance

- ✅ Typed Pydantic models: `EmailObservation`, `TriageAction`, `TriageReward`
- ✅ `step(action)` → returns observation, reward, done, info
- ✅ `reset()` → returns initial observation
- ✅ `state()` → returns full episode state
- ✅ `openenv.yaml` with full metadata
- ✅ 3 tasks with deterministic graders, scores in [0.0, 1.0]
- ✅ Meaningful per-step reward (not just binary end-of-episode)
- ✅ Baseline inference script (`inference.py`) using OpenAI client
- ✅ Dockerfile + HuggingFace Space deployment
- ✅ Runtime < 20 minutes, compatible with 2 vCPU / 8GB RAM

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes | HuggingFace API key |
| `API_BASE_URL` | Yes | LLM API endpoint |
| `MODEL_NAME` | Yes | Model identifier |
| `ENV_BASE_URL` | No | Environment server URL (default: http://localhost:7860) |
| `EMAIL_TRIAGE_TASK` | No | Task to run: `all`, `priority-classification`, `category-routing`, `full-triage-pipeline` (default: `all`) |

---

## 📜 License

MIT
