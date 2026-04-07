---
title: LexCrisis
emoji: ⚖️
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - law
  - litigation
  - legal-ops
---

# LexCrisis

LexCrisis is an OpenEnv benchmark for **legal operations incident response** in a live
product-liability crisis. Instead of asking an agent to answer static legal questions,
it asks the agent to do the kind of work a law firm conflicts team, litigation support
team, or in-house legal operations group actually does:

- screen incoming clients without creating ethics conflicts
- review documents without waiving privilege
- coordinate litigation response under deadline pressure

That makes the domain both practical and novel: it is a real legal workflow, but the
environment adds **time pressure, partial information, adversarial discovery traps, and
workflow trade-offs** that typical legal benchmarks do not model.

## Why this domain is strong

- **Real-world utility:** firms and legal departments care about conflicts, privilege,
  discovery, and crisis coordination because mistakes cost money, sanctions, and
  strategic leverage.
- **Novelty:** this is not generic document analysis. It is a legal-ops control-room
  simulation with sequencing pressure and irreversible mistakes.
- **Easy to explain:** "AI legal operations agent for litigation incident response" is
  immediately understandable without hackathon framing.

## Tasks

### `task_1` - Conflict-Safe Client Intake (Easy)

The agent screens prospective clients in a pharmaceutical litigation wave.

Goal:
- identify conflict pairs
- cite the correct Bar Council of India rule
- make correct accept or decline decisions

### `task_2` - Privilege Review Under Litigation Pressure (Medium)

The agent reviews eight litigation documents before production.

Goal:
- classify privilege status correctly
- detect waiver and exception scenarios
- recommend the right production action

### `task_3` - Litigation Incident Command (Hard)

The agent runs a legal incident-response workflow during a live pharma litigation crisis.

Goal:
- issue a litigation hold before preservation failure
- respond to emergency motion pressure
- avoid privilege waiver in discovery
- surface a former-client ethics conflict
- make coordinated, well-prioritized decisions

## Action Space

The environment uses a typed `Action` model with an `action_type` plus action-specific
`parameters`.

Key actions include:
- `review_client`, `check_conflict`, `cite_rule`, `accept_client`, `decline_client`
- `review_document`, `classify_privilege`, `identify_waiver`, `identify_exception`, `recommend_action`
- `review_event`, `issue_litigation_hold`, `file_motion`, `respond_discovery`, `assess_expert`, `flag_adversarial`, `flag_ethical_issue`

## Observation Space

Each `Observation` includes:
- `task_id`, `task_name`, `difficulty`, `task_description`
- `documents`: selectable clients, documents, or crisis events
- `current_content`: the last reviewed item
- `available_actions`
- `findings`: accumulated work product
- `feedback`
- `step_count`, `max_steps`
- `active_deadlines`
- `ethical_alerts`

## Reward Design

Step reward is dense and non-cumulative:

```text
reward = grader_score_delta + milestone_bonus + penalty
```

Where:
- `grader_score_delta` rewards actual progress toward the task objective
- `milestone_bonus` rewards productive intermediate work such as reviewing new items
- `penalty` discourages loops, bad privilege calls, late actions, and off-task behavior

`state()` returns exactly:

```json
{
  "observation": { "...": "..." },
  "reward": 0.0,
  "done": false
}
```

The `reward` field is always the **last step reward only**, never accumulated reward.

## Deterministic Graders

All graders are deterministic and bounded in `[0.0, 1.0]`.

- Task 1: conflict identification, rule citation, intake decision quality
- Task 2: privilege classification, doctrine quality, waiver detection, exception handling, recommendation quality
- Task 3: deadline compliance, ethical handling, adversarial detection, discovery response quality, expert assessment, action ordering

## Setup

### Local

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

Then open `http://127.0.0.1:7860`.

Quick workflow in the UI:

1. Select a task on the left.
2. Click `Reset Task`.
3. Start with the default review action already placed in the action box.
4. Edit the JSON action and press `Send Action`.
5. Watch `Last Reward`, `Score`, `Done`, and `Agent Findings` on the right.
6. Finish with the task submit action:
   - `submit_intake`
   - `submit_review`
   - `submit_triage`

### Validate

```bash
openenv validate
```

### Docker

```bash
docker build -t lexcrisis .
docker run -p 7860:7860 lexcrisis
```

## Inference

`inference.py` is the baseline agent runner. It executes all three tasks, queries the
LLM via the provided proxy at every step, and emits structured logs in the required
format.

**All LLM calls go through the evaluator-provided LiteLLM proxy.** The script reads
`API_BASE_URL` and `API_KEY` from environment variables and will exit immediately if
either is missing.

Environment variables (injected by the evaluator):

- `API_BASE_URL` – LiteLLM proxy base URL (**required**, no default)
- `API_KEY` – LiteLLM proxy API key (**required**, no default)
- `MODEL_NAME` – model to request (default: `Qwen/Qwen2.5-72B-Instruct`)

Run:

```bash
export API_BASE_URL=https://your-proxy-url/v1
export API_KEY=your_key_here
python inference.py
```

Windows PowerShell:

```powershell
$env:API_BASE_URL="https://your-proxy-url/v1"
$env:API_KEY="your_key_here"
python inference.py
```

### API Keys And GitHub Safety

- Put keys in environment variables locally.
- Put them in Hugging Face Space secrets when deploying.
- Never commit `.env` or token values to GitHub.
- `.gitignore` already excludes `.env`, `.env.local`, `*.key`, and `*.pem`.

## Baseline Scores

These are filled from the deterministic reference runner:

- `task_1`: `0.5400`
- `task_2`: `0.8438`
- `task_3`: `0.9150`

## Project Layout

```text
lexcrisis/
├── Dockerfile
├── openenv.yaml
├── inference.py
├── main.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── my_env/
│   ├── __init__.py
│   ├── env.py
│   ├── tasks.py
│   ├── graders.py
│   └── models.py
├── server/
│   ├── __init__.py
│   ├── app.py
│   └── ui.html
└── lexcrisis_env/
    ├── __init__.py
    ├── env.py
    ├── tasks.py
    ├── graders.py
    └── models.py
```
