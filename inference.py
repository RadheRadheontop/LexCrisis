#!/usr/bin/env python3
"""
Baseline inference script for the LexCrisis Legal Environment.

Uses any OpenAI-compatible API to run an LLM agent against all 3 tasks.

Required environment variables:
    API_BASE_URL    – The API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME      – Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN or API_KEY – API key

Usage:
    export HF_TOKEN=your_token
    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
    python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import requests
from openai import OpenAI


# ── Configuration ───────────────────────────────────────────────────────────

ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY: str = os.environ.get("HF_TOKEN", os.environ.get("API_KEY", ""))

TASK_IDS: List[str] = ["task_1", "task_2", "task_3"]
MAX_LLM_RETRIES: int = 3


# ── System Prompts ──────────────────────────────────────────────────────────

SYSTEM_PROMPTS: Dict[str, str] = {
    "task_1": (
        "You are a senior associate at a law firm screening clients for conflicts of interest "
        "in a pharmaceutical product liability case (NovaChem/Veridex crisis).\n\n"
        "You must:\n"
        "1. Review each client using review_client(client_id)\n"
        "2. Check for conflicts between client pairs using check_conflict(client_a, client_b)\n"
        "3. Cite BCI rules using cite_rule(client_a, client_b, rule)\n"
        "4. Accept or decline each client using accept_client or decline_client\n"
        "5. Submit final decisions using submit_intake()\n\n"
        "Key BCI Rules (Part VI, Chapter II, Section II):\n"
        "- Rule 22: Not appear for opposite party if advised other side.\n"
        "- Rule 33: Not act for person whose interest is adverse to advised party.\n\n"
        'Respond with ONLY a JSON action: {"action_type": "...", "parameters": {...}}'
    ),
    "task_2": (
        "You are reviewing litigation discovery documents for privilege classification "
        "in the NovaChem/Veridex product liability case.\n\n"
        "You must:\n"
        "1. Review each document using review_document(doc_id)\n"
        "2. Classify privilege using classify_privilege(doc_id, classification, doctrine)\n"
        "   Classifications: attorney_client, work_product, both, none, waived\n"
        "3. Identify waivers using identify_waiver(doc_id, waiver_type, explanation)\n"
        "4. Identify exceptions using identify_exception(doc_id, exception_type, explanation)\n"
        "5. Recommend actions using recommend_action(doc_id, action, reasoning)\n"
        "   Actions: withhold, produce, clawback, redact\n"
        "6. Submit using submit_review()\n\n"
        "Key doctrines: IEA Sec 126 (professional communications), IEA Sec 129 (legal adviser), "
        "crime-fraud exception (Proviso 1), at-issue waiver.\n\n"
        'Respond with ONLY a JSON action: {"action_type": "...", "parameters": {...}}'
    ),
    "task_3": (
        "You are managing a multi-front legal crisis in NovaChem/Veridex litigation. "
        "5 crisis events need attention with varying deadlines.\n\n"
        "You must:\n"
        "1. Review events using review_event(event_id)\n"
        "2. Issue litigation hold using issue_litigation_hold(scope, custodians)\n"
        "3. File motions using file_motion(motion_type, court, arguments)\n"
        "4. Respond to discovery using respond_discovery(request_id, response_type, objections)\n"
        "   response_type: object, produce, partial_produce, privilege_log\n"
        "5. Assess experts using assess_expert(expert_id, qualification)\n"
        "6. Flag adversarial elements using flag_adversarial(item_id, threat_type, explanation)\n"
        "7. Flag ethical issues using flag_ethical_issue(issue_type, affected_clients, resolution)\n"
        "8. Submit using submit_triage()\n\n"
        "WARNING: Contains adversarial traps and ethical dilemmas. Do NOT blindly produce privileged documents.\n\n"
        'Respond with ONLY a JSON action: {"action_type": "...", "parameters": {...}}'
    ),
}


# ── Environment Client ──────────────────────────────────────────────────────


class EnvClient:
    """HTTP client for the LexCrisis environment API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def reset(self, task_id: str) -> Dict[str, Any]:
        """Reset the environment for a new episode."""
        resp = self.session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def step(
        self, action_type: str, parameters: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Execute one step."""
        resp = self.session.post(
            f"{self.base_url}/step",
            json={"action_type": action_type, "parameters": parameters or {}},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        """Get the current episode state."""
        resp = self.session.get(f"{self.base_url}/state", timeout=10)
        resp.raise_for_status()
        return resp.json()


# ── LLM Agent ──────────────────────────────────────────────────────────────


def format_observation(obs: Dict[str, Any]) -> str:
    """Format an observation dict into a prompt for the LLM."""
    parts: List[str] = [f"Task: {obs.get('task_description', '')[:500]}"]
    parts.append(f"\nStep {obs.get('step_count', 0)}/{obs.get('max_steps', 0)}")

    docs = obs.get("documents", [])
    if docs:
        parts.append(f"\nAvailable items ({len(docs)}):")
        for d in docs[:10]:
            parts.append(
                f"  [{d.get('index', '?')}] {d.get('title', '')} ({d.get('doc_type', '')})"
            )

    deadlines = obs.get("active_deadlines", [])
    if deadlines:
        parts.append("\nACTIVE DEADLINES:")
        for dl in deadlines:
            parts.append(
                f"  {dl.get('deadline_id', '')}: {dl.get('steps_remaining', '?')} steps remaining"
            )

    alerts = obs.get("ethical_alerts", [])
    if alerts:
        parts.append("\nETHICAL ALERTS:")
        for alert in alerts:
            parts.append(f"  {alert}")

    feedback = obs.get("feedback", "")
    if feedback:
        parts.append(f"\nFeedback:\n{feedback[:2000]}")

    findings = obs.get("findings", {})
    non_empty = {k: v for k, v in findings.items() if v and v != [] and v != {}}
    if non_empty:
        parts.append(
            f"\nFindings: {json.dumps(non_empty, indent=2, default=str)[:1000]}"
        )

    parts.append(f"\nAvailable actions: {obs.get('available_actions', [])}")
    parts.append("\nRespond with ONLY a JSON action object.")
    return "\n".join(parts)


def parse_llm_response(content: str) -> Dict[str, Any]:
    """Parse the LLM response, extracting a JSON action."""
    content = content.strip()

    # Strip markdown code fences
    if content.startswith("```"):
        lines = content.split("\n")
        end_idx = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        content = "\n".join(lines[1:end_idx]).strip()

    # Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to find embedded JSON
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        idx = content.find(start_char)
        if idx >= 0:
            depth = 0
            for i in range(idx, len(content)):
                if content[i] == start_char:
                    depth += 1
                elif content[i] == end_char:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(content[idx : i + 1])
                        except json.JSONDecodeError:
                            break

    # Fallback: noop
    return {"action_type": "noop", "parameters": {}}


def get_submit_action(task_id: str) -> str:
    """Get the terminal submit action for a task."""
    return {
        "task_1": "submit_intake",
        "task_2": "submit_review",
        "task_3": "submit_triage",
    }.get(task_id, "submit_intake")


def run_llm_agent(
    client: OpenAI, env: EnvClient, task_id: str, model: str
) -> Dict[str, Any]:
    """Run the LLM agent on a single task. Returns dict with success, steps, rewards list."""

    reset_data = env.reset(task_id)
    obs = reset_data.get("observation", reset_data)

    system_prompt = SYSTEM_PROMPTS.get(task_id, SYSTEM_PROMPTS["task_1"])
    history: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    done = False
    step_num = 0
    rewards_list: List[float] = []
    last_error: str | None = None
    success = False

    while not done:
        step_num += 1
        obs_text = format_observation(obs)
        history.append({"role": "user", "content": obs_text})

        action_dict: Dict[str, Any] = {"action_type": "noop", "parameters": {}}

        for attempt in range(MAX_LLM_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=history,
                    temperature=0.1,
                    max_tokens=512,
                )
                content = response.choices[0].message.content or ""
                action_dict = parse_llm_response(content)
                history.append({"role": "assistant", "content": content})
                last_error = None
                break
            except Exception as e:
                last_error = str(e)[:200]
                if attempt == MAX_LLM_RETRIES - 1:
                    action_dict = {
                        "action_type": get_submit_action(task_id),
                        "parameters": {},
                    }
                time.sleep(1.5 * (attempt + 1))

        action_type = action_dict.get("action_type", get_submit_action(task_id))
        parameters = action_dict.get("parameters", {})

        try:
            step_data = env.step(action_type, parameters)
            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", False)
            obs = step_data.get("observation", step_data)
            rewards_list.append(round(reward, 2))
            error_str = last_error if last_error else "null"
            print(
                f"[STEP] step={step_num} action={action_type} reward={reward:.2f} done={str(done).lower()} error={error_str}"
            )
        except Exception as e:
            last_error = str(e)[:200]
            rewards_list.append(0.0)
            print(
                f"[STEP] step={step_num} action={action_type} reward=0.00 done=false error={last_error}"
            )
            # Try to recover with submit
            try:
                submit_action = get_submit_action(task_id)
                step_data = env.step(submit_action, {})
                done = step_data.get("done", True)
                obs = step_data.get("observation", step_data)
            except Exception:
                done = True

        # Keep history manageable
        if len(history) > 20:
            history = history[:1] + history[-18:]

    # Determine success based on final score
    try:
        final_state = env.state()
        score = final_state.get("reward", 0.0)
        info = final_state.get("info", {})
        final_score = info.get("score", 0.0)
        success = final_score > 0.0
    except Exception:
        success = False
        final_score = 0.0

    return {
        "success": success,
        "steps": step_num,
        "rewards": rewards_list,
        "score": final_score,
    }


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point: validate env vars, connect to LLM + environment, run all tasks."""
    if not API_BASE_URL:
        print("ERROR: Set API_BASE_URL (e.g. https://router.huggingface.co/v1)")
        sys.exit(1)
    if not MODEL_NAME:
        print("ERROR: Set MODEL_NAME (e.g. Qwen/Qwen2.5-72B-Instruct)")
        sys.exit(1)
    if not API_KEY:
        print("ERROR: Set HF_TOKEN or API_KEY")
        sys.exit(1)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_client = EnvClient(ENV_BASE_URL)

    # Health check
    try:
        resp = requests.get(f"{ENV_BASE_URL}/health", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach {ENV_BASE_URL}: {e}")
        sys.exit(1)

    for task_id in TASK_IDS:
        try:
            result = run_llm_agent(llm_client, env_client, task_id, MODEL_NAME)
            rewards_str = ",".join(f"{r:.2f}" for r in result["rewards"])
            print(
                f"[END] success={str(result['success']).lower()} steps={result['steps']} rewards=[{rewards_str}]"
            )
        except Exception as e:
            print(f"[END] success=false steps=0 rewards=[0.00]")


if __name__ == "__main__":
    main()
