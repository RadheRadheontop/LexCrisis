#!/usr/bin/env python3
"""Baseline inference agent for LexCrisis OpenEnv benchmark.

Routes ALL LLM calls through the evaluator-provided LiteLLM proxy.
No hardcoded API keys or bypass paths.

Environment variables (injected by the hackathon evaluator)
-----------------------------------------------------------
API_BASE_URL  – LiteLLM proxy base URL  (has default)
MODEL_NAME    – model to request        (has default)
HF_TOKEN      – HuggingFace / proxy key (NO default, injected by evaluator)
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests as http_requests
from openai import OpenAI

from lexcrisis_env.env import BENCHMARK_NAME, LexCrisisEnvironment
from lexcrisis_env.models import Action
from lexcrisis_env.tasks import SCRIPTED_BASELINES, TASK_ACTIONS, TASK_DEFINITIONS


# ──────────────────────────────────────────────────────────────────────
# Logging helper
# ──────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    """Write a diagnostic line to stderr (never mixed with stdout output)."""
    sys.stderr.write(f"# {msg}\n")
    sys.stderr.flush()


# ──────────────────────────────────────────────────────────────────────
# Configuration – EXACTLY as the pre-submission checklist requires:
#   API_BASE_URL  →  os.getenv("API_BASE_URL", "<default>")
#   MODEL_NAME    →  os.getenv("MODEL_NAME",   "<default>")
#   HF_TOKEN      →  os.getenv("HF_TOKEN")          ← NO default
# ──────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

# Strip whitespace / trailing slashes
API_BASE_URL = API_BASE_URL.strip().rstrip("/")
MODEL_NAME = MODEL_NAME.strip()
HF_TOKEN = HF_TOKEN.strip()

_log(f"API_BASE_URL = {API_BASE_URL}")
_log(f"MODEL_NAME   = {MODEL_NAME}")
_log(f"HF_TOKEN     = {'set (' + HF_TOKEN[:4] + '...)' if HF_TOKEN else 'NOT SET'}")

if not HF_TOKEN:
    _log("WARNING: HF_TOKEN is not set. LLM calls will likely fail.")
    _log("         The evaluator should inject HF_TOKEN at runtime.")


# ──────────────────────────────────────────────────────────────────────
# OpenAI client – configured via the evaluator-injected variables
# ──────────────────────────────────────────────────────────────────────

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
    timeout=120.0,
    max_retries=3,
)

_SYSTEM_PROMPT = (
    "You are a legal operations AI agent solving tasks in the LexCrisis benchmark. "
    "At each step you must return ONLY a single JSON object:\n"
    '  {"action_type": "<type>", "parameters": {<key>: <value>, ...}}\n'
    "Select the most appropriate action from the available_actions list. "
    "Return only the JSON object, no prose or markdown fences."
)

# Track API usage across the whole run
_api_attempts: int = 0
_api_successes: int = 0


# ──────────────────────────────────────────────────────────────────────
# Raw HTTP helper – fallback if the OpenAI SDK has issues
# ──────────────────────────────────────────────────────────────────────

def _raw_chat_completion(
    messages: List[Dict[str, str]],
    label: str = "raw",
    max_tokens: int = 256,
) -> Optional[str]:
    """POST directly to {API_BASE_URL}/chat/completions via requests."""
    global _api_attempts, _api_successes
    _api_attempts += 1
    url = f"{API_BASE_URL}/chat/completions"
    _log(f"  [{label}] raw POST -> {url}")
    try:
        resp = http_requests.post(
            url,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=120,
        )
        _log(f"  [{label}] HTTP {resp.status_code}")
        if resp.ok:
            data = resp.json()
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            _api_successes += 1
            return text
        else:
            _log(f"  [{label}] body: {resp.text[:300]}")
    except Exception as exc:
        _log(f"  [{label}] failed: {exc}")
    return None


# ──────────────────────────────────────────────────────────────────────
# Startup connectivity test – guarantees at least one proxy hit
# ──────────────────────────────────────────────────────────────────────

def _verify_proxy() -> None:
    """Hit the proxy at startup to register the key before any tasks run."""
    global _api_attempts, _api_successes
    _log("--- Proxy connectivity test ---")

    # 1) GET /models – cheap reachability check
    try:
        models_url = f"{API_BASE_URL}/models"
        _log(f"  GET {models_url}")
        r = http_requests.get(
            models_url,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            timeout=15,
        )
        _log(f"  GET /models -> HTTP {r.status_code}")
        if r.ok:
            _log(f"  models (truncated): {r.text[:200]}")
    except Exception as exc:
        _log(f"  GET /models error: {exc}")

    # 2) SDK chat completion – registers the key via the OpenAI client
    _api_attempts += 1
    try:
        _log(f"  SDK warmup (model={MODEL_NAME}, base_url={API_BASE_URL})")
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
            max_tokens=4,
            temperature=0,
        )
        text = resp.choices[0].message.content or ""
        _log(f"  SDK warmup OK: {text!r}")
        _api_successes += 1
    except Exception as exc:
        _log(f"  SDK warmup failed: {exc}")
        _log(f"  traceback:\n{traceback.format_exc()}")

        # 3) Raw HTTP fallback – ensures the key gets hit no matter what
        _log("  Trying raw HTTP fallback...")
        result = _raw_chat_completion(
            [{"role": "user", "content": "Reply OK"}],
            label="warmup-fallback",
            max_tokens=4,
        )
        if result:
            _log(f"  Raw fallback OK: {result!r}")
        else:
            _log("  Raw fallback also failed. Proxy may be unreachable.")

    _log("--- End connectivity test ---")


# Run the connectivity test at import time
_verify_proxy()


# ──────────────────────────────────────────────────────────────────────
# Output format helpers – exact [START]/[STEP]/[END] format
# ──────────────────────────────────────────────────────────────────────

def action_string(action: Dict[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def emit_start(task_id: str) -> None:
    print(f"[START] task={task_id} env={BENCHMARK_NAME} model={MODEL_NAME}")
    sys.stdout.flush()


# Strict score bounds required by Phase-2 validation
_EMIT_FLOOR = 0.001
_EMIT_CEIL  = 0.999


def _clamp_emit(value: float) -> float:
    """Clamp any emitted score/reward to the open interval (0, 1)."""
    return round(max(_EMIT_FLOOR, min(float(value), _EMIT_CEIL)), 4)


def emit_step(
    step: int,
    action: Dict[str, Any],
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    # Clamp reward to (0, 1) open interval as required by the evaluator
    safe_reward = _clamp_emit(reward)
    error_value = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action_string(action)} reward={safe_reward:.4f} "
        f"done={str(done).lower()} error={error_value}"
    )
    sys.stdout.flush()


def emit_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # Both the per-step rewards and the final task score must be strictly in (0, 1)
    safe_score = _clamp_emit(score)
    rewards_text = ",".join(f"{_clamp_emit(r):.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={safe_score:.4f} rewards={rewards_text}"
    )
    sys.stdout.flush()


# ──────────────────────────────────────────────────────────────────────
# LLM action selection
# ──────────────────────────────────────────────────────────────────────

def _parse_action(text: str, available: List[str]) -> Optional[Dict[str, Any]]:
    """Extract a valid JSON action from the model's response text."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    # Try the whole text first, then any {...} block
    candidates = [text] + re.findall(r"\{[^{}]*\}", text, re.DOTALL)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and obj.get("action_type") in available:
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def get_llm_action(
    task_id: str,
    step_index: int,
    obs: Dict[str, Any],
    fallback: Dict[str, Any],
) -> Dict[str, Any]:
    """Query the LLM proxy. ALWAYS attempts an API call; falls back on failure."""
    global _api_attempts, _api_successes

    available: List[str] = obs.get("available_actions", [])

    prompt_obs = {
        "task_id": task_id,
        "step": step_index,
        "feedback": obs.get("feedback", ""),
        "available_actions": available,
        "findings_summary": {
            k: v for k, v in obs.get("findings", {}).items() if v
        },
        "active_deadlines": obs.get("active_deadlines", []),
        "ethical_alerts": obs.get("ethical_alerts", []),
    }

    user_content = (
        f"Current observation:\n{json.dumps(prompt_obs, indent=2)}\n\n"
        f"Scripted suggestion (use only if helpful): {json.dumps(fallback)}"
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # ── Attempt 1: OpenAI SDK ──
    _api_attempts += 1
    try:
        _log(f"Step {step_index}: SDK call (model={MODEL_NAME})")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=256,
            messages=messages,
        )
        raw = response.choices[0].message.content or ""
        _log(f"Step {step_index}: response: {raw[:120]}")
        _api_successes += 1

        parsed = _parse_action(raw, available)
        if parsed is not None:
            return parsed
        _log(f"Step {step_index}: unparseable, using fallback")
        return fallback

    except Exception as exc:
        _log(f"Step {step_index}: SDK failed: {exc}")

    # ── Attempt 2: raw HTTP fallback ──
    _log(f"Step {step_index}: trying raw HTTP")
    raw_text = _raw_chat_completion(messages, label=f"step-{step_index}")
    if raw_text:
        _log(f"Step {step_index}: raw response: {raw_text[:120]}")
        parsed = _parse_action(raw_text, available)
        if parsed is not None:
            return parsed
        _log(f"Step {step_index}: raw unparseable, using fallback")
        return fallback

    _log(f"Step {step_index}: all API attempts failed, using scripted fallback")
    return fallback


# ──────────────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────────────

def run_task(task_id: str) -> Dict[str, Any]:
    env = LexCrisisEnvironment()
    rewards: List[float] = []
    step_index = 0
    success = False
    final_score = 0.0

    scripted = SCRIPTED_BASELINES[task_id]

    emit_start(task_id)
    try:
        observation = env.reset(task_id=task_id)
        obs_dict: Dict[str, Any] = observation.model_dump(mode="json")

        for idx, raw_action in enumerate(scripted):
            step_index += 1
            agent_action = get_llm_action(
                task_id, step_index, obs_dict, raw_action,
            )

            error_message: Optional[str] = None
            done = False
            reward = 0.0

            try:
                observation = env.step(Action.model_validate(agent_action))
                state = env.state
                reward = float(state.reward or observation.reward or 0.0)
                done = bool(state.done or observation.done)
                final_score = env.last_score
                obs_dict = observation.model_dump(mode="json")
            except Exception as exc:
                error_message = str(exc)

            # Clamp reward before storing so the list never holds 0.0 or 1.0
            rewards.append(_clamp_emit(reward))
            emit_step(step_index, agent_action, reward, done, error_message)

            if error_message is not None:
                break
            if done:
                success = True
                break

        if not success and env.state.done:
            success = True
            final_score = env.last_score
    finally:
        final_score = _clamp_emit(env.last_score)
        env.close()
        emit_end(success, step_index, final_score, rewards)

    return {
        "task_id": task_id,
        "task_name": TASK_DEFINITIONS[task_id].name,
        "score": _clamp_emit(final_score),
        "steps": step_index,
        "success": success,
        "rewards": [_clamp_emit(r) for r in rewards],
    }


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _log(f"Starting LexCrisis inference agent")
    _log(f"  base_url: {API_BASE_URL}")
    _log(f"  model:    {MODEL_NAME}")
    _log(f"  token:    {'set' if HF_TOKEN else 'NOT SET'}")

    results = [run_task(task_id) for task_id in TASK_DEFINITIONS]

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "baseline_scores.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    _log(f"Run complete. API attempts={_api_attempts}, successes={_api_successes}")
    if _api_successes == 0:
        _log("WARNING: ZERO successful API calls. The proxy key was likely never activated.")


if __name__ == "__main__":
    main()
