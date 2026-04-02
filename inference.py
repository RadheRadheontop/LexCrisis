#!/usr/bin/env python3
"""Deterministic baseline runner for LexCrisis."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from lexcrisis_env.env import BENCHMARK_NAME, LexCrisisEnvironment
from lexcrisis_env.models import Action
from lexcrisis_env.tasks import SCRIPTED_BASELINES, TASK_DEFINITIONS

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
USE_LLM_BASELINE = os.environ.get("USE_LLM_BASELINE", "0").lower() in {"1", "true", "yes"}


def build_client() -> Optional[OpenAI]:
    """Build the optional OpenAI client used for LLM-assisted variants."""

    if not API_KEY:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def maybe_call_llm(client: Optional[OpenAI], task_id: str, action: Dict[str, Any]) -> None:
    """Optional no-op extension point for LLM-assisted baselines.

    All LLM calls, when enabled, go through the OpenAI client. The default
    competition baseline is deterministic and does not require network access.
    """

    if client is None or not USE_LLM_BASELINE:
        return
    client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=32,
        messages=[
            {
                "role": "system",
                "content": "Acknowledge the action for benchmarking. Return a one-line summary.",
            },
            {
                "role": "user",
                "content": json.dumps({"task_id": task_id, "action": action}, sort_keys=True),
            },
        ],
    )


def action_string(action: Dict[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def emit_start(task_id: str) -> None:
    print(f"[START] task={task_id} env={BENCHMARK_NAME} model={MODEL_NAME}")


def emit_step(step: int, action: Dict[str, Any], reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action_string(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}"
    )


def emit_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_text}")


def run_task(task_id: str, client: Optional[OpenAI]) -> Dict[str, Any]:
    env = LexCrisisEnvironment()
    rewards: List[float] = []
    step_index = 0
    success = False
    final_score = 0.0

    emit_start(task_id)
    try:
        env.reset(task_id=task_id)
        for raw_action in SCRIPTED_BASELINES[task_id]:
            step_index += 1
            maybe_call_llm(client, task_id, raw_action)
            error_message: Optional[str] = None
            done = False
            reward = 0.0

            try:
                _, reward, done, _ = env.step(Action.model_validate(raw_action))
                reward = float(reward)
                done = bool(done)
                final_score = env.last_score
            except Exception as exc:  # pragma: no cover - defensive logging path
                error_message = str(exc)

            rewards.append(round(reward, 2))
            emit_step(step_index, raw_action, reward, done, error_message)

            if error_message is not None:
                success = False
                break
            if done:
                success = True
                break

        if not success and env.state().done:
            success = True
            final_score = env.last_score
    finally:
        final_score = env.last_score
        env.close()
        emit_end(success, step_index, rewards)

    return {
        "task_id": task_id,
        "task_name": TASK_DEFINITIONS[task_id].name,
        "score": final_score,
        "steps": step_index,
        "success": success,
        "rewards": rewards,
    }


def main() -> None:
    client = build_client()
    results = [run_task(task_id, client) for task_id in TASK_DEFINITIONS]

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "baseline_scores.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
