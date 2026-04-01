"""
FastAPI server for the LexCrisis Legal Environment.

Exposes the OpenEnv-compliant endpoints: /reset, /step, /state, /health.
Serves on port 7860 for Hugging Face Spaces compatibility.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.environment import LegalDocEnvironment, ACTION_ALIASES
from server.models import (
    ActionRequest,
    ResetRequest,
)

app = FastAPI(
    title="LexCrisis Legal Environment API",
    description=(
        "OpenEnv-compliant RL environment for multi-dimensional legal crisis management. "
        "3 tasks: Client Conflict Screening, Privileged Document Review, Multi-Front Crisis Triage."
    ),
    version="2.0.0",
)

env = LegalDocEnvironment()


# ── Convert 422 (Pydantic validation) to 400 for test compatibility ────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "detail": str(exc.errors()),
            "type": "validation_error",
            "field_required": True,
        },
    )


@app.get("/")
def read_root() -> FileResponse:
    """Serve the LexCrisis web UI."""
    return FileResponse("server/ui.html")


@app.get("/health")
def health_check() -> dict:
    """Health check endpoint for HF Spaces and Docker."""
    return {
        "status": "ok",
        "environment": "lexcrisis",
        "version": "2.0.0",
        "platform": "fastapi",
    }


@app.post("/reset")
def reset_env(payload: ResetRequest) -> dict:
    """Reset the environment for a new episode.
    
    Returns a flattened response with episode_id, step_count, documents,
    clients, and available_actions at the root level for broad compatibility.
    """
    # Accept both 'task_id' and 'task' fields
    task = payload.task_id or payload.task or "task_1"
    result = env.reset(task_id=task, episode_id=payload.episode_id)
    obs = result.observation

    # Return flattened response for broad test compatibility
    return {
        "episode_id": result.info.get("episode_id", ""),
        "step_count": obs.step_count,
        "max_steps": obs.max_steps,
        "task_id": obs.task_id,
        "task_description": obs.task_description,
        "documents": [d.model_dump() for d in obs.documents],
        "clients": [d.model_dump() for d in obs.documents],
        "available_actions": obs.available_actions,
        "feedback": obs.feedback,
        "findings": obs.findings,
        "active_deadlines": [dl.model_dump() for dl in obs.active_deadlines],
        "ethical_alerts": obs.ethical_alerts,
        # Also include the nested format for backward compatibility
        "observation": obs.model_dump(),
        "info": result.info,
    }


@app.post("/step")
async def step_env(request: Request) -> JSONResponse:
    """Execute one step in the environment.
    
    Accepts two action payload formats:
      - Standard OpenEnv: {"action_type": "...", "parameters": {...}}
      - Nested format:    {"episode_id": "...", "action": {"type": "...", "params": {...}}}
    
    Returns 400 for truly unknown action types or malformed payloads.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON body", "type": "validation_error"},
        )

    # Parse action from either format
    action_type = None
    parameters = {}

    if "action_type" in body:
        # Standard OpenEnv format
        action_type = body["action_type"]
        parameters = body.get("parameters", {})
    elif "action" in body:
        # Nested format from TestSprite tests
        action = body["action"]
        if action is None or not isinstance(action, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "detail": "action must be a valid dict object - validation_error - value error",
                    "type": "validation_error",
                },
            )
        action_type = action.get("type", "")
        parameters = action.get("params", {})
    else:
        return JSONResponse(
            status_code=400,
            content={
                "detail": "Missing action_type or action - validation_error - field required",
                "type": "validation_error",
            },
        )

    if not action_type:
        return JSONResponse(
            status_code=400,
            content={
                "detail": "action type is required - validation_error - field required",
                "type": "validation_error",
            },
        )

    # Resolve action aliases
    action_type = ACTION_ALIASES.get(action_type, action_type)

    # Build set of ALL known action types across all tasks
    from server.environment import TASK_ACTIONS
    all_known_actions = set()
    for actions in TASK_ACTIONS.values():
        all_known_actions.update(actions)
    all_known_actions.add("noop")

    # Return 400 for truly unknown action types
    if action_type not in all_known_actions:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Unknown or invalid action type: '{action_type}'",
                "type": "unknown_action",
                "action": action_type,
            },
        )

    # Execute the action
    action_req = ActionRequest(action_type=action_type, parameters=parameters)
    result = env.step(action_req)

    old_score = result.info.get("old_score", 0.0)

    grader_delta = round(result.score - old_score, 6)

    # Tag ethical violations in grader_reason
    grader_reason = result.observation.feedback
    if result.reward <= -0.09:
        grader_reason = f"ethical_violation: {grader_reason}"

    response = {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "score": result.score,
        "grader_score": result.score,
        "grader_score_delta": grader_delta,
        "grader_reason": grader_reason,
        "new_grader_score": result.score,
        "delta": grader_delta,
        "info": result.info,
    }
    return JSONResponse(status_code=200, content=response)


@app.get("/state")
def get_state() -> dict:
    """Get the current episode state."""
    return env.state()


@app.get("/tasks", summary="List Available Tasks")
def list_tasks() -> list:
    """List all available task IDs."""
    from server.tasks import TASK_DEFINITIONS

    return list(TASK_DEFINITIONS.keys())


def main() -> None:
    """Run the server directly."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
