"""
FastAPI server for the LexCrisis Legal Environment.

Exposes the OpenEnv-compliant endpoints: /reset, /step, /state, /health.
Serves on port 7860 for Hugging Face Spaces compatibility.
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.environment import LegalDocEnvironment
from server.models import (
    ActionRequest,
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepResponse,
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


@app.get("/")
def read_root() -> FileResponse:
    """Serve the LexCrisis web UI."""
    return FileResponse("server/ui.html")


@app.get("/health")
def health_check() -> dict:
    """Health check endpoint for HF Spaces and Docker."""
    return {"status": "healthy", "environment": "lexcrisis", "version": "2.0.0"}


@app.post(
    "/reset",
    response_model=ResetResponse,
    summary="Reset Environment",
    description="Initializes a new episode for the given task ID (task_1 through task_3).",
)
def reset_env(payload: ResetRequest) -> ResetResponse:
    """Reset the environment for a new episode."""
    return env.reset(task_id=payload.task_id, episode_id=payload.episode_id)


@app.post(
    "/step",
    response_model=StepResponse,
    summary="Execute Agent Action",
    description="Executes a discrete agent action and returns reward, observation, and done flag.",
)
def step_env(action: ActionRequest) -> StepResponse:
    """Execute one step in the environment."""
    return env.step(action)


@app.get(
    "/state",
    response_model=StateResponse,
    summary="Get Current State",
    description="Returns the current episode state without modifying it.",
)
def get_state() -> StateResponse:
    """Get the current episode state."""
    return env.state()


@app.get("/tasks", summary="List Available Tasks")
def list_tasks() -> dict:
    """List all available tasks with metadata."""
    from server.tasks import TASK_DEFINITIONS

    return {
        tid: {
            "name": td.name,
            "difficulty": td.difficulty,
            "max_steps": td.max_steps,
            "actions": td.relevant_actions,
        }
        for tid, td in TASK_DEFINITIONS.items()
    }


def main() -> None:
    """Run the server directly."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
