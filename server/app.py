"""FastAPI server for the LexCrisis OpenEnv benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from lexcrisis_env.env import BENCHMARK_NAME, BENCHMARK_VERSION, LexCrisisEnvironment
from lexcrisis_env.models import (
    Action,
    EnvironmentState,
    MetadataResponse,
    Observation,
    ResetRequest,
    ResetResponse,
    StepResponse,
)
from lexcrisis_env.tasks import SCRIPTED_BASELINES, TASK_DEFINITIONS

APP_DESCRIPTION = (
    "LexCrisis is a law-focused environment for legal operations incident response in "
    "high-stakes product-liability litigation. The current case study is pharmaceutical, "
    "but the workflow generalizes to regulated-industry litigation more broadly. It "
    "evaluates conflict screening, privilege review, and crisis triage under deadline pressure."
)

UI_PATH = Path(__file__).with_name("ui.html")
ENVIRONMENT = LexCrisisEnvironment()

app = FastAPI(
    title="LexCrisis OpenEnv API",
    version=BENCHMARK_VERSION,
    description=APP_DESCRIPTION,
)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    response = FileResponse(UI_PATH)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> MetadataResponse:
    return MetadataResponse(
        name="LexCrisis",
        description=APP_DESCRIPTION,
        version=BENCHMARK_VERSION,
        benchmark=BENCHMARK_NAME,
        domain="legal-operations",
        tags=["openenv", "law", "litigation", "legal-ops", "privilege-review"],
    )


@app.get("/schema")
def schema() -> Dict[str, Any]:
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": EnvironmentState.model_json_schema(),
    }


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "task_id": task.task_id,
                "name": task.name,
                "difficulty": task.difficulty,
                "description": task.description,
                "max_steps": task.max_steps,
                "baseline_steps": len(SCRIPTED_BASELINES[task.task_id]),
            }
            for task in TASK_DEFINITIONS.values()
        ]
    }


@app.get("/baselines")
def baselines() -> Dict[str, Any]:
    return {"baselines": SCRIPTED_BASELINES}


@app.get("/episode")
def episode() -> Dict[str, Any]:
    return ENVIRONMENT.episode_info()


@app.post("/reset")
async def reset(request: Request) -> JSONResponse:
    """Reset endpoint — accepts empty body, JSON body, or null body."""
    payload = ResetRequest()
    try:
        body = await request.body()
        if body:
            data = await request.json()
            if data:
                payload = ResetRequest.model_validate(data)
    except Exception:
        pass  # empty or malformed body → use all defaults
    observation = ENVIRONMENT.reset(
        task_id=payload.task_id,
        seed=payload.seed,
        episode_id=payload.episode_id,
    )
    result = ResetResponse(
        observation=observation,
        reward=0.0,
        done=False,
        info={"episode_id": ENVIRONMENT.episode_id, "task_id": observation.task_id},
    )
    return JSONResponse(content=result.model_dump())


def _extract_action(body: Dict[str, Any]) -> Action:
    if "action" in body and isinstance(body["action"], dict):
        return Action.model_validate(body["action"])
    return Action.model_validate(body)


@app.post("/step")
async def step(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"detail": "Invalid JSON body"})

    try:
        action = _extract_action(body)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    observation, reward, done, info = ENVIRONMENT.step(action)
    result = StepResponse(observation=observation, reward=reward, done=done, info=info)
    return JSONResponse(content=result.model_dump())


@app.get("/state")
def state() -> EnvironmentState:
    return ENVIRONMENT.state()


def _jsonrpc_success(result: Any, request_id: Any = None) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(message: str, request_id: Any = None, code: int = -32600) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


@app.post("/mcp")
async def mcp(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        return _jsonrpc_error("Invalid JSON", code=-32700)

    request_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params", {})

    if method in (None, ""):
        return _jsonrpc_error("Missing method", request_id=request_id)

    if method == "tools/list":
        return _jsonrpc_success(
            {
                "tools": [
                    {
                        "name": "reset",
                        "description": "Reset the environment to a task.",
                        "inputSchema": ResetRequest.model_json_schema(),
                    },
                    {
                        "name": "step",
                        "description": "Execute one action in the active episode.",
                        "inputSchema": Action.model_json_schema(),
                    },
                    {
                        "name": "state",
                        "description": "Return the current observation, last reward, and done flag.",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ]
            },
            request_id=request_id,
        )

    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        try:
            if tool_name == "reset":
                request_model = ResetRequest.model_validate(arguments)
                observation = ENVIRONMENT.reset(**request_model.model_dump())
                result = ResetResponse(
                    observation=observation,
                    reward=0.0,
                    done=False,
                    info={"episode_id": ENVIRONMENT.episode_id, "task_id": observation.task_id},
                )
                return _jsonrpc_success(result.model_dump(), request_id=request_id)
            if tool_name == "step":
                observation, reward, done, info = ENVIRONMENT.step(Action.model_validate(arguments))
                result = StepResponse(observation=observation, reward=reward, done=done, info=info)
                return _jsonrpc_success(result.model_dump(), request_id=request_id)
            if tool_name == "state":
                return _jsonrpc_success(ENVIRONMENT.state().model_dump(), request_id=request_id)
        except ValidationError as exc:
            return _jsonrpc_error(json.dumps(exc.errors()), request_id=request_id, code=-32602)
        return _jsonrpc_error(f"Unknown tool: {tool_name}", request_id=request_id, code=-32601)

    return _jsonrpc_error(f"Unsupported method: {method}", request_id=request_id, code=-32601)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
