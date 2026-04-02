"""Typed models for the LexCrisis OpenEnv environment."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


ActionType = Literal[
    "review_client",
    "check_conflict",
    "cite_rule",
    "accept_client",
    "decline_client",
    "submit_intake",
    "review_document",
    "classify_privilege",
    "identify_waiver",
    "identify_exception",
    "recommend_action",
    "submit_review",
    "review_event",
    "issue_litigation_hold",
    "file_motion",
    "respond_discovery",
    "assess_expert",
    "flag_adversarial",
    "flag_ethical_issue",
    "submit_triage",
    "noop",
]

Difficulty = Literal["easy", "medium", "hard"]


class Action(BaseModel):
    """Agent action accepted by the environment."""

    model_config = ConfigDict(extra="forbid")

    action_type: ActionType = Field(description="Action verb to execute.")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific arguments.",
    )


class Reward(BaseModel):
    """Structured reward breakdown for the last step."""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(description="Final step reward.")
    score_delta: float = Field(description="Change in deterministic grader score.")
    milestone_bonus: float = Field(description="Dense bonus from productive progress.")
    penalty: float = Field(description="Negative shaping from bad or repeated actions.")
    reason: str = Field(description="Human-readable explanation for the reward.")


class DocumentSummary(BaseModel):
    """Compact description of a selectable item in the episode."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    title: str
    item_type: str
    category: str = ""


class DeadlineSummary(BaseModel):
    """Pending deadline shown to the agent."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    title: str
    steps_remaining: int
    consequence: str


class Observation(BaseModel):
    """Observation exposed to the agent after reset and each step."""

    model_config = ConfigDict(extra="forbid")

    benchmark: str = Field(default="lexcrisis")
    task_id: str
    task_name: str
    difficulty: Difficulty
    task_description: str
    documents: List[DocumentSummary] = Field(default_factory=list)
    current_content: Optional[str] = None
    available_actions: List[str] = Field(default_factory=list)
    findings: Dict[str, Any] = Field(default_factory=dict)
    feedback: str = ""
    step_count: int = 0
    max_steps: int = 0
    active_deadlines: List[DeadlineSummary] = Field(default_factory=list)
    ethical_alerts: List[str] = Field(default_factory=list)


class EnvironmentState(BaseModel):
    """State response required by the OpenEnv competition prompt."""

    model_config = ConfigDict(extra="forbid")

    observation: Observation
    reward: float = Field(description="Reward from the last step only.")
    done: bool


class ResetRequest(BaseModel):
    """Reset request accepted by the HTTP server."""

    model_config = ConfigDict(extra="allow")

    task_id: Optional[str] = Field(default=None)
    seed: Optional[int] = Field(default=None, ge=0)
    episode_id: Optional[str] = Field(default=None)


class ResetResponse(BaseModel):
    """HTTP reset response."""

    model_config = ConfigDict(extra="forbid")

    observation: Observation
    reward: float = Field(default=0.0)
    done: bool = Field(default=False)
    info: Dict[str, Any] = Field(default_factory=dict)


class StepResponse(BaseModel):
    """HTTP step response."""

    model_config = ConfigDict(extra="forbid")

    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class MetadataResponse(BaseModel):
    """Environment metadata response."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    version: str
    benchmark: str
    domain: str
    tags: List[str] = Field(default_factory=list)
