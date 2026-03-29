"""
Pydantic models for the LexCrisis Legal Environment.

Defines typed Action, Observation, Reward, and State models per the OpenEnv specification.
All models use explicit type hints and Field descriptions for API documentation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request Models ──────────────────────────────────────────────────────────


class ResetRequest(BaseModel):
    """Body for the /reset endpoint."""

    task_id: str = Field(
        default="task_1",
        description=(
            "Task to load: task_1 (Easy: Conflict Screening), "
            "task_2 (Medium: Privilege Review), "
            "task_3 (Hard: Crisis Triage)"
        ),
    )
    seed: Optional[int] = Field(default=None, description="Optional random seed (reserved for future use)")
    episode_id: Optional[str] = Field(default=None, description="Optional episode ID override")


class ActionRequest(BaseModel):
    """Action submitted by the agent at each step.

    Supported action_type values per task:

    Task 1 — Conflict Screening:
        review_client       — Review client intake (params: client_id)
        check_conflict      — Check conflict between clients (params: client_a, client_b)
        cite_rule           — Cite BCI rule for conflict (params: client_a, client_b, rule)
        accept_client       — Accept client (params: client_id, justification)
        decline_client      — Decline client (params: client_id, reason)
        submit_intake       — Submit final intake decisions

    Task 2 — Privilege Review:
        review_document     — Read document (params: doc_id)
        classify_privilege  — Classify privilege (params: doc_id, classification, doctrine)
        identify_waiver     — Flag waiver event (params: doc_id, waiver_type, explanation)
        identify_exception  — Flag exception (params: doc_id, exception_type, explanation)
        recommend_action    — Recommend action (params: doc_id, action, reasoning)
        submit_review       — Submit final review

    Task 3 — Crisis Triage:
        review_event             — Read crisis event (params: event_id)
        issue_litigation_hold    — Issue preservation notice (params: scope, custodians)
        file_motion              — File court motion (params: motion_type, court, arguments)
        respond_discovery        — Respond to discovery (params: request_id, response_type, objections)
        assess_expert            — Evaluate expert (params: expert_id, qualification)
        flag_adversarial         — Flag adversarial element (params: item_id, threat_type, explanation)
        flag_ethical_issue       — Flag ethical issue (params: issue_type, affected_clients, resolution)
        submit_triage            — Submit final triage report
    """

    action_type: str = Field(..., description="Type of legal action to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action-specific parameters")


# ── Observation Models ──────────────────────────────────────────────────────


class DocumentInfo(BaseModel):
    """Summary info for an item available in the episode (client, document, or event)."""

    index: int = Field(description="Zero-based index")
    title: str = Field(description="Display title")
    doc_type: str = Field(description="Item type (client_intake, legal_document, crisis_event)")
    category: str = Field(default="", description="Sub-category for filtering")


class DeadlineInfo(BaseModel):
    """An active deadline in the crisis triage task."""

    deadline_id: str = Field(description="Unique deadline identifier")
    description: str = Field(description="What must be done")
    steps_remaining: int = Field(description="Steps until deadline expires")
    consequence: str = Field(description="What happens if missed")


class ObservationResponse(BaseModel):
    """The observation space returned to the agent after each action."""

    task_id: str = Field(description="Current task identifier")
    task_description: str = Field(description="Human-readable task objective")
    documents: List[DocumentInfo] = Field(default_factory=list, description="Available items")
    current_content: Optional[str] = Field(default=None, description="Content of last-read item")
    feedback: str = Field(default="", description="Environment feedback from last action")
    available_actions: List[str] = Field(default_factory=list, description="Valid actions for this task")
    findings: Dict[str, Any] = Field(default_factory=dict, description="Agent's accumulated decisions")
    step_count: int = Field(default=0, description="Current step in episode")
    max_steps: int = Field(default=15, description="Maximum steps allowed")
    active_deadlines: List[DeadlineInfo] = Field(default_factory=list, description="Upcoming deadlines")
    ethical_alerts: List[str] = Field(default_factory=list, description="Active ethical constraint warnings")


# ── Response Models ─────────────────────────────────────────────────────────


class ResetResponse(BaseModel):
    """Standard OpenEnv Reset Response."""

    observation: ObservationResponse
    info: Dict[str, Any] = Field(default_factory=dict)


class StepResponse(BaseModel):
    """Standard OpenEnv Step Response mirroring Gymnasium."""

    observation: ObservationResponse
    reward: float = Field(description="Step reward signal")
    done: bool = Field(description="Whether episode has terminated")
    score: float = Field(default=0.0, description="Current grader score in [0.0, 1.0]")
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    """Current episode state returned by state()."""

    episode_id: str
    task_id: str
    step_count: int
    max_steps: int
    score: float
    cumulative_reward: float
    done: bool
    findings: Dict[str, Any] = Field(default_factory=dict)
