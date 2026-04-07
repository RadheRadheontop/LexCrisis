"""Core environment logic for the LexCrisis benchmark."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from lexcrisis_env.graders import GROUND_TRUTH, GRADERS, _SCORE_FLOOR, _SCORE_CEIL, _clamp_score
from lexcrisis_env.models import (
    Action,
    DeadlineSummary,
    DocumentSummary,
    EnvironmentState,
    Observation,
    Reward,
)
from lexcrisis_env.tasks import (
    CLIENTS,
    CONFLICT_RULES,
    CRISIS_EVENTS,
    CRISIS_GROUND_TRUTH,
    PRIVILEGE_DOCUMENTS,
    PRIVILEGE_GROUND_TRUTH,
    TASK_ACTIONS,
    TASK_DEFINITIONS,
    TERMINAL_ACTIONS,
    WAIVER_EVENTS,
    first_matching,
    get_client,
    get_document,
    get_event,
    normalize,
)

BENCHMARK_NAME = "lexcrisis"
BENCHMARK_VERSION = "1.0.0"


class LexCrisisEngine:
    """Stateful legal-ops engine for litigation incident response."""

    def __init__(self) -> None:
        self._episode_id = str(uuid4())
        self._task_id = "task_1"
        self._step_count = 0
        self._score = 0.0
        self._done = False
        self._last_reward = 0.0
        self._last_reward_model = Reward(
            value=0.0,
            score_delta=0.0,
            milestone_bonus=0.0,
            penalty=0.0,
            reason="Environment initialized.",
        )
        self._feedback = "Environment ready."
        self._current_content: Optional[str] = None
        self._ethical_alerts: List[str] = []
        self._findings: Dict[str, Any] = {}
        self._action_history: List[str] = []
        self._cumulative_reward = 0.0
        self.reset()

    def reset(
        self,
        task_id: str | None = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> Observation:
        """Reset the active episode and return a clean observation."""

        del seed  # Reserved for future stochastic variants.

        selected_task = task_id or self._task_id
        if selected_task not in TASK_DEFINITIONS:
            selected_task = "task_1"

        self._task_id = selected_task
        self._episode_id = episode_id or str(uuid4())
        self._step_count = 0
        self._score = 0.0
        self._done = False
        self._last_reward = 0.0
        self._cumulative_reward = 0.0
        self._current_content = None
        self._ethical_alerts = []
        self._findings = self._empty_findings(selected_task)
        self._action_history = []
        definition = TASK_DEFINITIONS[selected_task]
        self._feedback = (
            f"Environment reset for {definition.name}. "
            f"Work the problem like a legal operations team under pressure and submit before step {definition.max_steps}."
        )
        self._last_reward_model = Reward(
            value=0.0,
            score_delta=0.0,
            milestone_bonus=0.0,
            penalty=0.0,
            reason="Environment reset.",
        )

        observation = self._build_observation()
        return observation

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Apply one action and return the next observation."""

        if self._done:
            self._feedback = "Episode already complete. Call reset() before taking more actions."
            observation = self._build_observation()
            return observation, 0.0, True, self._info_payload()

        self._step_count += 1
        self._current_content = None

        old_score = self._run_grader()
        milestone_bonus, penalty, feedback = self._dispatch(action)
        loop_penalty = self._loop_penalty(action)
        penalty += loop_penalty
        new_score = self._run_grader()
        score_delta = round(new_score - old_score, 4)

        reward_value = round(score_delta + milestone_bonus + penalty, 4)
        self._last_reward = reward_value
        self._cumulative_reward = round(self._cumulative_reward + reward_value, 4)
        self._score = round(new_score, 4)

        terminal_action = TERMINAL_ACTIONS[self._task_id]
        if action.action_type == terminal_action:
            self._done = True
        if self._step_count >= TASK_DEFINITIONS[self._task_id].max_steps:
            self._done = True
            if action.action_type != terminal_action:
                feedback += " Max steps reached."
        if self._done:
            feedback += f" Final score: {self._score:.2f}."

        self._feedback = feedback
        self._last_reward_model = Reward(
            value=reward_value,
            score_delta=score_delta,
            milestone_bonus=round(milestone_bonus, 4),
            penalty=round(penalty, 4),
            reason=feedback,
        )

        observation = self._build_observation()
        self._action_history.append(self._fingerprint(action))
        return observation, reward_value, self._done, self._info_payload()

    def state(self) -> EnvironmentState:
        """Return the exact state contract requested in the prompt."""

        observation = self._build_observation()
        return EnvironmentState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            observation=observation.model_dump(exclude={"reward", "done", "metadata"}),
            reward=self._last_reward,
            done=self._done,
        )

    @property
    def last_score(self) -> float:
        return round(max(_SCORE_FLOOR, min(self._score, _SCORE_CEIL)), 4)

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def close(self) -> None:
        """No-op cleanup for API compatibility."""

    def episode_info(self) -> Dict[str, Any]:
        """Return UI-friendly episode metadata without changing the OpenEnv state contract."""

        payload = self._info_payload()
        payload.update(
            {
                "done": self._done,
                "mode": "simulation",
                "benchmark": BENCHMARK_NAME,
                "last_reward": self._last_reward,
                "last_reward_reason": self._last_reward_model.reason,
            }
        )
        return payload

    def _empty_findings(self, task_id: str) -> Dict[str, Any]:
        if task_id == "task_1":
            return {
                "reviewed_clients": [],
                "conflicts_identified": [],
                "rule_citations": [],
                "decisions": {},
            }
        if task_id == "task_2":
            return {
                "reviewed_documents": [],
                "privilege_classifications": {},
                "waivers_identified": [],
                "exceptions_identified": [],
                "recommendations": {},
            }
        return {
            "reviewed_events": [],
            "deadlines_met": {},
            "adversarial_flagged": [],
            "ethical_issues_flagged": [],
            "actions_taken": [],
            "expert_assessed": {},
            "discovery_response": {},
        }

    def _run_grader(self) -> float:
        grader = GRADERS[self._task_id]
        truth = GROUND_TRUTH[self._task_id]
        try:
            return grader(copy.deepcopy(self._findings), truth)
        except Exception:
            return _SCORE_FLOOR

    def _build_observation(self) -> Observation:
        definition = TASK_DEFINITIONS[self._task_id]
        documents: List[DocumentSummary]
        active_deadlines: List[DeadlineSummary] = []

        if self._task_id == "task_1":
            documents = [
                DocumentSummary(
                    item_id=client.client_id,
                    title=client.name,
                    item_type="client_intake",
                    category=client.client_type,
                )
                for client in CLIENTS
            ]
        elif self._task_id == "task_2":
            documents = [
                DocumentSummary(
                    item_id=document.doc_id,
                    title=document.title,
                    item_type="litigation_document",
                    category="privilege_review",
                )
                for document in PRIVILEGE_DOCUMENTS
            ]
        else:
            documents = [
                DocumentSummary(
                    item_id=event.event_id,
                    title=event.title,
                    item_type="crisis_event",
                    category=event.event_type,
                )
                for event in CRISIS_EVENTS
            ]
            for event in CRISIS_EVENTS:
                if event.deadline_step <= 0:
                    continue
                if event.event_id in self._findings.get("deadlines_met", {}):
                    continue
                remaining = event.deadline_step - self._step_count
                if remaining > 0:
                    active_deadlines.append(
                        DeadlineSummary(
                            item_id=event.event_id,
                            title=event.title,
                            steps_remaining=remaining,
                            consequence=event.consequence,
                        )
                    )

        return Observation(
            task_id=definition.task_id,
            task_name=definition.name,
            difficulty=definition.difficulty,  # type: ignore[arg-type]
            task_description=definition.description,
            documents=documents,
            current_content=self._current_content,
            available_actions=TASK_ACTIONS[self._task_id],
            findings=copy.deepcopy(self._findings),
            feedback=self._feedback,
            step_count=self._step_count,
            max_steps=definition.max_steps,
            active_deadlines=active_deadlines,
            ethical_alerts=list(self._ethical_alerts),
            done=self._done,
            reward=self._last_reward,
            metadata=self._info_payload(),
        )

    def _info_payload(self) -> Dict[str, Any]:
        return {
            "episode_id": self._episode_id,
            "task_id": self._task_id,
            "step_count": self._step_count,
            "score": self._score,
            "cumulative_reward": self._cumulative_reward,
            "reward_breakdown": self._last_reward_model.model_dump(),
        }

    def _fingerprint(self, action: Action) -> str:
        return f"{action.action_type}|{normalize(action.parameters)}"

    def _loop_penalty(self, action: Action) -> float:
        fingerprint = self._fingerprint(action)
        if action.action_type == "noop":
            overdue_count = 0
            if self._task_id == "task_3":
                for event_id, details in CRISIS_GROUND_TRUTH["deadlines"].items():
                    if self._step_count > details["deadline_step"] and event_id not in self._findings["deadlines_met"]:
                        overdue_count += 1
            return -0.02 - (0.03 * overdue_count)
        recent = self._action_history[-2:]
        if fingerprint in recent:
            return -0.02
        return 0.0

    def _dispatch(self, action: Action) -> Tuple[float, float, str]:
        handlers = {
            "review_client": self._review_client,
            "check_conflict": self._check_conflict,
            "cite_rule": self._cite_rule,
            "accept_client": self._decide_client,
            "decline_client": self._decide_client,
            "submit_intake": self._submit,
            "review_document": self._review_document,
            "classify_privilege": self._classify_privilege,
            "identify_waiver": self._identify_waiver,
            "identify_exception": self._identify_exception,
            "recommend_action": self._recommend_action,
            "submit_review": self._submit,
            "review_event": self._review_event,
            "issue_litigation_hold": self._issue_litigation_hold,
            "file_motion": self._file_motion,
            "respond_discovery": self._respond_discovery,
            "assess_expert": self._assess_expert,
            "flag_adversarial": self._flag_adversarial,
            "flag_ethical_issue": self._flag_ethical_issue,
            "submit_triage": self._submit,
            "noop": self._noop,
        }
        handler = handlers.get(action.action_type)
        if handler is None:
            return 0.0, -0.05, f"Unknown action '{action.action_type}'."
        return handler(action)

    def _review_client(self, action: Action) -> Tuple[float, float, str]:
        client_id = str(action.parameters.get("client_id", "")).upper()
        client = get_client(client_id)
        if client is None:
            return 0.0, -0.03, f"Unknown client '{client_id}'."
        reviewed = self._findings["reviewed_clients"]
        milestone_bonus = 0.02 if client_id not in reviewed else 0.0
        penalty = -0.01 if client_id in reviewed else 0.0
        if client_id not in reviewed:
            reviewed.append(client_id)
        self._current_content = (
            f"{client.name} ({client.client_id})\n"
            f"Type: {client.client_type}\n"
            f"Summary: {client.summary}\n"
            f"Details: {client.details}\n"
            f"Relationships: {', '.join(client.relationships) if client.relationships else 'None'}"
        )
        return milestone_bonus, penalty, f"Reviewed client intake for {client.name}."

    def _check_conflict(self, action: Action) -> Tuple[float, float, str]:
        client_a = str(action.parameters.get("client_a", "")).upper()
        client_b = str(action.parameters.get("client_b", "")).upper()
        if not client_a or not client_b or client_a == client_b:
            return 0.0, -0.03, "Provide two distinct client IDs to check a conflict."
        pair = frozenset((client_a, client_b))
        existing = {
            frozenset((entry["client_a"], entry["client_b"]))
            for entry in self._findings["conflicts_identified"]
        }
        if pair not in existing:
            self._findings["conflicts_identified"].append({"client_a": client_a, "client_b": client_b})

        review_penalty = 0.0
        reviewed = set(self._findings["reviewed_clients"])
        if client_a not in reviewed or client_b not in reviewed:
            review_penalty -= 0.01

        if pair in CONFLICT_RULES:
            return 0.03, review_penalty, f"Conflict identified between {client_a} and {client_b}."
        return 0.0, review_penalty - 0.04, f"No conflict exists between {client_a} and {client_b}."

    def _cite_rule(self, action: Action) -> Tuple[float, float, str]:
        client_a = str(action.parameters.get("client_a", "")).upper()
        client_b = str(action.parameters.get("client_b", "")).upper()
        rule = str(action.parameters.get("rule", ""))
        if not client_a or not client_b or not rule:
            return 0.0, -0.03, "Rule citation requires client_a, client_b, and rule."
        pair = frozenset((client_a, client_b))
        citations = self._findings["rule_citations"]
        citations[:] = [entry for entry in citations if frozenset((entry["client_a"], entry["client_b"])) != pair]
        citations.append({"client_a": client_a, "client_b": client_b, "rule": rule})
        expected = normalize(CONFLICT_RULES.get(pair, ""))
        provided = normalize(rule)
        if expected and (provided == expected or provided in expected or expected in provided):
            return 0.02, 0.0, f"Applied the correct conflict rule for {client_a} and {client_b}."
        return 0.0, -0.03, f"Rule citation for {client_a} and {client_b} does not match the expected basis."

    def _decide_client(self, action: Action) -> Tuple[float, float, str]:
        client_id = str(action.parameters.get("client_id", "")).upper()
        if not client_id:
            return 0.0, -0.03, "Client decision requires a client_id."
        decision = "accept" if action.action_type == "accept_client" else "decline"
        penalty = 0.0
        if client_id not in self._findings["reviewed_clients"]:
            penalty -= 0.01
        if client_id in self._findings["decisions"]:
            penalty -= 0.01
        self._findings["decisions"][client_id] = decision
        expected = GROUND_TRUTH["task_1"]["correct_decisions"].get(client_id)
        if normalize(expected) == decision:
            return 0.02, penalty, f"{client_id} marked as {decision}."
        return 0.0, penalty - 0.03, f"{client_id} marked as {decision}, but that choice increases conflict risk."

    def _submit(self, action: Action) -> Tuple[float, float, str]:
        if action.action_type == "submit_intake" and len(self._findings["decisions"]) < 4:
            return 0.0, -0.02, "Intake submitted early with too few client decisions."
        if action.action_type == "submit_review" and len(self._findings["privilege_classifications"]) < 2:
            return 0.0, -0.02, "Privilege review submitted before enough documents were analyzed."
        if action.action_type == "submit_triage" and len(self._findings["actions_taken"]) < 4:
            return 0.0, -0.02, "Triage submitted before enough crisis actions were taken."
        return 0.0, 0.0, "Submission recorded for grading."

    def _review_document(self, action: Action) -> Tuple[float, float, str]:
        doc_id = str(action.parameters.get("doc_id", "")).upper()
        document = get_document(doc_id)
        if document is None:
            return 0.0, -0.03, f"Unknown document '{doc_id}'."
        reviewed = self._findings["reviewed_documents"]
        milestone_bonus = 0.02 if doc_id not in reviewed else 0.0
        penalty = -0.01 if doc_id in reviewed else 0.0
        if doc_id not in reviewed:
            reviewed.append(doc_id)
        self._current_content = (
            f"{document.title} ({document.doc_id})\n"
            f"Doctrine hint: {document.doctrine}\n"
            f"Content: {document.content}"
        )
        return milestone_bonus, penalty, f"Reviewed document {doc_id}."

    def _classify_privilege(self, action: Action) -> Tuple[float, float, str]:
        doc_id = str(action.parameters.get("doc_id", "")).upper()
        classification = normalize(action.parameters.get("classification"))
        doctrine = str(action.parameters.get("doctrine", ""))
        valid = {"attorney_client", "work_product", "both", "none", "waived"}
        if doc_id not in PRIVILEGE_GROUND_TRUTH or classification not in valid:
            return 0.0, -0.03, "Privilege classification requires a valid doc_id and classification."
        penalty = 0.0
        if doc_id not in self._findings["reviewed_documents"]:
            penalty -= 0.01
        self._findings["privilege_classifications"][doc_id] = {
            "classification": classification,
            "doctrine": doctrine,
        }
        expected = normalize(PRIVILEGE_GROUND_TRUTH[doc_id]["classification"])
        if classification == expected:
            return 0.03, penalty, f"{doc_id} classified correctly as {classification}."
        if classification in {"attorney_client", "work_product", "both"} and expected in {
            "attorney_client",
            "work_product",
            "both",
        }:
            return 0.01, penalty, f"{doc_id} is privileged, but the subtype needs refinement."
        return 0.0, penalty - 0.03, f"{doc_id} classification is incorrect."

    def _identify_waiver(self, action: Action) -> Tuple[float, float, str]:
        doc_id = str(action.parameters.get("doc_id", "")).upper()
        waiver_type = normalize(action.parameters.get("waiver_type"))
        if not doc_id or not waiver_type:
            return 0.0, -0.03, "Waiver identification requires doc_id and waiver_type."
        entries = self._findings["waivers_identified"]
        entries[:] = [entry for entry in entries if entry.get("doc_id") != doc_id]
        entries.append(
            {
                "doc_id": doc_id,
                "waiver_type": waiver_type,
                "explanation": str(action.parameters.get("explanation", "")),
            }
        )
        expected = normalize(WAIVER_EVENTS.get(doc_id, ""))
        if waiver_type == expected:
            return 0.03, 0.0, f"Waiver risk correctly identified for {doc_id}."
        return 0.0, -0.03, f"Waiver call for {doc_id} does not match the ground truth."

    def _identify_exception(self, action: Action) -> Tuple[float, float, str]:
        doc_id = str(action.parameters.get("doc_id", "")).upper()
        exception_type = normalize(action.parameters.get("exception_type"))
        if not doc_id or not exception_type:
            return 0.0, -0.03, "Exception identification requires doc_id and exception_type."
        entries = self._findings["exceptions_identified"]
        entries[:] = [entry for entry in entries if entry.get("doc_id") != doc_id]
        entries.append(
            {
                "doc_id": doc_id,
                "exception_type": exception_type,
                "explanation": str(action.parameters.get("explanation", "")),
            }
        )
        expected = normalize(PRIVILEGE_GROUND_TRUTH.get(doc_id, {}).get("exception"))
        if expected != "none" and exception_type == expected:
            return 0.02, 0.0, f"Privilege exception correctly identified for {doc_id}."
        return 0.0, -0.02, f"Exception assessment for {doc_id} is not supported by the scenario."

    def _recommend_action(self, action: Action) -> Tuple[float, float, str]:
        doc_id = str(action.parameters.get("doc_id", "")).upper()
        recommendation = normalize(action.parameters.get("action"))
        if doc_id not in PRIVILEGE_GROUND_TRUTH or not recommendation:
            return 0.0, -0.03, "Recommendation requires doc_id and action."
        penalty = 0.0
        if doc_id not in self._findings["reviewed_documents"]:
            penalty -= 0.01
        self._findings["recommendations"][doc_id] = {
            "action": recommendation,
            "reasoning": str(action.parameters.get("reasoning", "")),
        }
        expected = normalize(PRIVILEGE_GROUND_TRUTH[doc_id]["action"])
        if recommendation == expected:
            return 0.02, penalty, f"Production recommendation for {doc_id} is correct."
        return 0.0, penalty - 0.02, f"Production recommendation for {doc_id} is misaligned with the privilege posture."

    def _review_event(self, action: Action) -> Tuple[float, float, str]:
        event_id = str(action.parameters.get("event_id", "")).upper()
        event = get_event(event_id)
        if event is None:
            return 0.0, -0.03, f"Unknown event '{event_id}'."
        reviewed = self._findings["reviewed_events"]
        bonus = 0.02 if event_id not in reviewed else 0.0
        penalty = -0.01 if event_id in reviewed else 0.0
        if event_id not in reviewed:
            reviewed.append(event_id)
        self._findings["actions_taken"].append(
            {"event_id": event_id, "action": "review_event", "step": self._step_count}
        )
        self._current_content = (
            f"{event.title} ({event.event_id})\n"
            f"Type: {event.event_type}\n"
            f"Deadline step: {event.deadline_step}\n"
            f"Consequence: {event.consequence}\n"
            f"Scenario: {event.content}"
        )
        return bonus, penalty, f"Reviewed crisis event {event_id}."

    def _issue_litigation_hold(self, action: Action) -> Tuple[float, float, str]:
        scope = str(action.parameters.get("scope", ""))
        custodians = action.parameters.get("custodians", [])
        if isinstance(custodians, str):
            custodians = [item.strip() for item in custodians.split(",") if item.strip()]
        if not scope or not custodians:
            return 0.0, -0.03, "Litigation hold requires scope and custodians."
        self._findings["deadlines_met"]["EVENT-001"] = {
            "step": self._step_count,
            "scope": scope,
            "custodians": custodians,
        }
        self._findings["actions_taken"].append(
            {"event_id": "EVENT-001", "action": "issue_litigation_hold", "step": self._step_count}
        )
        deadline = CRISIS_GROUND_TRUTH["deadlines"]["EVENT-001"]["deadline_step"]
        coverage = first_matching(custodians, ["morton", "ames", "wong", "liu", "park"])
        coverage_bonus = 0.02 if coverage else 0.0
        if self._step_count <= deadline:
            return 0.05 + coverage_bonus, 0.0, "Litigation hold issued before the preservation deadline."
        return 0.0, -0.08, "Litigation hold was issued after the preservation deadline."

    def _file_motion(self, action: Action) -> Tuple[float, float, str]:
        motion_type = normalize(action.parameters.get("motion_type"))
        court = str(action.parameters.get("court", ""))
        if not motion_type:
            return 0.0, -0.03, "Motion filing requires motion_type."
        target_event = ""
        if any(term in motion_type for term in ("injunction", "tro", "opposition")):
            target_event = "EVENT-002"
        elif any(term in motion_type for term in ("transfer", "consolidation", "forum")):
            target_event = "EVENT-005"
        if not target_event:
            self._findings["actions_taken"].append(
                {"event_id": "UNMAPPED", "action": "file_motion", "step": self._step_count}
            )
            return 0.0, -0.02, "The motion was filed, but it does not resolve a scored crisis event."
        self._findings["deadlines_met"][target_event] = {
            "step": self._step_count,
            "motion_type": motion_type,
            "court": court,
        }
        self._findings["actions_taken"].append(
            {"event_id": target_event, "action": "file_motion", "step": self._step_count}
        )
        deadline = CRISIS_GROUND_TRUTH["deadlines"][target_event]["deadline_step"]
        if self._step_count <= deadline:
            return 0.05, 0.0, f"Motion work for {target_event} was filed on time."
        return 0.0, -0.06, f"Motion work for {target_event} was filed late."

    def _respond_discovery(self, action: Action) -> Tuple[float, float, str]:
        response_type = normalize(action.parameters.get("response_type"))
        objections = str(action.parameters.get("objections", ""))
        if not response_type:
            return 0.0, -0.03, "Discovery response requires response_type."
        self._findings["discovery_response"] = {
            "step": self._step_count,
            "response_type": response_type,
            "objections": objections,
        }
        self._findings["deadlines_met"]["EVENT-003"] = {
            "step": self._step_count,
            "response_type": response_type,
        }
        self._findings["actions_taken"].append(
            {"event_id": "EVENT-003", "action": "respond_discovery", "step": self._step_count}
        )
        deadline = CRISIS_GROUND_TRUTH["deadlines"]["EVENT-003"]["deadline_step"]
        if response_type == "produce" and not objections:
            self._ethical_alerts.append(
                "Producing advocate communications without objections risks broad privilege waiver."
            )
            return 0.0, -0.12, "Discovery response risks waiving privilege by producing without objections."
        bonus = 0.03 if self._step_count <= deadline else 0.0
        penalty = -0.05 if self._step_count > deadline else 0.0
        if response_type in {"privilege_log", "object", "partial_produce"} and any(
            term in normalize(objections) for term in ("privilege", "section 126", "section 129", "advocate")
        ):
            return bonus + 0.03, penalty, "Discovery response preserved privilege and documented objections."
        return bonus, penalty - 0.02, "Discovery response was recorded, but the privilege rationale is weak."

    def _assess_expert(self, action: Action) -> Tuple[float, float, str]:
        qualification = str(action.parameters.get("qualification", ""))
        if not qualification:
            return 0.0, -0.03, "Expert assessment requires qualification details."
        self._findings["expert_assessed"] = {
            "expert_id": str(action.parameters.get("expert_id", "EXPERT")),
            "qualification": qualification,
            "step": self._step_count,
        }
        self._findings["actions_taken"].append(
            {"event_id": "EVENT-005", "action": "assess_expert", "step": self._step_count}
        )
        if any(term in normalize(qualification) for term in ("special skill", "science", "toxicology", "section 45")):
            return 0.03, 0.0, "Expert qualification analysis addresses the key admissibility factors."
        return 0.0, -0.01, "Expert qualification analysis is too shallow for Section 45 review."

    def _flag_adversarial(self, action: Action) -> Tuple[float, float, str]:
        item_id = str(action.parameters.get("item_id", "")).upper()
        threat_type = str(action.parameters.get("threat_type", ""))
        if not item_id or not threat_type:
            return 0.0, -0.03, "Adversarial flagging requires item_id and threat_type."
        flags = self._findings["adversarial_flagged"]
        flags[:] = [entry for entry in flags if entry.get("item_id") != item_id]
        flags.append(
            {
                "item_id": item_id,
                "threat_type": threat_type,
                "explanation": str(action.parameters.get("explanation", "")),
            }
        )
        if item_id in CRISIS_GROUND_TRUTH["adversarial_items"]:
            return 0.03, 0.0, f"Adversarial pattern correctly flagged for {item_id}."
        return 0.0, -0.02, f"{item_id} is not a scored adversarial event."

    def _flag_ethical_issue(self, action: Action) -> Tuple[float, float, str]:
        issue_type = str(action.parameters.get("issue_type", ""))
        resolution = str(action.parameters.get("resolution", ""))
        if not issue_type:
            return 0.0, -0.03, "Ethical issue flagging requires issue_type."
        entries = self._findings["ethical_issues_flagged"]
        entries[:] = [entry for entry in entries if entry.get("event_id") != "EVENT-004"]
        entries.append(
            {
                "event_id": "EVENT-004",
                "issue_type": issue_type,
                "affected_clients": str(action.parameters.get("affected_clients", "")),
                "resolution": resolution,
            }
        )
        self._findings["actions_taken"].append(
            {"event_id": "EVENT-004", "action": "flag_ethical_issue", "step": self._step_count}
        )
        keywords = ("withdraw", "screen", "consent", "disclose", "rule 33", "former client")
        if any(keyword in normalize(resolution) for keyword in keywords):
            return 0.05, 0.0, "Ethical conflict surfaced with a defensible mitigation plan."
        return 0.02, 0.0, "Ethical conflict was flagged, but the resolution needs stronger mitigation language."

    def _noop(self, action: Action) -> Tuple[float, float, str]:
        del action
        return 0.0, 0.0, "No action taken."


_ENGINE = LexCrisisEngine()


class LexCrisisEnvironment(Environment[Action, Observation, EnvironmentState]):
    """OpenEnv-compatible wrapper around the shared LexCrisis engine."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        super().__init__()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        del kwargs
        return _ENGINE.reset(task_id=task_id, seed=seed, episode_id=episode_id)

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        del timeout_s, kwargs
        observation, _, _, _ = _ENGINE.step(action)
        return observation

    @property
    def state(self) -> EnvironmentState:
        return _ENGINE.state()

    @property
    def last_score(self) -> float:
        return _ENGINE.last_score

    @property
    def episode_id(self) -> str:
        return _ENGINE.episode_id

    def episode_info(self) -> Dict[str, Any]:
        return _ENGINE.episode_info()

    def get_metadata(self) -> EnvironmentMetadata:
        readme_path = Path(__file__).resolve().parents[1] / "README.md"
        readme_content = None
        if readme_path.exists():
            readme_content = readme_path.read_text(encoding="utf-8")
        return EnvironmentMetadata(
            name="LexCrisis",
            description=(
                "Law-focused benchmark for legal operations incident response in "
                "high-stakes product-liability litigation."
            ),
            readme_content=readme_content,
            version=BENCHMARK_VERSION,
            author="OpenEnv Hackathon Submission",
        )

    def close(self) -> None:
        """Environment instances are stateless wrappers over the shared engine."""
        return None
