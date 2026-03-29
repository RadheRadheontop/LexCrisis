"""
Core environment logic for the LexCrisis Legal Crisis Management Environment.

Implements reset(), step(), and state() with full reward shaping
and deterministic grading per the OpenEnv specification.

Supports 3 tasks across escalating difficulty:
  task_1: Client Conflict Screening (Easy)
  task_2: Privileged Document Review (Medium)
  task_3: Multi-Front Crisis Triage (Hard)

Reward Design:
  reward = (new_grader_score - old_grader_score) + step_penalty

  - Grader delta captures meaningful progress toward task completion.
  - step_penalty is applied for invalid/incorrect actions.
  - This provides dense, well-scaled signals for RL training.

Legal Citations:
  Bar Council of India (BCI) Rules, Part VI, Chapter II, Section II
  Civil Procedure Code (CPC) Order XI
  Indian Evidence Act, 1872, Sections 45, 126, 129
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from server.models import (
    ActionRequest,
    DeadlineInfo,
    DocumentInfo,
    ObservationResponse,
    ResetResponse,
    StateResponse,
    StepResponse,
)
from server.scenarios import (
    CLIENTS,
    CONFLICT_PAIRS,
    CORRECT_DECISIONS,
    CRISIS_EVENTS,
    CRISIS_GROUND_TRUTH,
    PRIVILEGE_DOCUMENTS,
    PRIVILEGE_GROUND_TRUTH,
    WAIVER_EVENTS,
    get_client,
    get_document,
    get_event,
)
from server.tasks import GRADERS, TASK_DEFINITIONS, _normalize


# Valid actions per task
TASK_ACTIONS: Dict[str, set] = {
    "task_1": {
        "review_client", "check_conflict", "cite_rule",
        "accept_client", "decline_client", "submit_intake",
    },
    "task_2": {
        "review_document", "classify_privilege", "identify_waiver",
        "identify_exception", "recommend_action", "submit_review",
    },
    "task_3": {
        "review_event", "issue_litigation_hold", "file_motion",
        "respond_discovery", "assess_expert", "flag_adversarial",
        "flag_ethical_issue", "submit_triage",
    },
}

TERMINAL_ACTIONS = {"submit_intake", "submit_review", "submit_triage"}


class LegalDocEnvironment:
    """LexCrisis: Multi-Dimensional Legal Crisis Management Environment.

    Manages episode state, action dispatch, grading, and reward computation
    for all 3 legal crisis management tasks.
    """

    def __init__(self) -> None:
        self._task_id: str = "task_1"
        self._findings: Dict[str, Any] = {}
        self._step_count: int = 0
        self._max_steps: int = 15
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._score: float = 0.0
        self._episode_id: str = str(uuid4())
        self._ground_truth: Dict[str, Any] = {}
        self._last_content: Optional[str] = None
        self._ethical_alerts: List[str] = []

    # ── OpenEnv API ─────────────────────────────────────────────────────────

    def reset(self, task_id: str = "task_1", episode_id: Optional[str] = None) -> ResetResponse:
        """Reset environment for a new episode.

        Clears all state, loads scenario data for the specified task,
        and returns the initial observation with an empty findings map.
        """
        if task_id not in TASK_DEFINITIONS:
            task_id = "task_1"

        task_def = TASK_DEFINITIONS[task_id]
        self._task_id = task_id
        self._step_count = 0
        self._max_steps = task_def.max_steps
        self._cumulative_reward = 0.0
        self._done = False
        self._score = 0.0
        self._episode_id = episode_id or str(uuid4())
        self._last_content = None
        self._ethical_alerts = []

        # Initialize clean findings map (no carryover between episodes)
        self._findings = self._init_findings(task_id)
        self._ground_truth = self._build_ground_truth(task_id)

        obs = self._make_observation(
            feedback=(
                f"Environment reset. Task: {task_def.name} ({task_def.difficulty}).\n"
                f"Complete the task in ≤{self._max_steps} steps.\n\n"
                f"{task_def.description}"
            )
        )
        return ResetResponse(observation=obs, info={"episode_id": self._episode_id})

    def step(self, action: ActionRequest) -> StepResponse:
        """Process one agent action and return observation + reward.

        Reward computation:
            1. Snapshot grader score before processing action.
            2. Execute action handler which updates findings.
            3. Re-run grader to get new score.
            4. reward = (new_score - old_score) + step_penalty
        """
        if self._done:
            return StepResponse(
                observation=self._make_observation(
                    "Episode is already complete. Call reset() to start a new episode."
                ),
                reward=0.0,
                done=True,
                score=self._score,
                info={"episode_id": self._episode_id, "score": self._score},
            )

        self._step_count += 1
        self._last_content = None
        action_type = action.action_type
        params = action.parameters

        # Snapshot grader score before action
        old_score = self._run_grader()

        # Validate action is valid for current task
        valid_actions = TASK_ACTIONS.get(self._task_id, set())
        if action_type not in valid_actions:
            step_penalty = -0.05
            feedback = (
                f"Invalid action '{action_type}' for {self._task_id}. "
                f"Valid actions: {sorted(valid_actions)}"
            )
        else:
            handler_reward, feedback = self._dispatch_action(action_type, params)
            # Only apply negative handler results as penalties
            step_penalty = handler_reward if handler_reward < 0 else 0.0

        # Re-run grader after action
        new_score = self._run_grader()

        reward = (new_score - old_score) + step_penalty
        self._cumulative_reward += reward
        self._score = new_score

        # Check terminal conditions
        if action_type in TERMINAL_ACTIONS or self._step_count >= self._max_steps:
            self._done = True
            feedback += f"\n\n=== EPISODE COMPLETE ===\nFinal Score: {self._score:.4f}"

        obs = self._make_observation(feedback=feedback)
        return StepResponse(
            observation=obs,
            reward=round(reward, 6),
            done=self._done,
            score=round(self._score, 4),
            info={
                "episode_id": self._episode_id,
                "score": round(self._score, 4),
                "cumulative_reward": round(self._cumulative_reward, 6),
            },
        )

    def state(self) -> StateResponse:
        """Return current episode state without modifying it."""
        return StateResponse(
            episode_id=self._episode_id,
            task_id=self._task_id,
            step_count=self._step_count,
            max_steps=self._max_steps,
            score=round(self._score, 4),
            cumulative_reward=round(self._cumulative_reward, 6),
            done=self._done,
            findings=self._findings,
        )

    # ── Action Dispatch ─────────────────────────────────────────────────────

    def _dispatch_action(self, action_type: str, params: Dict[str, Any]) -> Tuple[float, str]:
        """Route to the correct handler for the current task."""
        handlers = {
            # Task 1
            "review_client": self._act_review_client,
            "check_conflict": self._act_check_conflict,
            "cite_rule": self._act_cite_rule,
            "accept_client": self._act_accept_client,
            "decline_client": self._act_decline_client,
            "submit_intake": self._act_submit_terminal,
            # Task 2
            "review_document": self._act_review_document,
            "classify_privilege": self._act_classify_privilege,
            "identify_waiver": self._act_identify_waiver,
            "identify_exception": self._act_identify_exception,
            "recommend_action": self._act_recommend_action,
            "submit_review": self._act_submit_terminal,
            # Task 3
            "review_event": self._act_review_event,
            "issue_litigation_hold": self._act_litigation_hold,
            "file_motion": self._act_file_motion,
            "respond_discovery": self._act_respond_discovery,
            "assess_expert": self._act_assess_expert,
            "flag_adversarial": self._act_flag_adversarial,
            "flag_ethical_issue": self._act_flag_ethical,
            "submit_triage": self._act_submit_terminal,
        }

        handler = handlers.get(action_type)
        if handler is None:
            return -0.05, f"Unhandled action: {action_type}"

        try:
            return handler(params)
        except Exception as e:
            return -0.05, f"Error processing action '{action_type}': {e}"

    # ── Task 1 Handlers: Client Conflict Screening ──────────────────────────

    def _act_review_client(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """Review a client's intake record."""
        client_id = str(params.get("client_id", "")).upper()
        client = get_client(client_id)
        if not client:
            return -0.02, f"Unknown client '{client_id}'. Valid: CLIENT-001 through CLIENT-006."

        text = (
            f"{'='*60}\n"
            f"  CLIENT INTAKE: {client.name} ({client.client_id})\n"
            f"{'='*60}\n\n"
            f"Type: {client.client_type}\n"
            f"Summary: {client.summary}\n\n"
            f"Details:\n{client.details}\n\n"
            f"Known relationships: {', '.join(client.relationships) if client.relationships else 'None'}\n"
            f"{'='*60}"
        )
        self._last_content = text
        return 0.0, text

    def _act_check_conflict(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """Check for conflict of interest between two clients."""
        client_a = str(params.get("client_a", "")).upper()
        client_b = str(params.get("client_b", "")).upper()

        if not client_a or not client_b:
            return -0.02, "Missing 'client_a' and/or 'client_b' parameters."
        if client_a == client_b:
            return -0.02, "Cannot check conflict between a client and itself."

        pair = frozenset([client_a, client_b])
        entry = {"client_a": client_a, "client_b": client_b}

        # Avoid duplicate entries
        existing_pairs = {
            frozenset([c["client_a"], c["client_b"]])
            for c in self._findings.get("conflicts_identified", [])
        }
        if pair not in existing_pairs:
            self._findings["conflicts_identified"].append(entry)

        # Check against ground truth
        gt_pairs = self._ground_truth.get("conflict_pairs", set())
        if pair in gt_pairs:
            return 0.0, f"✓ Conflict identified between {client_a} and {client_b}. Cite the applicable BCI rule."
        else:
            return -0.03, f"No conflict found between {client_a} and {client_b}. Recorded."

    def _act_cite_rule(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """Cite the BCI Rule applicable to a conflict pair."""
        client_a = str(params.get("client_a", "")).upper()
        client_b = str(params.get("client_b", "")).upper()
        rule = str(params.get("rule", "")).strip()

        if not client_a or not client_b or not rule:
            return -0.02, "Missing 'client_a', 'client_b', or 'rule' parameter."

        entry = {"client_a": client_a, "client_b": client_b, "rule": rule}
        self._findings["rule_citations"].append(entry)

        gt_rules = self._ground_truth.get("conflict_rules", {})
        pair = frozenset([client_a, client_b])
        if pair in gt_rules:
            gt_rule = _normalize(gt_rules[pair])
            cited_rule = _normalize(rule)
            if cited_rule == gt_rule or gt_rule.startswith(cited_rule.split("(")[0]):
                return 0.0, f"✓ Rule {rule} cited for {client_a}/{client_b}. Correct."
            else:
                return -0.02, f"Rule {rule} cited. Expected a different rule for this pair."
        return 0.0, f"Rule {rule} cited for {client_a}/{client_b}. Recorded."

    def _act_accept_client(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """Accept a client for representation."""
        client_id = str(params.get("client_id", "")).upper()
        justification = str(params.get("justification", ""))

        if not client_id:
            return -0.02, "Missing 'client_id' parameter."

        self._findings["decisions"][client_id] = "accept"

        gt = self._ground_truth.get("correct_decisions", {})
        if client_id in gt and _normalize(gt[client_id]) == "accept":
            return 0.0, f"✓ {client_id} accepted for representation."
        elif client_id in gt:
            return -0.05, f"{client_id} accepted. Decision recorded."
        return 0.0, f"{client_id} accepted. Decision recorded."

    def _act_decline_client(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """Decline to represent a client."""
        client_id = str(params.get("client_id", "")).upper()
        reason = str(params.get("reason", ""))

        if not client_id:
            return -0.02, "Missing 'client_id' parameter."

        self._findings["decisions"][client_id] = "decline"

        gt = self._ground_truth.get("correct_decisions", {})
        if client_id in gt and _normalize(gt[client_id]) == "decline":
            return 0.0, f"✓ {client_id} declined. Reason: {reason[:100]}"
        elif client_id in gt:
            return -0.05, f"{client_id} declined. Decision recorded."
        return 0.0, f"{client_id} declined. Decision recorded."

    # ── Task 2 Handlers: Privilege Review ───────────────────────────────────

    def _act_review_document(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """Review a document from the privilege review set."""
        doc_id = str(params.get("doc_id", "")).upper()
        doc = get_document(doc_id)
        if not doc:
            return -0.02, f"Unknown document '{doc_id}'. Valid: DOC-001 through DOC-008."

        text = (
            f"{'='*60}\n"
            f"  DOCUMENT: {doc.title} ({doc.doc_id})\n"
            f"{'='*60}\n\n"
            f"{doc.content}\n\n"
            f"{'='*60}"
        )
        self._last_content = text
        return 0.0, text

    def _act_classify_privilege(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """Classify a document's privilege status."""
        doc_id = str(params.get("doc_id", "")).upper()
        classification = _normalize(params.get("classification", ""))
        doctrine = str(params.get("doctrine", ""))

        if not doc_id or not classification:
            return -0.02, "Missing 'doc_id' or 'classification' parameter."

        valid_classes = {"attorney_client", "work_product", "both", "none", "waived"}
        if classification not in valid_classes:
            return -0.02, f"Invalid classification '{classification}'. Valid: {sorted(valid_classes)}"

        self._findings["privilege_classifications"][doc_id] = {
            "classification": classification,
            "doctrine": doctrine,
        }

        gt = PRIVILEGE_GROUND_TRUTH.get(doc_id, {})
        if gt and _normalize(gt.get("classification", "")) == classification:
            return 0.0, f"✓ {doc_id} classified as '{classification}'. Correct."
        elif gt:
            return -0.03, f"{doc_id} classified as '{classification}'. Recorded."
        return 0.0, f"{doc_id} classified as '{classification}'. Recorded."

    def _act_identify_waiver(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """Identify a privilege waiver event."""
        doc_id = str(params.get("doc_id", "")).upper()
        waiver_type = _normalize(params.get("waiver_type", ""))
        explanation = str(params.get("explanation", ""))

        if not doc_id or not waiver_type:
            return -0.02, "Missing 'doc_id' or 'waiver_type' parameter."

        entry = {"doc_id": doc_id, "waiver_type": waiver_type, "explanation": explanation}
        existing_docs = {w["doc_id"] for w in self._findings["waivers_identified"]}
        if doc_id not in existing_docs:
            self._findings["waivers_identified"].append(entry)

        # Check ground truth
        gt_waiver_docs = {w["doc_id"].upper() for w in WAIVER_EVENTS}
        if doc_id in gt_waiver_docs:
            return 0.0, f"✓ Waiver event identified in {doc_id}: {waiver_type}."
        else:
            return -0.03, f"Waiver flagged for {doc_id}. No ground-truth waiver known."

    def _act_identify_exception(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """Identify a privilege exception (e.g., crime-fraud)."""
        doc_id = str(params.get("doc_id", "")).upper()
        exception_type = _normalize(params.get("exception_type", ""))
        explanation = str(params.get("explanation", ""))

        if not doc_id or not exception_type:
            return -0.02, "Missing 'doc_id' or 'exception_type' parameter."

        self._findings["exceptions_identified"].append({
            "doc_id": doc_id, "exception_type": exception_type, "explanation": explanation,
        })

        gt = PRIVILEGE_GROUND_TRUTH.get(doc_id, {})
        if gt and _normalize(gt.get("exception", "none")) == exception_type:
            return 0.0, f"✓ Exception '{exception_type}' correctly identified in {doc_id}."
        return -0.03, f"Exception '{exception_type}' flagged for {doc_id}. Recorded."

    def _act_recommend_action(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """Recommend handling action for a document."""
        doc_id = str(params.get("doc_id", "")).upper()
        action = _normalize(params.get("action", ""))
        reasoning = str(params.get("reasoning", ""))

        if not doc_id or not action:
            return -0.02, "Missing 'doc_id' or 'action' parameter."

        valid_actions = {"withhold", "produce", "clawback", "redact"}
        if action not in valid_actions:
            return -0.02, f"Invalid action '{action}'. Valid: {sorted(valid_actions)}"

        self._findings["recommendations"][doc_id] = {"action": action, "reasoning": reasoning}

        gt = PRIVILEGE_GROUND_TRUTH.get(doc_id, {})
        if gt and _normalize(gt.get("action", "")) == action:
            return 0.0, f"✓ Recommendation for {doc_id}: {action}. Correct."
        return -0.02, f"Recommendation for {doc_id}: {action}. Recorded."

    # ── Task 3 Handlers: Crisis Triage ──────────────────────────────────────

    def _act_review_event(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """Review a crisis event."""
        event_id = str(params.get("event_id", "")).upper()
        event = get_event(event_id)
        if not event:
            return -0.02, f"Unknown event '{event_id}'. Valid: EVENT-001 through EVENT-005."

        text = (
            f"{'='*60}\n"
            f"  CRISIS EVENT: {event.title} ({event.event_id})\n"
            f"  Type: {event.event_type}"
        )
        if event.deadline_step > 0:
            remaining = max(0, event.deadline_step - self._step_count)
            text += f" | Deadline: {remaining} steps remaining"
        text += (
            f"\n{'='*60}\n\n"
            f"{event.content}\n\n"
            f"{'='*60}"
        )
        self._last_content = text

        # Track that this event was reviewed
        self._findings["actions_taken"].append({
            "event_id": event_id, "action": "review_event", "step": self._step_count,
        })
        return 0.0, text

    def _act_litigation_hold(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """Issue a litigation hold notice."""
        scope = str(params.get("scope", ""))
        custodians = params.get("custodians", [])
        if isinstance(custodians, str):
            custodians = [c.strip() for c in custodians.split(",") if c.strip()]

        if not scope:
            return -0.02, "Missing 'scope' parameter (describe what must be preserved)."
        if not custodians:
            return -0.02, "Missing 'custodians' parameter (list of people who must preserve documents)."

        self._findings["deadlines_met"]["EVENT-001"] = {
            "step": self._step_count,
            "scope": scope,
            "custodians": custodians,
        }
        self._findings["actions_taken"].append({
            "event_id": "EVENT-001", "action": "issue_litigation_hold", "step": self._step_count,
        })

        gt_deadline = CRISIS_GROUND_TRUTH["deadlines"].get("EVENT-001", {})
        deadline_step = gt_deadline.get("deadline_step", 999)
        if self._step_count <= deadline_step:
            # Check custodian coverage
            key_custodians = {"morton", "ames", "wong", "liu", "park"}
            found = sum(1 for c in custodians if any(k in c.lower() for k in key_custodians))
            quality = min(1.0, found / 5)
            msg = f"✓ Litigation hold issued at step {self._step_count} (deadline: step {deadline_step}). "
            msg += f"Custodian coverage: {found}/5 key custodians identified."
            return 0.0, msg
        else:
            return -0.08, (
                f"⚠ Litigation hold issued at step {self._step_count}, but deadline was step {deadline_step}. "
                f"LATE — potential spoliation sanctions under CPC Order XI."
            )

    def _act_file_motion(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """File a court motion."""
        motion_type = _normalize(params.get("motion_type", ""))
        court = str(params.get("court", ""))
        arguments = str(params.get("arguments", ""))

        if not motion_type:
            return -0.02, "Missing 'motion_type' parameter."

        # Determine which event this addresses
        event_id = None
        if "tro" in motion_type or "restraining" in motion_type or "opposition" in motion_type:
            event_id = "EVENT-002"
        elif "mdl" in motion_type or "transfer" in motion_type or "consolidat" in motion_type:
            event_id = "EVENT-005"

        if event_id:
            self._findings["deadlines_met"][event_id] = {
                "step": self._step_count,
                "motion_type": motion_type,
                "court": court,
            }
            self._findings["actions_taken"].append({
                "event_id": event_id, "action": "file_motion", "step": self._step_count,
            })

            gt_deadline = CRISIS_GROUND_TRUTH["deadlines"].get(event_id, {})
            deadline_step = gt_deadline.get("deadline_step", 999)
            if self._step_count <= deadline_step:
                return 0.0, f"✓ Motion '{motion_type}' filed in {court or 'unspecified court'} at step {self._step_count}."
            else:
                return -0.08, f"⚠ Motion filed LATE at step {self._step_count} (deadline: step {deadline_step})."

        self._findings["actions_taken"].append({
            "event_id": "UNKNOWN", "action": "file_motion", "step": self._step_count,
        })
        return 0.0, f"Motion '{motion_type}' filed. Recorded."

    def _act_respond_discovery(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """Respond to a discovery request."""
        request_id = str(params.get("request_id", ""))
        response_type = _normalize(params.get("response_type", ""))
        objections = str(params.get("objections", ""))

        if not response_type:
            return -0.02, "Missing 'response_type' parameter."

        valid_responses = {"object", "produce", "partial_produce", "privilege_log"}
        if response_type not in valid_responses:
            return -0.02, f"Invalid response_type. Valid: {sorted(valid_responses)}"

        self._findings["deadlines_met"]["EVENT-003"] = {
            "step": self._step_count,
            "response_type": response_type,
            "objections": objections,
        }
        self._findings["actions_taken"].append({
            "event_id": "EVENT-003", "action": "respond_discovery", "step": self._step_count,
        })

        # Check if agent fell for the privilege trap
        if response_type == "produce" and not objections:
            self._ethical_alerts.append(
                "WARNING: Producing all documents without privilege objections may waive "
                "attorney-client privilege and work product protection for ALL related communications."
            )
            return -0.10, (
                "⚠ PRIVILEGE WARNING: Producing all requested documents without objection "
                "risks waiving privilege. Discovery Request 14 seeks privileged attorney-client "
                "communications. Request 15 seeks opinion work product. Consider objecting "
                "and producing a privilege log."
            )

        # Correct approach: object with privilege log
        if response_type in ("object", "partial_produce", "privilege_log"):
            privilege_terms = {"privilege", "professional communication", "sec 126", "sec 129", "evidence act", "adviser"}
            if any(t in objections.lower() for t in privilege_terms):
                return 0.0, (
                    "✓ Discovery response filed with appropriate privilege objections. "
                    "Privilege log will be prepared for withheld documents."
                )
        return 0.0, f"Discovery response filed: {response_type}. Recorded."

    def _act_assess_expert(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """Assess expert witness qualification under IEA Section 45."""
        expert_id = str(params.get("expert_id", "")).upper()
        qualification = str(params.get("qualification", ""))

        if not qualification:
            return -0.02, "Missing 'qualification' parameter."

        self._findings["expert_assessed"] = {
            "expert_id": expert_id,
            "qualification": qualification,
            "step": self._step_count,
        }
        self._findings["actions_taken"].append({
            "event_id": "EVENT-005", "action": "assess_expert", "step": self._step_count,
        })

        # Check IEA Sec 45 factors mentioned
        iea_factors = {"special skill", "science", "art", "foreign law",
                           "handwriting", "finger impressions", "relevant"}
        qual_lower = qualification.lower()
        factors_cited = sum(1 for f in iea_factors if f in qual_lower)

        if factors_cited >= 3:
            return 0.0, f"✓ Expert assessment with {factors_cited} IEA Sec 45 factors addressed. Strong qualification."
        elif factors_cited >= 1:
            return 0.0, f"Expert assessment with {factors_cited} IEA factors. Consider addressing more."
        return -0.02, "Expert assessment lacks IEA Sec 45 analysis. Address: special skill, science, art, or relevance."

    def _act_flag_adversarial(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """Flag an adversarial element in the crisis."""
        item_id = str(params.get("item_id", "")).upper()
        threat_type = str(params.get("threat_type", ""))
        explanation = str(params.get("explanation", ""))

        if not item_id or not threat_type:
            return -0.02, "Missing 'item_id' or 'threat_type' parameter."

        entry = {"item_id": item_id, "threat_type": threat_type, "explanation": explanation}
        existing_items = {a["item_id"] for a in self._findings.get("adversarial_flagged", [])}
        if item_id not in existing_items:
            self._findings["adversarial_flagged"].append(entry)

        gt_adversarial = set(CRISIS_GROUND_TRUTH.get("adversarial_items", []))
        if item_id in gt_adversarial:
            return 0.0, f"✓ Adversarial element correctly identified in {item_id}: {threat_type}."
        return -0.03, f"Adversarial flag for {item_id}. No known adversarial element."

    def _act_flag_ethical(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """Flag and resolve an ethical issue."""
        issue_type = str(params.get("issue_type", ""))
        affected_clients = str(params.get("affected_clients", ""))
        resolution = str(params.get("resolution", ""))

        if not issue_type:
            return -0.02, "Missing 'issue_type' parameter."

        # Determine event_id from issue context
        event_id = "EVENT-004"  # The only ethical event
        entry = {
            "event_id": event_id,
            "issue_type": issue_type,
            "affected_clients": affected_clients,
            "resolution": resolution,
        }
        existing = {e["event_id"] for e in self._findings.get("ethical_issues_flagged", [])}
        if event_id not in existing:
            self._findings["ethical_issues_flagged"].append(entry)

        self._findings["actions_taken"].append({
            "event_id": event_id, "action": "flag_ethical_issue", "step": self._step_count,
        })

        gt_ethical = set(CRISIS_GROUND_TRUTH.get("ethical_issues", []))
        if event_id in gt_ethical:
            return 0.0, (
                f"✓ Ethical issue flagged: {issue_type}. "
                f"Resolution: {resolution[:100]}. "
                f"This will be evaluated for compliance with BCI Rules."
            )
        return -0.03, f"Ethical issue flagged. Recorded."

    def _act_submit_terminal(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """Submit final analysis/review/triage for grading."""
        task_names = {
            "task_1": "Intake decisions",
            "task_2": "Privilege review",
            "task_3": "Crisis triage report",
        }
        return 0.0, f"{task_names.get(self._task_id, 'Analysis')} submitted for grading."

    # ── Internal Helpers ────────────────────────────────────────────────────

    def _init_findings(self, task_id: str) -> Dict[str, Any]:
        """Initialize empty findings map for the given task."""
        if task_id == "task_1":
            return {
                "conflicts_identified": [],
                "rule_citations": [],
                "decisions": {},
            }
        elif task_id == "task_2":
            return {
                "privilege_classifications": {},
                "waivers_identified": [],
                "exceptions_identified": [],
                "recommendations": {},
            }
        elif task_id == "task_3":
            return {
                "deadlines_met": {},
                "adversarial_flagged": [],
                "ethical_issues_flagged": [],
                "actions_taken": [],
                "expert_assessed": {},
                "discovery_responses": {},
            }
        return {}

    def _build_ground_truth(self, task_id: str) -> Dict[str, Any]:
        """Build the ground-truth lookup for grading."""
        gt: Dict[str, Any] = {}

        if task_id == "task_1":
            gt["conflict_pairs"] = {
                frozenset([cp.client_a, cp.client_b]) for cp in CONFLICT_PAIRS
            }
            gt["correct_decisions"] = CORRECT_DECISIONS
            gt["conflict_rules"] = {
                frozenset([cp.client_a, cp.client_b]): cp.rule
                for cp in CONFLICT_PAIRS
            }

        elif task_id == "task_2":
            gt["classifications"] = PRIVILEGE_GROUND_TRUTH
            gt["waiver_events"] = {w["doc_id"].upper() for w in WAIVER_EVENTS}

        elif task_id == "task_3":
            gt["deadlines"] = CRISIS_GROUND_TRUTH["deadlines"]
            gt["adversarial_items"] = CRISIS_GROUND_TRUTH["adversarial_items"]
            gt["ethical_issues"] = CRISIS_GROUND_TRUTH["ethical_issues"]
            gt["optimal_priority"] = CRISIS_GROUND_TRUTH["optimal_priority"]

        return gt

    def _run_grader(self) -> float:
        """Run the deterministic grader for the current task. Returns [0.0, 1.0]."""
        grader = GRADERS.get(self._task_id)
        if grader is None:
            return 0.0
        try:
            return grader(self._findings, self._ground_truth)
        except Exception:
            return 0.0

    def _make_observation(self, feedback: str) -> ObservationResponse:
        """Construct the observation response for the current state."""
        task_def = TASK_DEFINITIONS.get(self._task_id)
        docs: List[DocumentInfo] = []
        deadlines: List[DeadlineInfo] = []

        if self._task_id == "task_1":
            docs = [
                DocumentInfo(
                    index=i, title=f"{c.name} ({c.client_id})",
                    doc_type="client_intake", category=c.client_type,
                )
                for i, c in enumerate(CLIENTS)
            ]
        elif self._task_id == "task_2":
            docs = [
                DocumentInfo(
                    index=i, title=f"{d.title} ({d.doc_id})",
                    doc_type="legal_document", category="discovery",
                )
                for i, d in enumerate(PRIVILEGE_DOCUMENTS)
            ]
        elif self._task_id == "task_3":
            docs = [
                DocumentInfo(
                    index=i, title=f"{e.title} ({e.event_id})",
                    doc_type="crisis_event", category=e.event_type,
                )
                for i, e in enumerate(CRISIS_EVENTS)
            ]
            # Compute active deadlines
            for e in CRISIS_EVENTS:
                if e.deadline_step > 0:
                    remaining = max(0, e.deadline_step - self._step_count)
                    if remaining > 0 and e.event_id not in self._findings.get("deadlines_met", {}):
                        deadlines.append(DeadlineInfo(
                            deadline_id=e.event_id,
                            description=e.title,
                            steps_remaining=remaining,
                            consequence=f"Missed deadline for {e.event_id}",
                        ))

        task_actions = TASK_ACTIONS.get(self._task_id, set())

        return ObservationResponse(
            task_id=self._task_id,
            task_description=task_def.description if task_def else "",
            documents=docs,
            current_content=self._last_content,
            feedback=feedback,
            available_actions=sorted(task_actions),
            findings=self._findings,
            step_count=self._step_count,
            max_steps=self._max_steps,
            active_deadlines=deadlines,
            ethical_alerts=self._ethical_alerts,
        )
