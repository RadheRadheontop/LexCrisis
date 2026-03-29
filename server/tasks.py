"""
Task definitions and deterministic graders for the LexCrisis environment.

Three tasks with escalating difficulty. Each grader is a pure function
that takes (findings, ground_truth) and returns a float in [0.0, 1.0].

Graders produce VARYING scores across different agent strategies — no single
heuristic achieves the maximum score. Partial credit is awarded for each
correct sub-finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Set


# ── Utility Functions ───────────────────────────────────────────────────────


def _normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace. Handles None gracefully."""
    if not s:
        return ""
    return " ".join(str(s).lower().split())


def _f1(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _set_f1(predicted: set, actual: set) -> float:
    """Compute F1 between two sets."""
    if not predicted and not actual:
        return 1.0
    if not predicted or not actual:
        return 0.0
    tp = len(predicted & actual)
    precision = tp / len(predicted)
    recall = tp / len(actual)
    return _f1(precision, recall)


# ── Task Definitions ────────────────────────────────────────────────────────


@dataclass
class TaskDefinition:
    """Definition of a single task in the environment."""

    task_id: str
    name: str
    difficulty: str
    description: str
    max_steps: int
    relevant_actions: List[str]


TASK_DEFINITIONS: Dict[str, TaskDefinition] = {
    "task_1": TaskDefinition(
        task_id="task_1",
        name="Client Conflict Screening",
        difficulty="easy",
        description=(
            "You are a senior advocate at Sterling & Associates LLP in New Delhi. NovaChem India's drug "
            "Veridex has caused severe adverse effects in 47 patients. Six potential clients "
            "are seeking representation.\n\n"
            "Your objectives:\n"
            "1. Review each client's intake record to understand their situation.\n"
            "2. Check for conflicts of interest between client pairs under the Bar Council of India (BCI) Rules.\n"
            "3. For each identified conflict, cite the applicable BCI rule (e.g., 'BCI Rule 33').\n"
            "4. Make accept/decline decisions for each client based on conflict analysis.\n"
            "5. Submit your final intake decisions.\n\n"
            "Key BCI Rules (Part VI, Chapter II, Section II):\n"
            "- Rule 22: An advocate shall not appear for the opposite party if they have given advice to the other side.\n"
            "- Rule 33: An advocate who has advised a party in connection with a suit shall not act for a person whose interest is adverse.\n\n"
            "Available clients: CLIENT-001 through CLIENT-006."
        ),
        max_steps=15,
        relevant_actions=[
            "review_client", "check_conflict", "cite_rule",
            "accept_client", "decline_client", "submit_intake",
        ],
    ),
    "task_2": TaskDefinition(
        task_id="task_2",
        name="Privileged Document Review & Classification",
        difficulty="medium",
        description=(
            "You are reviewing 8 documents from litigation discovery in the NovaChem India "
            "product liability case before the NCDRC.\n\n"
            "Your objectives:\n"
            "1. Review each document's content carefully.\n"
            "2. Classify each document's privilege status:\n"
            "   - 'attorney_client' — Professional communication (IEA Sec 126)\n"
            "   - 'work_product' — Confidential communication with legal advisers (IEA Sec 129)\n"
            "   - 'both' — Protected by both Sections 126 and 129\n"
            "   - 'none' — Not privileged\n"
            "   - 'waived' — Privilege existed but was waived\n"
            "3. Cite the applicable doctrine for each classification (e.g., 'IEA Sec 126').\n"
            "4. Identify any waiver events (Crime-Fraud Exception under Proviso 1, at-issue waiver).\n"
            "5. Recommend an action for each document: withhold, produce, clawback, or redact.\n"
            "6. Submit your final privilege review.\n\n"
            "Key Doctrines (Indian Evidence Act, 1872):\n"
            "- Section 126: Protection of professional communications with advocates.\n"
            "- Section 129: Confidential communications with legal advisers.\n"
            "- Section 126 Proviso 1: Communications made in furtherance of any illegal purpose are NOT protected (Crime-fraud exception).\n"
            "- At-issue waiver: Voluntarily deploying privileged material in pleadings waives protection.\n\n"
            "Available documents: DOC-001 through DOC-008."
        ),
        max_steps=25,
        relevant_actions=[
            "review_document", "classify_privilege", "identify_waiver",
            "identify_exception", "recommend_action", "submit_review",
        ],
    ),
    "task_3": TaskDefinition(
        task_id="task_3",
        name="Multi-Front Crisis Triage",
        difficulty="hard",
        description=(
            "You are managing a multi-front legal crisis in the NovaChem India litigation. "
            "Five crisis events require your attention across the High Courts and NCDRC.\n\n"
            "Your objectives:\n"
            "1. Review each crisis event to understand the situation and deadline.\n"
            "2. Issue a litigation hold to prevent evidence spoliation.\n"
            "3. File necessary court motions (Interim Injunction opposition, CPC Sec 25 Transfer Petition).\n"
            "4. Respond to adversarial discovery under CPC Order XI WITHOUT waiving privilege.\n"
            "5. Identify and flag adversarial tactics by opposing counsel.\n"
            "6. Detect and properly handle ethical issues (emerging conflicts under BCI Rules).\n"
            "7. Assess expert witness qualification under IEA Section 45.\n"
            "8. Submit your triage report.\n\n"
            "WARNING: This task contains adversarial elements designed to mislead you. "
            "Some discovery requests are designed to trick you into waiving privilege. "
            "Some events contain ethical traps where the 'obvious' action is wrong.\n\n"
            "Key Rules:\n"
            "- CPC Order XI: Discovery and Inspection of Documents\n"
            "- BCI Rule 33: Former client conflicts (duty not to act against former client in same matter)\n"
            "- CPC Section 25: Power of Supreme Court to transfer suits\n"
            "- IEA Section 45: Opinions of experts\n\n"
            "Available events: EVENT-001 through EVENT-005."
        ),
        max_steps=40,
        relevant_actions=[
            "review_event", "issue_litigation_hold", "file_motion",
            "respond_discovery", "assess_expert", "flag_adversarial",
            "flag_ethical_issue", "submit_triage",
        ],
    ),
}


# ── Grader Functions ────────────────────────────────────────────────────────


def grade_task1(findings: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Grade Task 1: Client Conflict Screening.

    Scoring breakdown:
        Conflict pair identification (F1):    0.40
        Accept/decline decision accuracy:     0.35
        BCI rule citation accuracy:           0.25

    Returns: float in [0.0, 1.0]
    """
    score = 0.0

    # ── Conflict pair identification (0.40) ─────────────────────────────
    gt_pairs: set = ground_truth.get("conflict_pairs", set())
    pred_conflicts = findings.get("conflicts_identified", [])

    # Normalize predicted pairs to frozensets for comparison
    pred_pairs: set = set()
    for conflict in pred_conflicts:
        a = str(conflict.get("client_a", "")).upper()
        b = str(conflict.get("client_b", "")).upper()
        if a and b:
            pred_pairs.add(frozenset([a, b]))

    conflict_f1 = _set_f1(pred_pairs, gt_pairs)
    score += 0.40 * conflict_f1

    # ── Accept/decline accuracy (0.35) ──────────────────────────────────
    gt_decisions: Dict[str, str] = ground_truth.get("correct_decisions", {})
    pred_decisions: Dict[str, str] = findings.get("decisions", {})

    if gt_decisions:
        correct = 0
        for client_id, gt_decision in gt_decisions.items():
            pred = _normalize(pred_decisions.get(client_id, ""))
            if pred == _normalize(gt_decision):
                correct += 1
        decision_accuracy = correct / len(gt_decisions)
        score += 0.35 * decision_accuracy

    # ── Rule citation accuracy (0.25) ───────────────────────────────────
    gt_rules: Dict[frozenset, str] = ground_truth.get("conflict_rules", {})
    pred_rules_list = findings.get("rule_citations", [])

    if gt_rules:
        correct_rules = 0
        for citation in pred_rules_list:
            a = str(citation.get("client_a", "")).upper()
            b = str(citation.get("client_b", "")).upper()
            rule = _normalize(citation.get("rule", ""))
            pair = frozenset([a, b])
            if pair in gt_rules:
                gt_rule = _normalize(gt_rules[pair])
                # Accept partial matches (e.g., "1.7" matches "1.7(a)(1)")
                if rule == gt_rule or gt_rule.startswith(rule) or rule.startswith(gt_rule.split("(")[0]):
                    correct_rules += 1
        rule_accuracy = correct_rules / len(gt_rules) if gt_rules else 0.0
        score += 0.25 * min(1.0, rule_accuracy)

    return round(min(score, 1.0), 4)


def grade_task2(findings: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Grade Task 2: Privileged Document Review.

    Scoring breakdown:
        Privilege classification accuracy:    0.45
        Waiver event identification (F1):     0.30
        Doctrine citation quality:            0.25

    Returns: float in [0.0, 1.0]
    """
    score = 0.0

    gt_classifications = ground_truth.get("classifications", {})
    pred_classifications = findings.get("privilege_classifications", {})

    # ── Classification accuracy (0.45) ──────────────────────────────────
    if gt_classifications:
        correct = 0
        partial = 0
        for doc_id, gt_data in gt_classifications.items():
            pred = pred_classifications.get(doc_id, {})
            pred_class = _normalize(pred.get("classification", ""))
            gt_class = _normalize(gt_data.get("classification", ""))

            if pred_class == gt_class:
                correct += 1
            elif (pred_class in ("attorney_client", "work_product", "both") and
                  gt_class in ("attorney_client", "work_product", "both")):
                # Partial credit for getting privilege right but wrong sub-type
                partial += 0.5

        accuracy = (correct + partial) / len(gt_classifications)
        score += 0.45 * accuracy

    # ── Waiver identification (0.30) ────────────────────────────────────
    gt_waivers = ground_truth.get("waiver_events", set())
    pred_waivers_raw = findings.get("waivers_identified", [])
    pred_waivers: set = {str(w.get("doc_id", "")).upper() for w in pred_waivers_raw}

    waiver_f1 = _set_f1(pred_waivers, gt_waivers)
    score += 0.30 * waiver_f1

    # ── Doctrine quality (0.25) ─────────────────────────────────────────
    if gt_classifications:
        doctrine_score = 0.0
        for doc_id, gt_data in gt_classifications.items():
            pred = pred_classifications.get(doc_id, {})
            pred_doctrine = _normalize(pred.get("doctrine", ""))
            gt_doctrine = _normalize(gt_data.get("doctrine", ""))

            if not gt_doctrine:
                continue

            # Check for key terms in doctrine citation
            gt_terms = set(gt_doctrine.split())
            pred_terms = set(pred_doctrine.split())
            # Key legal terms that indicate understanding
            key_terms = {"iea", "sec", "126", "129", "evidence", "act", "crime-fraud", "crime", "fraud",
                         "waiver", "professional", "communication", "adviser", "advocate", "at-issue"}
            relevant_gt = gt_terms & key_terms
            relevant_pred = pred_terms & key_terms
            if relevant_gt:
                overlap = len(relevant_gt & relevant_pred) / len(relevant_gt)
                doctrine_score += overlap

        doctrine_accuracy = doctrine_score / len(gt_classifications) if gt_classifications else 0.0
        score += 0.25 * min(1.0, doctrine_accuracy)

    return round(min(score, 1.0), 4)


def grade_task3(findings: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Grade Task 3: Multi-Front Crisis Triage.

    Scoring breakdown:
        Deadline compliance:                  0.25
        Ethical compliance:                   0.25
        Adversarial detection:                0.25
        Action sequence quality:              0.25

    Returns: float in [0.0, 1.0]
    """
    score = 0.0

    # ── Deadline compliance (0.25) ──────────────────────────────────────
    gt_deadlines = ground_truth.get("deadlines", {})
    met_deadlines = findings.get("deadlines_met", {})

    if gt_deadlines:
        compliant = 0
        for event_id, info in gt_deadlines.items():
            if event_id in met_deadlines:
                met_step = met_deadlines[event_id].get("step", float("inf"))
                deadline_step = info.get("deadline_step", 0)
                if met_step <= deadline_step:
                    compliant += 1
                else:
                    # Partial credit: action taken but after deadline
                    compliant += 0.3
        score += 0.25 * (compliant / len(gt_deadlines))

    # ── Ethical compliance (0.25) ───────────────────────────────────────
    gt_ethical = set(ground_truth.get("ethical_issues", []))
    pred_ethical_raw = findings.get("ethical_issues_flagged", [])
    pred_ethical: set = {str(e.get("event_id", "")).upper() for e in pred_ethical_raw}

    if gt_ethical:
        ethical_f1 = _set_f1(pred_ethical, gt_ethical)
        score += 0.25 * ethical_f1

        # Bonus: check if resolution is appropriate
        for issue in pred_ethical_raw:
            event_id = str(issue.get("event_id", "")).upper()
            if event_id in gt_ethical:
                resolution = _normalize(issue.get("resolution", ""))
                # Look for key terms indicating proper handling
                good_terms = {"consent", "withdraw", "disclose", "bci rule", "rule 33", "former client"}
                if any(term in resolution for term in good_terms):
                    score += 0.05  # Small bonus for good resolution
    else:
        score += 0.25  # No ethical issues = full marks

    # ── Adversarial detection (0.25) ────────────────────────────────────
    gt_adversarial = set(ground_truth.get("adversarial_items", []))
    pred_adversarial_raw = findings.get("adversarial_flagged", [])
    pred_adversarial: set = {str(a.get("item_id", "")).upper() for a in pred_adversarial_raw}

    adversarial_f1 = _set_f1(pred_adversarial, gt_adversarial)
    score += 0.25 * adversarial_f1

    # ── Action sequence quality (0.25) ──────────────────────────────────
    gt_priority = ground_truth.get("optimal_priority", [])
    pred_actions = findings.get("actions_taken", [])

    if gt_priority and pred_actions:
        # Check if agent addressed events in roughly correct priority order
        pred_event_order = []
        for action in pred_actions:
            eid = str(action.get("event_id", "")).upper()
            if eid and eid not in pred_event_order:
                pred_event_order.append(eid)

        # Compute Kendall tau-like score
        correct_orderings = 0
        total_comparisons = 0
        for i, ev_a in enumerate(gt_priority):
            for j, ev_b in enumerate(gt_priority[i + 1:], i + 1):
                ea_upper = ev_a.upper()
                eb_upper = ev_b.upper()
                if ea_upper in pred_event_order and eb_upper in pred_event_order:
                    total_comparisons += 1
                    if pred_event_order.index(ea_upper) < pred_event_order.index(eb_upper):
                        correct_orderings += 1

        ordering_quality = correct_orderings / total_comparisons if total_comparisons > 0 else 0.0
        # Also reward coverage
        coverage = len(set(pred_event_order) & set(e.upper() for e in gt_priority)) / len(gt_priority)
        score += 0.25 * (0.5 * ordering_quality + 0.5 * coverage)

    return round(min(max(score, 0.0), 1.0), 4)


# ── Grader Registry ─────────────────────────────────────────────────────────

GRADERS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], float]] = {
    "task_1": grade_task1,
    "task_2": grade_task2,
    "task_3": grade_task3,
}
