"""Task data and scripted reference policies for LexCrisis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


def normalize(text: Any) -> str:
    """Normalize free-form text for deterministic comparison."""

    if text is None:
        return ""
    return " ".join(str(text).strip().lower().split())


@dataclass(frozen=True)
class ClientProfile:
    client_id: str
    name: str
    client_type: str
    summary: str
    details: str
    relationships: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PrivilegeDocument:
    doc_id: str
    title: str
    doctrine: str
    content: str


@dataclass(frozen=True)
class CrisisEvent:
    event_id: str
    title: str
    event_type: str
    deadline_step: int
    consequence: str
    content: str


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    name: str
    difficulty: str
    description: str
    max_steps: int
    relevant_actions: List[str]


CLIENTS: List[ClientProfile] = [
    ClientProfile(
        client_id="CLIENT-001",
        name="Ravi Sharma",
        client_type="plaintiff",
        summary="Severe liver injury claimant tied to the Veridex litigation wave.",
        details=(
            "Ravi took Veridex for six months and now alleges acute hepatic failure. "
            "He wants Sterling & Associates to pursue NovaChem India and related entities."
        ),
        relationships=["Adverse to NovaChem India", "Purchased through MedDistro Ltd"],
    ),
    ClientProfile(
        client_id="CLIENT-002",
        name="Priya Patel",
        client_type="plaintiff",
        summary="Potential class representative coordinating dozens of claimants.",
        details=(
            "Priya is organizing 40+ patients with similar injuries and wants coordinated "
            "representation against NovaChem India."
        ),
        relationships=["Adverse to NovaChem India"],
    ),
    ClientProfile(
        client_id="CLIENT-003",
        name="NovaChem India Pvt Ltd",
        client_type="defendant",
        summary="Drug manufacturer seeking defense counsel.",
        details=(
            "NovaChem manufactured Veridex and faces parallel product liability matters, "
            "consumer claims, and reputational exposure."
        ),
        relationships=["Adverse to patient plaintiffs", "Potential indemnity dispute with MedDistro Ltd"],
    ),
    ClientProfile(
        client_id="CLIENT-004",
        name="Dr. Anil Kapoor",
        client_type="prescriber",
        summary="Physician co-defendant in plaintiff suits.",
        details=(
            "Dr. Kapoor prescribed Veridex to Ravi Sharma and is being sued alongside "
            "NovaChem for malpractice and failure-to-warn theories."
        ),
        relationships=["Aligned with NovaChem in some filings", "Adverse to Ravi Sharma"],
    ),
    ClientProfile(
        client_id="CLIENT-005",
        name="MedDistro Ltd",
        client_type="distributor",
        summary="Distributor seeking indemnity from NovaChem.",
        details=(
            "MedDistro distributed the affected batch and is exploring indemnification "
            "claims against NovaChem while defending separate regulatory inquiries."
        ),
        relationships=["Potentially adverse to NovaChem India"],
    ),
    ClientProfile(
        client_id="CLIENT-006",
        name="Arjun Mehta",
        client_type="plaintiff",
        summary="Milder injury claimant wanting to join plaintiff-side litigation.",
        details=(
            "Arjun reports side effects and wants plaintiff representation. "
            "He is not independently adverse to existing plaintiff claimants."
        ),
        relationships=["Adverse to NovaChem India"],
    ),
]

CONFLICT_RULES: Dict[frozenset[str], str] = {
    frozenset(("CLIENT-001", "CLIENT-003")): "BCI Rule 33",
    frozenset(("CLIENT-002", "CLIENT-003")): "BCI Rule 33",
    frozenset(("CLIENT-003", "CLIENT-004")): "BCI Rule 22",
    frozenset(("CLIENT-001", "CLIENT-004")): "BCI Rule 33",
    frozenset(("CLIENT-003", "CLIENT-005")): "BCI Rule 22",
}

CONFLICT_DECISIONS: Dict[str, str] = {
    "CLIENT-001": "accept",
    "CLIENT-002": "accept",
    "CLIENT-003": "decline",
    "CLIENT-004": "decline",
    "CLIENT-005": "decline",
    "CLIENT-006": "accept",
}

PRIVILEGE_DOCUMENTS: List[PrivilegeDocument] = [
    PrivilegeDocument(
        doc_id="DOC-001",
        title="External Counsel Settlement Memo",
        doctrine="IEA Section 126",
        content=(
            "From: General Counsel\nTo: CEO\nAttached is outside counsel's memo on settlement "
            "strategy for the National Consumer Disputes matter. Counsel recommends early resolution."
        ),
    ),
    PrivilegeDocument(
        doc_id="DOC-002",
        title="Draft Hearing Affidavit",
        doctrine="IEA Section 129",
        content=(
            "Internal litigation team draft prepared in anticipation of the Delhi High Court hearing. "
            "Contains factual chronologies, legal theories, and witness preparation notes."
        ),
    ),
    PrivilegeDocument(
        doc_id="DOC-003",
        title="Advocate Strategy Notes",
        doctrine="IEA Sections 126 and 129",
        content=(
            "Handwritten advocate notes from a client briefing. The notes discuss weaknesses in the "
            "defense narrative and privileged litigation strategy."
        ),
    ),
    PrivilegeDocument(
        doc_id="DOC-004",
        title="Sales Territory Review",
        doctrine="None",
        content=(
            "Quarterly sales performance review covering Veridex distribution targets, "
            "channel performance, and field-marketing plans."
        ),
    ),
    PrivilegeDocument(
        doc_id="DOC-005",
        title="Draft Toxicology Report",
        doctrine="IEA Section 129",
        content=(
            "Draft expert report commissioned by counsel for upcoming trial testimony. "
            "It includes attorney comments and work product annotations."
        ),
    ),
    PrivilegeDocument(
        doc_id="DOC-006",
        title="Deletion Directive Email",
        doctrine="Crime-fraud exception",
        content=(
            "From: In-House Legal\nTo: IT Admin\nDelete any internal emails mentioning "
            "'Veridex liver toxicity' before the court issues a preservation order."
        ),
    ),
    PrivilegeDocument(
        doc_id="DOC-007",
        title="Public Press Release Draft",
        doctrine="At-issue waiver",
        content=(
            "Draft press release quoting counsel's exact legal conclusion that Veridex is safe "
            "and compliant under all applicable regulations."
        ),
    ),
    PrivilegeDocument(
        doc_id="DOC-008",
        title="Drug Approval Certificate",
        doctrine="None",
        content=(
            "Official Central Drugs Standard Control Organisation certificate "
            "memorializing the original approval of Veridex."
        ),
    ),
]

PRIVILEGE_GROUND_TRUTH: Dict[str, Dict[str, str]] = {
    "DOC-001": {
        "classification": "attorney_client",
        "doctrine": "IEA Section 126",
        "action": "withhold",
        "exception": "none",
    },
    "DOC-002": {
        "classification": "work_product",
        "doctrine": "IEA Section 129",
        "action": "withhold",
        "exception": "none",
    },
    "DOC-003": {
        "classification": "both",
        "doctrine": "IEA Sections 126 and 129",
        "action": "withhold",
        "exception": "none",
    },
    "DOC-004": {
        "classification": "none",
        "doctrine": "",
        "action": "produce",
        "exception": "none",
    },
    "DOC-005": {
        "classification": "work_product",
        "doctrine": "IEA Section 129",
        "action": "withhold",
        "exception": "none",
    },
    "DOC-006": {
        "classification": "waived",
        "doctrine": "Crime-fraud exception",
        "action": "produce",
        "exception": "crime_fraud",
    },
    "DOC-007": {
        "classification": "waived",
        "doctrine": "At-issue waiver",
        "action": "produce",
        "exception": "at_issue",
    },
    "DOC-008": {
        "classification": "none",
        "doctrine": "",
        "action": "produce",
        "exception": "none",
    },
}

WAIVER_EVENTS: Dict[str, str] = {
    "DOC-006": "crime_fraud",
    "DOC-007": "at_issue",
}

CRISIS_EVENTS: List[CrisisEvent] = [
    CrisisEvent(
        event_id="EVENT-001",
        title="Imminent Preservation Failure",
        event_type="spoliation_risk",
        deadline_step=6,
        consequence="Potential sanctions for spoliation and lost ESI.",
        content=(
            "Auto-delete remains active on senior custodians' mailboxes while the tribunal is preparing "
            "a preservation order. A litigation hold must go out immediately."
        ),
    ),
    CrisisEvent(
        event_id="EVENT-002",
        title="Emergency Sales Injunction",
        event_type="court_motion",
        deadline_step=9,
        consequence="A sales halt could trigger revenue loss and class-action momentum.",
        content=(
            "Plaintiffs filed for an ex parte injunction to halt Veridex sales. Counsel needs an immediate "
            "response strategy and motion work."
        ),
    ),
    CrisisEvent(
        event_id="EVENT-003",
        title="Aggressive Discovery Request",
        event_type="discovery",
        deadline_step=12,
        consequence="Producing privileged communications could waive privilege across the matter.",
        content=(
            "Opposing counsel demanded all communications about Veridex safety, including correspondence "
            "with advocates, under a broad Order XI request."
        ),
    ),
    CrisisEvent(
        event_id="EVENT-004",
        title="Former Client Conflict",
        event_type="ethics",
        deadline_step=0,
        consequence="Failure to address the issue risks a disqualification fight and ethics breach.",
        content=(
            "The lead partner previously represented MedDistro in a related distribution dispute. "
            "The issue must be surfaced and resolved before strategic work continues."
        ),
    ),
    CrisisEvent(
        event_id="EVENT-005",
        title="Forum Consolidation Decision",
        event_type="coordination",
        deadline_step=18,
        consequence="Fragmented proceedings increase cost and risk inconsistent rulings.",
        content=(
            "Parallel suits across jurisdictions are multiplying. The team needs to assess transfer and "
            "consolidation options while preserving expert strategy."
        ),
    ),
]

CRISIS_GROUND_TRUTH: Dict[str, Any] = {
    "deadlines": {
        "EVENT-001": {"deadline_step": 6, "required_action": "issue_litigation_hold"},
        "EVENT-002": {"deadline_step": 9, "required_action": "file_motion"},
        "EVENT-003": {"deadline_step": 12, "required_action": "respond_discovery"},
        "EVENT-005": {"deadline_step": 18, "required_action": "file_motion"},
    },
    "ethical_issues": {"EVENT-004"},
    "adversarial_items": {"EVENT-003"},
    "priority_order": ["EVENT-001", "EVENT-004", "EVENT-002", "EVENT-003", "EVENT-005"],
}

TASK_DEFINITIONS: Dict[str, TaskDefinition] = {
    "task_1": TaskDefinition(
        task_id="task_1",
        name="Conflict-Safe Client Intake",
        difficulty="easy",
        description=(
            "A law firm intake team must decide which prospective clients to accept in a fast-moving "
            "product-liability crisis. The current scenario uses a pharmaceutical case study, but the "
            "workflow applies more broadly to regulated-industry litigation. The agent needs to identify conflicts of interest, "
            "cite the right Bar Council of India rule, and make intake decisions that a real conflicts team "
            "would stand behind."
        ),
        max_steps=16,
        relevant_actions=[
            "review_client",
            "check_conflict",
            "cite_rule",
            "accept_client",
            "decline_client",
            "submit_intake",
        ],
    ),
    "task_2": TaskDefinition(
        task_id="task_2",
        name="Privilege Review Under Litigation Pressure",
        difficulty="medium",
        description=(
            "An in-house legal operations team must review eight documents before production. The agent "
            "needs to classify privilege correctly, spot waiver or exception risks, and decide whether each "
            "document should be produced, withheld, clawed back, or redacted."
        ),
        max_steps=24,
        relevant_actions=[
            "review_document",
            "classify_privilege",
            "identify_waiver",
            "identify_exception",
            "recommend_action",
            "submit_review",
        ],
    ),
    "task_3": TaskDefinition(
        task_id="task_3",
        name="Litigation Incident Command",
        difficulty="hard",
        description=(
            "This task simulates a legal operations incident room for a live product-liability crisis. The "
            "current scenario is pharmaceutical, but the trade-offs are intended to transfer to broader "
            "regulated litigation work. The "
            "agent must balance preservation deadlines, an ethics conflict, adversarial discovery, motion "
            "practice, and expert strategy. Solving it well requires prioritization under time pressure and "
            "avoiding hidden privilege-waiver traps."
        ),
        max_steps=20,
        relevant_actions=[
            "review_event",
            "issue_litigation_hold",
            "file_motion",
            "respond_discovery",
            "assess_expert",
            "flag_adversarial",
            "flag_ethical_issue",
            "submit_triage",
        ],
    ),
}

TASK_ACTIONS: Dict[str, List[str]] = {
    task_id: definition.relevant_actions for task_id, definition in TASK_DEFINITIONS.items()
}

TERMINAL_ACTIONS = {
    "task_1": "submit_intake",
    "task_2": "submit_review",
    "task_3": "submit_triage",
}


def get_client(client_id: str) -> Optional[ClientProfile]:
    return next((client for client in CLIENTS if client.client_id == client_id), None)


def get_document(doc_id: str) -> Optional[PrivilegeDocument]:
    return next((document for document in PRIVILEGE_DOCUMENTS if document.doc_id == doc_id), None)


def get_event(event_id: str) -> Optional[CrisisEvent]:
    return next((event for event in CRISIS_EVENTS if event.event_id == event_id), None)


def first_matching(sequence: Iterable[str], keywords: Iterable[str]) -> bool:
    """Return True if any keyword appears in any item of the sequence."""

    normalized_items = [normalize(item) for item in sequence]
    return any(keyword in item for keyword in keywords for item in normalized_items)


SCRIPTED_BASELINES: Dict[str, List[Dict[str, Any]]] = {
    "task_1": [
        {"action_type": "review_client", "parameters": {"client_id": "CLIENT-001"}},
        {"action_type": "review_client", "parameters": {"client_id": "CLIENT-003"}},
        {
            "action_type": "check_conflict",
            "parameters": {"client_a": "CLIENT-001", "client_b": "CLIENT-003"},
        },
        {
            "action_type": "cite_rule",
            "parameters": {
                "client_a": "CLIENT-001",
                "client_b": "CLIENT-003",
                "rule": "BCI Rule 33",
            },
        },
        {"action_type": "accept_client", "parameters": {"client_id": "CLIENT-001"}},
        {"action_type": "accept_client", "parameters": {"client_id": "CLIENT-002"}},
        {"action_type": "decline_client", "parameters": {"client_id": "CLIENT-003"}},
        {"action_type": "decline_client", "parameters": {"client_id": "CLIENT-004"}},
        {"action_type": "decline_client", "parameters": {"client_id": "CLIENT-005"}},
        {"action_type": "accept_client", "parameters": {"client_id": "CLIENT-006"}},
        {"action_type": "submit_intake", "parameters": {}},
    ],
    "task_2": [
        {"action_type": "review_document", "parameters": {"doc_id": "DOC-001"}},
        {
            "action_type": "classify_privilege",
            "parameters": {
                "doc_id": "DOC-001",
                "classification": "attorney_client",
                "doctrine": "IEA Section 126",
            },
        },
        {
            "action_type": "recommend_action",
            "parameters": {"doc_id": "DOC-001", "action": "withhold"},
        },
        {"action_type": "review_document", "parameters": {"doc_id": "DOC-006"}},
        {
            "action_type": "classify_privilege",
            "parameters": {
                "doc_id": "DOC-006",
                "classification": "waived",
                "doctrine": "Crime-fraud exception",
            },
        },
        {
            "action_type": "identify_waiver",
            "parameters": {
                "doc_id": "DOC-006",
                "waiver_type": "crime_fraud",
                "explanation": "Illegal-purpose communication is not protected.",
            },
        },
        {
            "action_type": "identify_exception",
            "parameters": {
                "doc_id": "DOC-006",
                "exception_type": "crime_fraud",
                "explanation": "Instruction to destroy evidence triggers the exception.",
            },
        },
        {
            "action_type": "recommend_action",
            "parameters": {"doc_id": "DOC-006", "action": "produce"},
        },
        {"action_type": "review_document", "parameters": {"doc_id": "DOC-007"}},
        {
            "action_type": "identify_waiver",
            "parameters": {
                "doc_id": "DOC-007",
                "waiver_type": "at_issue",
                "explanation": "The draft quotes legal advice for public deployment.",
            },
        },
        {"action_type": "submit_review", "parameters": {}},
    ],
    "task_3": [
        {"action_type": "review_event", "parameters": {"event_id": "EVENT-001"}},
        {
            "action_type": "issue_litigation_hold",
            "parameters": {
                "scope": "All Veridex custodial mail, chat, and product safety files.",
                "custodians": ["Morton", "Ames", "Wong", "Liu", "Park"],
            },
        },
        {"action_type": "review_event", "parameters": {"event_id": "EVENT-004"}},
        {
            "action_type": "flag_ethical_issue",
            "parameters": {
                "issue_type": "former_client_conflict",
                "affected_clients": "MedDistro Ltd",
                "resolution": "Disclose the prior representation, screen the partner, and withdraw if consent is unavailable under BCI Rule 33.",
            },
        },
        {"action_type": "review_event", "parameters": {"event_id": "EVENT-002"}},
        {
            "action_type": "file_motion",
            "parameters": {
                "motion_type": "injunction_opposition",
                "court": "Bombay High Court",
                "arguments": "Plaintiffs cannot justify emergency relief before a full evidentiary record.",
            },
        },
        {"action_type": "review_event", "parameters": {"event_id": "EVENT-003"}},
        {
            "action_type": "flag_adversarial",
            "parameters": {
                "item_id": "EVENT-003",
                "threat_type": "privilege_trap",
                "explanation": "The request targets advocate communications to trigger waiver.",
            },
        },
        {
            "action_type": "respond_discovery",
            "parameters": {
                "request_id": "REQ-14",
                "response_type": "privilege_log",
                "objections": "Object to advocate communications as privileged under IEA Sections 126 and 129 and produce a privilege log for withheld items.",
            },
        },
        {
            "action_type": "assess_expert",
            "parameters": {
                "expert_id": "EXP-TOX-01",
                "qualification": "Special skill in toxicology and regulatory science, with relevant expertise under IEA Section 45.",
            },
        },
        {"action_type": "submit_triage", "parameters": {}},
    ],
}
