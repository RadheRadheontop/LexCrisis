"""
Indian Legal Scenarios for LexCrisis Environment.
Ground-truth data for conflict screening (BCI), privilege review (IEA), and crisis triage (CPC/IEA).
"""

from typing import Any, Dict, List
from server.models import ClientIntake, ConflictPair, CrisisEvent, PrivilegedDocument

# ── TASK 1: Conflict Screening (BCI Rules) ──────────────────────────────────

CLIENTS = [
    ClientIntake(
        client_id="CLIENT-001", name="Ravi Sharma", client_type="plaintiff",
        summary="Claims severe liver damage from Veridex.",
        details="Took Veridex for 6 months. Medical records show acute hepatic failure.",
        relationships=["Sues NovaChem India", "Purchased from MedDistro Ltd"]
    ),
    ClientIntake(
        client_id="CLIENT-002", name="Priya Patel", client_type="plaintiff",
        summary="Class action representative.",
        details="Organizing 40 patients from Delhi with Veridex complications.",
        relationships=["Sues NovaChem India"]
    ),
    ClientIntake(
        client_id="CLIENT-003", name="NovaChem India Pvt Ltd", client_type="defendant",
        summary="Veridex manufacturer seeking defense.",
        details="Facing multiple consumer lawsuits in the National Consumer Disputes Redressal Commission (NCDRC).",
        relationships=["Manufactured Veridex"]
    ),
    ClientIntake(
        client_id="CLIENT-004", name="Dr. Anil Kapoor", client_type="prescriber",
        summary="Doctor seeking defense against malpractice claims.",
        details="Prescribed Veridex to Ravi Sharma. Now being sued alongside NovaChem.",
        relationships=["Prescribed to Ravi Sharma (CLIENT-001)", "Co-defendant with NovaChem"]
    ),
    ClientIntake(
        client_id="CLIENT-005", name="MedDistro Ltd", client_type="distributor",
        summary="Regional distributor of Veridex.",
        details="Seeking indemnification from NovaChem for defective batches.",
        relationships=["Distributed Veridex", "Adverse to NovaChem (CLIENT-003)"]
    ),
    ClientIntake(
        client_id="CLIENT-006", name="Arjun Mehta", client_type="plaintiff",
        summary="Individual plaintiff.",
        details="Has mild side effects, wants to join Priya Patel's class action.",
        relationships=["Sues NovaChem India"]
    ),
]

def get_client(client_id: str) -> ClientIntake | None:
    for c in CLIENTS:
        if c.client_id == client_id:
            return c
    return None

CONFLICT_PAIRS = [
    ConflictPair(client_a="CLIENT-001", client_b="CLIENT-003", rule="BCI Rule 33"),
    ConflictPair(client_a="CLIENT-002", client_b="CLIENT-003", rule="BCI Rule 33"),
    ConflictPair(client_a="CLIENT-003", client_b="CLIENT-004", rule="BCI Rule 22"), 
    ConflictPair(client_a="CLIENT-001", client_b="CLIENT-004", rule="BCI Rule 33"),
    ConflictPair(client_a="CLIENT-003", client_b="CLIENT-005", rule="BCI Rule 22"),
]

CORRECT_DECISIONS: Dict[str, str] = {
    "CLIENT-001": "accept", "CLIENT-002": "accept", "CLIENT-003": "decline", 
    "CLIENT-004": "decline", "CLIENT-005": "decline", "CLIENT-006": "accept",
}

# ── TASK 2: Privileged Document Review (IEA 1872) ──────────────────────────

PRIVILEGE_DOCUMENTS = [
    PrivilegedDocument(
        doc_id="DOC-001", title="Email: External Counsel Memo", doctrine="IEA Sec 126",
        content="From: General Counsel\nTo: CEO\nAttached is the legal memo from our advocates regarding the NCDRC filing strategy. They advise settling early."
    ),
    PrivilegedDocument(
        doc_id="DOC-002", title="Draft Affidavit (Internal)", doctrine="IEA Sec 129",
        content="Draft affidavit prepared by the litigation team in anticipation of the Delhi High Court hearing. Contains factual summaries and legal theories."
    ),
    PrivilegedDocument(
        doc_id="DOC-003", title="Counsel Notes on Strategy", doctrine="IEA Sec 126 & 129",
        content="Advocate's handwritten notes from the client briefing over Zoom. Outlines the defense weaknesses."
    ),
    PrivilegedDocument(
        doc_id="DOC-004", title="Sales Team Target Review", doctrine="None",
        content="From: VP Medical Sales\nTo: Sales Team\nVeridex Q3 targets were hit. Push the new marketing material to clinics in Mumbai."
    ),
    PrivilegedDocument(
        doc_id="DOC-005", title="Expert Witness Draft Report", doctrine="IEA Sec 129",
        content="Draft toxicology report commissioned by our advocates specifically for the upcoming trial."
    ),
    PrivilegedDocument(
        doc_id="DOC-006", title="Email: Deletion Directive", doctrine="IEA Sec 126 Proviso 1",
        content="From: In-House Legal\nTo: IT Admin\nDelete all internal emails mentioning 'Veridex liver toxicity' immediately before the court issues an order."
    ),
    PrivilegedDocument(
        doc_id="DOC-007", title="Press Release Draft", doctrine="At-Issue Waiver",
        content="Our advocates advised us that Veridex is completely safe under local regulations. We are releasing their exact legal analysis to the public tomorrow."
    ),
    PrivilegedDocument(
        doc_id="DOC-008", title="CDSCO Filing Certificate", doctrine="None",
        content="Official certificate of drug approval from the Central Drugs Standard Control Organisation."
    ),
]

def get_document(doc_id: str) -> PrivilegedDocument | None:
    for d in PRIVILEGE_DOCUMENTS:
        if d.doc_id == doc_id:
            return d
    return None

PRIVILEGE_GROUND_TRUTH: Dict[str, Dict[str, str]] = {
    "DOC-001": {"classification": "attorney_client", "action": "withhold", "doctrine": "iea sec 126"},
    "DOC-002": {"classification": "work_product", "action": "withhold", "doctrine": "iea sec 129"},
    "DOC-003": {"classification": "both", "action": "withhold", "doctrine": "iea sec 126"},
    "DOC-004": {"classification": "none", "action": "produce", "doctrine": ""},
    "DOC-005": {"classification": "work_product", "action": "withhold", "doctrine": "iea sec 129"},
    "DOC-006": {"classification": "waived", "action": "produce", "doctrine": "crime-fraud", "exception": "crime_fraud"},
    "DOC-007": {"classification": "waived", "action": "produce", "doctrine": "at-issue waiver", "exception": "at_issue"},
    "DOC-008": {"classification": "none", "action": "produce", "doctrine": ""},
}

WAIVER_EVENTS = [
    {"doc_id": "DOC-006", "type": "crime_fraud"},
    {"doc_id": "DOC-007", "type": "at_issue"},
]

# ── TASK 3: Crisis Triage (CPC / IEA) ───────────────────────────────────────

CRISIS_EVENTS = [
    CrisisEvent(
        event_id="EVENT-001", title="Imminent Discovery Order", event_type="legal_threat", deadline_step=8,
        content="The NCDRC is about to issue a formal directive to seize NovaChem's servers. Internal IT says auto-delete is still running on director emails."
    ),
    CrisisEvent(
        event_id="EVENT-002", title="Ex Parte Interim Injunction", event_type="court_order", deadline_step=10,
        content="Plaintiffs filed an application for an ex parte interim injunction in the Bombay High Court to halt all sales of Veridex."
    ),
    CrisisEvent(
        event_id="EVENT-003", title="Broad Discovery Notice", event_type="adversarial_action", deadline_step=15,
        content="Opposing counsel sent a Notice to Produce under Order XI Rule 15 of CPC demanding 'all communications regarding Veridex safety, including correspondence with your advocates'."
    ),
    CrisisEvent(
        event_id="EVENT-004", title="Partner's Former Client", event_type="ethical_issue", deadline_step=0,
        content="We just discovered that Senior Partner Mr. Singh previously defended MedDistro Ltd in a closely related pharmaceutical distribution case."
    ),
    CrisisEvent(
        event_id="EVENT-005", title="Transfer Petition to Supreme Court", event_type="court_filing", deadline_step=25,
        content="Over 40 separate suits filed in various High Courts across India. NovaChem needs to consolidate these to avoid contradictory judgments."
    ),
]

def get_event(event_id: str) -> CrisisEvent | None:
    for e in CRISIS_EVENTS:
        if e.event_id == event_id:
            return e
    return None

CRISIS_GROUND_TRUTH = {
    "deadlines": {
        "EVENT-001": {"deadline_step": 8, "required_action": "issue_litigation_hold"},
        "EVENT-002": {"deadline_step": 10, "required_action": "file_motion"},
        "EVENT-003": {"deadline_step": 15, "required_action": "respond_discovery"},
        "EVENT-005": {"deadline_step": 25, "required_action": "file_motion"},
    },
    "adversarial_items": ["EVENT-003"],
    "ethical_issues": ["EVENT-004"],
    "optimal_priority": ["EVENT-001", "EVENT-002", "EVENT-004", "EVENT-003", "EVENT-005"]
}
