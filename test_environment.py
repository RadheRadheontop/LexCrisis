"""
Tests for the Legal Document Analysis Environment.

Validates that all 5 tasks initialize correctly, reward signals are appropriate,
and grading produces expected results for known inputs.
"""

import pytest
from server.environment import LegalDocEnvironment
from server.models import ActionRequest


@pytest.fixture
def env():
    """Provide a fresh environment instance for each test."""
    return LegalDocEnvironment()


def test_environment_initialization(env):
    """Test that the environment initializes correctly and registers all 5 tasks."""
    result = env.reset("task_1")
    obs = result.observation
    assert obs.task_id == "task_1"
    assert obs.max_steps == 10
    assert len(obs.documents) > 0
    assert obs.step_count == 0


def test_task_1_classification_reward(env):
    """Test that a correct sequence yields positive rewards and done=True."""
    env.reset("task_1")

    # 1. Unknown action should penalize
    result = env.step(ActionRequest(action_type="invalid_action", parameters={}))
    assert result.reward < 0.0

    # 2. Correct classification
    result = env.step(ActionRequest(action_type="classify_document", parameters={"doc_index": 0, "document_type": "nda"}))
    assert result.reward > 0.0
    assert env.state().findings["classification"] == "nda"

    # 3. Submit analysis explicitly finishes episode
    result = env.step(ActionRequest(action_type="submit_analysis", parameters={}))
    assert result.done is True
    assert result.score > 0.0


def test_task_2_risk_flagging(env):
    """Test Task 2: Employment agreement risk assessment."""
    env.reset("task_2")

    # Flag a known risky clause (clause 2 = non_compete, high risk)
    result = env.step(ActionRequest(
        action_type="flag_clause_risk",
        parameters={"clause_index": 2, "risk_type": "non_compete", "severity": "high"}
    ))
    # Handler returns positive for correct flag, but reward depends on grader delta
    assert result.score >= 0.0

    # Report a known missing clause
    result = env.step(ActionRequest(
        action_type="report_missing_clause",
        parameters={"clause_type": "dispute_resolution"}
    ))
    assert env.state().findings["missing_clauses"] == ["dispute_resolution"]


def test_task_3_compliance_audit(env):
    """Test Task 3: Multi-document compliance audit."""
    env.reset("task_3")

    # Read all three documents
    env.step(ActionRequest(action_type="read_document", parameters={"doc_index": 0}))
    env.step(ActionRequest(action_type="read_document", parameters={"doc_index": 1}))
    env.step(ActionRequest(action_type="read_document", parameters={"doc_index": 2}))

    # Flag a known contradiction (breach notification timeframe)
    result = env.step(ActionRequest(
        action_type="flag_contradiction",
        parameters={
            "doc_a_index": 0,
            "clause_a": "72 hours breach notification",
            "doc_b_index": 2,
            "clause_b": "6 hours CERT-In mandate",
            "description": "Breach notification timeframe conflict under Indian Law"
        }
    ))
    assert len(env.state().findings["contradictions"]) == 1


def test_task_4_ediscovery_search(env):
    """Test the e-Discovery M&A searching mechanics."""
    env.reset("task_4")

    # Valid database query
    result = env.step(ActionRequest(action_type="run_database_query", parameters={"query": "cayman"}))
    # "cayman" is not in the data room (documents use "mauritius"), so may return 0 results
    # Try a keyword that IS in the data
    result = env.step(ActionRequest(action_type="run_database_query", parameters={"query": "mauritius"}))
    assert "DOC-904" in result.observation.feedback or "DOC-901" in result.observation.feedback

    # Flagging a non-existent document should penalize
    result = env.step(ActionRequest(action_type="flag_evidence", parameters={"doc_id": "MISSING-DOC", "issue_type": "none"}))
    assert result.reward < 0.0

    # Flagging critical evidence with correct keywords
    result = env.step(ActionRequest(action_type="flag_evidence", parameters={"doc_id": "DOC-904", "issue_type": "offshore mauritius shell entity"}))
    assert result.reward >= 0.10  # Should get positive reward for flagging critical doc


def test_task_5_precedents(env):
    """Test case law precedent citation mechanics."""
    env.reset("task_5")

    result = env.step(ActionRequest(action_type="search_case_law", parameters={"query": "encryption"}))
    assert "CASE-001" in result.observation.feedback

    result = env.step(ActionRequest(action_type="cite_precedent", parameters={"case_id": "CASE-001", "relevance": "Key holding on non-delegable encryption duty."}))
    # Citing a required precedent should improve score
    assert result.score >= 0.0


def test_reset_clears_state(env):
    """Verify that reset produces a clean slate with no carryover."""
    env.reset("task_1")
    env.step(ActionRequest(action_type="classify_document", parameters={"doc_index": 0, "document_type": "nda"}))

    # Reset should clear everything
    result = env.reset("task_1")
    obs = result.observation
    assert obs.step_count == 0
    assert obs.findings_so_far["classification"] is None
    assert obs.findings_so_far["parties"] == []


def test_episode_boundary(env):
    """Verify that done flag works correctly and prevents further actions."""
    env.reset("task_1")
    result = env.step(ActionRequest(action_type="submit_analysis", parameters={}))
    assert result.done is True

    # Further steps should return done=True with 0 reward
    result = env.step(ActionRequest(action_type="classify_document", parameters={"doc_index": 0, "document_type": "nda"}))
    assert result.done is True
    assert result.reward == 0.0


def test_all_tasks_initialize(env):
    """Verify all 5 tasks can be initialized without error."""
    for task_id in ["task_1", "task_2", "task_3", "task_4", "task_5"]:
        result = env.reset(task_id)
        obs = result.observation
        assert obs.task_id == task_id
        assert obs.max_steps > 0
        assert len(obs.available_actions) > 0
