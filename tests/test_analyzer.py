"""Tests for swo.analyzer — deviation assessment."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from swo.analyzer import analyse_deviations, analyse_and_save
from swo.schemas import DeviationReport

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_workflow(n_stages=3):
    stages = [
        {
            "stage_id": f"S{i}",
            "stage_name": f"Stage {i}",
            "stage_goal": f"Goal of stage {i}.",
            "required_vs_optional": "required" if i == 1 else "optional",
            "turn_pattern": "prose-reasoning",
            "artifact_expectation": "",
            "source_section": f"### Phase {i}",
            "source_quote": f"Do stage {i} work.",
        }
        for i in range(1, n_stages + 1)
    ]
    return {
        "skill_name": "ce-plan",
        "plugin_version": "3.9.3",
        "plugin_git_sha": "abc",
        "skill_doc_path": "/fake",
        "extracted_at": "2026-05-30T00:00:00+00:00",
        "stages": stages,
    }


def _make_trajectory(n_turns=2):
    turns = [
        {
            "turn_id": f"T{i}",
            "role": "user" if i % 2 == 1 else "agent",
            "content_summary": f"Turn {i} summary.",
            "tools_used": [],
            "artifacts_created": [],
            "decision_made": "",
            "inferred_stage_id": f"S{i}" if i <= 3 else None,
            "confidence": 0.7,
            "evidence_quote": f"Evidence from turn {i}.",
        }
        for i in range(1, n_turns + 1)
    ]
    return {
        "session_id": "test_001",
        "skill_name": "ce-plan",
        "extracted_at": "2026-05-30T00:00:00+00:00",
        "truncated": False,
        "turns": turns,
    }


# ---------------------------------------------------------------------------
# Stub LLM returns
# ---------------------------------------------------------------------------

_STUB_VERDICT = "The session followed most stages adequately with minor compressions."

_STUB_ASSESSMENTS = [
    {
        "expected_stage_id": "S1",
        "actual_turn_range": "T1-T2",
        "deviation_type": "followed",
        "fit_for_knowledge_work": "helpful",
        "rationale": "Stage 1 was fully executed.",
        "confidence": 0.9,
        "possible_skill_change": "none",
    },
    {
        "expected_stage_id": "S2",
        "actual_turn_range": "",
        "deviation_type": "skipped",
        "fit_for_knowledge_work": "neutral",
        "rationale": "Stage 2 was skipped appropriately.",
        "confidence": 0.75,
        "possible_skill_change": "none",
    },
    {
        "expected_stage_id": "S3",
        "actual_turn_range": "T2",
        "deviation_type": "compressed",
        "fit_for_knowledge_work": "helpful",
        "rationale": "Stage 3 was compressed into one turn.",
        "confidence": 0.6,
        "possible_skill_change": "invocation-guide",
    },
]


@patch("swo.analyzer._call_llm", return_value=(_STUB_VERDICT, _STUB_ASSESSMENTS))
def test_analyse_deviations_happy_path(mock_llm):
    workflow = _make_workflow(3)
    trajectory = _make_trajectory(2)
    report = analyse_deviations(workflow, trajectory)

    assert report["session_id"] == "test_001"
    assert report["skill_name"] == "ce-plan"
    assert len(report["assessments"]) == 3
    assert report["executive_verdict"] == _STUB_VERDICT


@patch("swo.analyzer._call_llm", return_value=(_STUB_VERDICT, _STUB_ASSESSMENTS))
def test_analyse_deviations_confidence_clamped(mock_llm):
    bad = [dict(a, confidence=2.5) for a in _STUB_ASSESSMENTS]
    with patch("swo.analyzer._call_llm", return_value=(_STUB_VERDICT, bad)):
        workflow = _make_workflow(3)
        trajectory = _make_trajectory(2)
        report = analyse_deviations(workflow, trajectory)
        for a in report["assessments"]:
            assert 0.0 <= a["confidence"] <= 1.0


@patch("swo.analyzer._call_llm", return_value=(_STUB_VERDICT, _STUB_ASSESSMENTS))
def test_analyse_deviations_all_fields_present(mock_llm):
    required = {
        "expected_stage_id", "actual_turn_range", "deviation_type",
        "fit_for_knowledge_work", "rationale", "confidence", "possible_skill_change",
    }
    workflow = _make_workflow(3)
    trajectory = _make_trajectory(2)
    report = analyse_deviations(workflow, trajectory)
    for a in report["assessments"]:
        assert required.issubset(set(a.keys()))


@patch("swo.analyzer._call_llm")
def test_analyse_deviations_all_stages_followed(mock_llm):
    followed_assessments = [
        {
            "expected_stage_id": f"S{i}",
            "actual_turn_range": f"T{i}",
            "deviation_type": "followed",
            "fit_for_knowledge_work": "helpful",
            "rationale": "Stage followed exactly.",
            "confidence": 0.95,
            "possible_skill_change": "none",
        }
        for i in range(1, 4)
    ]
    mock_llm.return_value = ("All stages followed.", followed_assessments)
    workflow = _make_workflow(3)
    trajectory = _make_trajectory(3)
    report = analyse_deviations(workflow, trajectory)
    assert all(a["deviation_type"] == "followed" for a in report["assessments"])


@patch("swo.analyzer._call_llm")
def test_analyse_deviations_invented_turn(mock_llm):
    invented = [
        {
            "expected_stage_id": None,
            "actual_turn_range": "T3",
            "deviation_type": "invented",
            "fit_for_knowledge_work": "helpful",
            "rationale": "Agent added a useful quick-start section not in the skill.",
            "confidence": 0.8,
            "possible_skill_change": "none",
        }
    ]
    mock_llm.return_value = ("One invented behaviour, helpful.", invented)
    workflow = _make_workflow(1)
    trajectory = _make_trajectory(3)
    report = analyse_deviations(workflow, trajectory)
    invented_items = [a for a in report["assessments"] if a["deviation_type"] == "invented"]
    assert len(invented_items) == 1
    assert invented_items[0]["expected_stage_id"] is None


@patch("swo.analyzer._call_llm", return_value=(_STUB_VERDICT, _STUB_ASSESSMENTS))
def test_analyse_deviations_executive_verdict_non_empty(mock_llm):
    report = analyse_deviations(_make_workflow(3), _make_trajectory(2))
    assert len(report["executive_verdict"]) > 0


@patch("swo.analyzer._call_llm", return_value=(_STUB_VERDICT, _STUB_ASSESSMENTS))
def test_analyse_and_save_writes_file(mock_llm, tmp_path):
    workflow = _make_workflow(3)
    trajectory = _make_trajectory(2)
    out_path = analyse_and_save(workflow, trajectory, tmp_path)
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["session_id"] == "test_001"
    assert "assessments" in data


# ---------------------------------------------------------------------------
# Fixture round-trip
# ---------------------------------------------------------------------------

def test_deviations_stub_fixture_loads():
    stub = json.loads((FIXTURES / "deviations-stub.json").read_text())
    assert stub["session_id"] == "test_session_001"
    assert len(stub["assessments"]) == 2
    for a in stub["assessments"]:
        assert 0.0 <= a["confidence"] <= 1.0
        assert a["deviation_type"] in {
            "followed", "skipped", "compressed", "reordered",
            "over-applied", "under-applied", "invented", "misclassified",
        }
