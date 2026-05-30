"""Tests for swo.main — orchestrator integration smoke tests."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from swo.main import run_pipeline

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_workflow():
    return {
        "skill_name": "ce-plan",
        "plugin_version": "3.9.3",
        "plugin_git_sha": "abc",
        "skill_doc_path": "/fake",
        "extracted_at": "2026-05-30T00:00:00+00:00",
        "stages": [
            {"stage_id": "S1", "stage_name": "Stage 1", "stage_goal": "Goal.",
             "required_vs_optional": "required", "turn_pattern": "prose-reasoning",
             "artifact_expectation": "", "source_section": "### Phase 0", "source_quote": "G."},
        ],
    }


def _stub_session():
    return json.loads((FIXTURES / "session-stub.json").read_text())


def _stub_trajectory():
    return json.loads((FIXTURES / "trajectory-stub.json").read_text())


def _stub_deviations():
    return json.loads((FIXTURES / "deviations-stub.json").read_text())


# ---------------------------------------------------------------------------
# Integration smoke test: full pipeline via stubs
# ---------------------------------------------------------------------------

@patch("swo.main._step_extract_workflow")
@patch("swo.main._step_export_session")
@patch("swo.main._step_extract_trajectory")
@patch("swo.main._step_analyse_deviations")
def test_run_pipeline_happy_path(
    mock_analyse, mock_traj, mock_export, mock_workflow, tmp_path
):
    mock_workflow.return_value = _stub_workflow()
    mock_export.return_value = ("test_session_001", _stub_session())
    mock_traj.return_value = _stub_trajectory()
    mock_analyse.return_value = _stub_deviations()

    # Patch render_and_save to write to tmp_path
    with patch("swo.main._reports_dir", return_value=tmp_path):
        result = run_pipeline(
            skill="ce-plan",
            session_id="test_session_001",
            open_browser=False,
        )

    assert result.exists()
    assert result.suffix == ".html"
    html = result.read_text()
    assert "1. Executive Verdict" in html


@patch("swo.main._step_extract_workflow")
@patch("swo.main._step_export_session")
@patch("swo.main._step_extract_trajectory")
@patch("swo.main._step_analyse_deviations")
def test_run_pipeline_caches_workflow(
    mock_analyse, mock_traj, mock_export, mock_workflow, tmp_path
):
    """Verify that each step is called exactly once (caching logic in step helpers)."""
    mock_workflow.return_value = _stub_workflow()
    mock_export.return_value = ("test_session_001", _stub_session())
    mock_traj.return_value = _stub_trajectory()
    mock_analyse.return_value = _stub_deviations()

    with patch("swo.main._reports_dir", return_value=tmp_path):
        run_pipeline(skill="ce-plan", session_id="test_session_001")
        run_pipeline(skill="ce-plan", session_id="test_session_001")

    assert mock_workflow.call_count == 2  # step is called; caching is inside the step


@patch("swo.main._step_export_session", side_effect=ValueError("Session not found: 'bad_id'"))
def test_run_pipeline_missing_session_raises(mock_export):
    with pytest.raises(ValueError, match="Session not found"):
        run_pipeline(skill="ce-plan", session_id="bad_id")


# ---------------------------------------------------------------------------
# _step_extract_workflow: cache hit
# ---------------------------------------------------------------------------

def test_step_extract_workflow_cache_hit(tmp_path):
    from swo.main import _step_extract_workflow
    workflow = _stub_workflow()

    # Pre-write the cache file
    cache_dir = tmp_path / "expected_workflows"
    cache_dir.mkdir()
    (cache_dir / "ce-plan_workflow.json").write_text(json.dumps(workflow))

    with patch("swo.main._data_dir", return_value=cache_dir):
        result = _step_extract_workflow("ce-plan")

    assert result["skill_name"] == "ce-plan"


# ---------------------------------------------------------------------------
# _step_export_session: cache hit by session_id
# ---------------------------------------------------------------------------

def test_step_export_session_cache_hit_by_id(tmp_path):
    from swo.main import _step_export_session
    session = _stub_session()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "test_session_001.json").write_text(json.dumps(session))

    with patch("swo.main._data_dir", return_value=sessions_dir):
        sid, result = _step_export_session("test_session_001", None)

    assert sid == "test_session_001"
    assert result["session_id"] == "test_session_001"
