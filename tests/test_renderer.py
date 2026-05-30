"""Tests for swo.renderer — HTML report generation."""

import json
from pathlib import Path

import pytest

from swo.renderer import _fit_counts, _followed_pct, build_report_model, render_report, render_and_save

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _make_full_model(
    assessments=None,
    turns=None,
    session_id="sess_001",
    skill_name="ce-plan",
):
    if assessments is None:
        assessments = [
            {
                "expected_stage_id": "S1",
                "actual_turn_range": "T1-T2",
                "deviation_type": "followed",
                "fit_for_knowledge_work": "helpful",
                "rationale": "Stage was followed well.",
                "confidence": 0.9,
                "possible_skill_change": "none",
            },
            {
                "expected_stage_id": "S2",
                "actual_turn_range": "",
                "deviation_type": "skipped",
                "fit_for_knowledge_work": "neutral",
                "rationale": "Stage was appropriately skipped.",
                "confidence": 0.75,
                "possible_skill_change": "invocation-guide",
            },
        ]
    if turns is None:
        turns = [
            {
                "turn_id": "T1",
                "role": "user",
                "content_summary": "User asked to plan.",
                "tools_used": [],
                "artifacts_created": [],
                "decision_made": "",
                "inferred_stage_id": "S1",
                "confidence": 0.8,
                "evidence_quote": "Plan a feature.",
            },
            {
                "turn_id": "T2",
                "role": "agent",
                "content_summary": "Agent ran research and wrote plan.",
                "tools_used": ["web_search"],
                "artifacts_created": ["docs/plans/2026-05-30-001-feat-x-plan.md"],
                "decision_made": "Scope is bounded to auth only",
                "inferred_stage_id": "S1",
                "confidence": 0.9,
                "evidence_quote": "Here is the plan...",
            },
        ]

    return {
        "session_id": session_id,
        "skill_name": skill_name,
        "session_title": "Test Session",
        "generated_at": "2026-05-30T12:00:00+00:00",
        "expected_workflow": {
            "skill_name": skill_name,
            "plugin_version": "3.9.3",
            "plugin_git_sha": "abc123def456",
            "skill_doc_path": "/fake/path",
            "extracted_at": "2026-05-30T00:00:00+00:00",
            "stages": [
                {"stage_id": "S1", "stage_name": "Resume, Source, Scope",
                 "stage_goal": "Determine how to proceed.", "required_vs_optional": "required",
                 "turn_pattern": "prose-reasoning", "artifact_expectation": "",
                 "source_section": "### Phase 0", "source_quote": "Determine."},
                {"stage_id": "S2", "stage_name": "Gather Context",
                 "stage_goal": "Run research.", "required_vs_optional": "optional",
                 "turn_pattern": "parallel-research", "artifact_expectation": "",
                 "source_section": "### Phase 1", "source_quote": "Research."},
            ],
        },
        "trajectory": {
            "session_id": session_id,
            "skill_name": skill_name,
            "extracted_at": "2026-05-30T00:00:00+00:00",
            "truncated": False,
            "turns": turns,
        },
        "deviations": {
            "session_id": session_id,
            "skill_name": skill_name,
            "analyzed_at": "2026-05-30T00:00:00+00:00",
            "executive_verdict": "The session was productive.",
            "assessments": assessments,
        },
    }


# ---------------------------------------------------------------------------
# Unit: helpers
# ---------------------------------------------------------------------------

def test_fit_counts_basic():
    assessments = [
        {"fit_for_knowledge_work": "helpful"},
        {"fit_for_knowledge_work": "helpful"},
        {"fit_for_knowledge_work": "harmful"},
        {"fit_for_knowledge_work": "neutral"},
    ]
    counts = _fit_counts(assessments)
    assert counts["helpful"] == 2
    assert counts["harmful"] == 1
    assert counts["neutral"] == 1


def test_fit_counts_empty():
    counts = _fit_counts([])
    assert all(v == 0 for v in counts.values())


def test_followed_pct_all_followed():
    assessments = [{"deviation_type": "followed"}, {"deviation_type": "followed"}]
    assert _followed_pct(assessments) == 100


def test_followed_pct_none_followed():
    assessments = [{"deviation_type": "skipped"}, {"deviation_type": "compressed"}]
    assert _followed_pct(assessments) == 0


def test_followed_pct_half():
    assessments = [{"deviation_type": "followed"}, {"deviation_type": "skipped"}]
    assert _followed_pct(assessments) == 50


def test_followed_pct_empty():
    assert _followed_pct([]) == 0


# ---------------------------------------------------------------------------
# Integration: render_report — structure
# ---------------------------------------------------------------------------

def test_render_report_all_six_sections_present():
    model = _make_full_model()
    html = render_report(model)
    for heading in [
        "1. Executive Verdict",
        "2. Workflow Map",
        "3. Deviations Table",
        "4. Knowledge-Work Fit Analysis",
        "5. Evidence Drawer",
        "6. Adaptation Recommendations",
    ]:
        assert heading in html, f"Missing section: {heading}"


def test_render_report_no_cdn_refs():
    model = _make_full_model()
    html = render_report(model)
    import re
    cdn_refs = re.findall(r'(?:src|href)\s*=\s*["\']https?://', html, re.IGNORECASE)
    assert cdn_refs == [], f"Found CDN refs: {cdn_refs}"


def test_render_report_viewport_tag_present():
    model = _make_full_model()
    html = render_report(model)
    assert 'name="viewport"' in html


def test_render_report_evidence_drawer_per_turn():
    turns = [
        {"turn_id": f"T{i}", "role": "user", "content_summary": f"Turn {i}",
         "tools_used": [], "artifacts_created": [], "decision_made": "",
         "inferred_stage_id": None, "confidence": 0.5, "evidence_quote": f"Quote {i}"}
        for i in range(1, 4)
    ]
    model = _make_full_model(turns=turns)
    html = render_report(model)
    assert html.count("<details>") == 3


def test_render_report_empty_deviations():
    model = _make_full_model(assessments=[])
    html = render_report(model)
    # Should render without error and contain the section headings
    assert "1. Executive Verdict" in html
    assert "No deviation assessments" in html or "No adaptation recommendations" in html


def test_render_report_truncated_flag_shown():
    model = _make_full_model()
    model["trajectory"]["truncated"] = True
    html = render_report(model)
    assert "truncated" in html.lower()


def test_render_report_recommendation_shown():
    model = _make_full_model()
    html = render_report(model)
    # S2 has possible_skill_change=invocation-guide — should appear in recs
    assert "invocation-guide" in html.lower() or "INVOCATION-GUIDE" in html


def test_render_report_no_recs_message_when_all_none():
    model = _make_full_model(assessments=[
        {"expected_stage_id": "S1", "actual_turn_range": "T1", "deviation_type": "followed",
         "fit_for_knowledge_work": "helpful", "rationale": "Good.", "confidence": 0.9,
         "possible_skill_change": "none"},
    ])
    html = render_report(model)
    assert "No adaptation recommendations" in html


# ---------------------------------------------------------------------------
# Integration: render_and_save
# ---------------------------------------------------------------------------

def test_render_and_save_creates_file(tmp_path):
    model = _make_full_model()
    out_path = render_and_save(model, tmp_path)
    assert out_path.exists()
    assert out_path.suffix == ".html"
    content = out_path.read_text()
    assert "<!DOCTYPE html>" in content


def test_render_and_save_filename_includes_session_id(tmp_path):
    model = _make_full_model(session_id="my_session_123")
    out_path = render_and_save(model, tmp_path)
    assert "my_session_123" in out_path.name


# ---------------------------------------------------------------------------
# build_report_model
# ---------------------------------------------------------------------------

def test_build_report_model_fields():
    workflow = _make_full_model()["expected_workflow"]
    trajectory = _make_full_model()["trajectory"]
    deviations = _make_full_model()["deviations"]
    m = build_report_model("sid", "My Title", "ce-plan", workflow, trajectory, deviations)
    assert m["session_id"] == "sid"
    assert m["session_title"] == "My Title"
    assert m["skill_name"] == "ce-plan"
    assert "generated_at" in m
