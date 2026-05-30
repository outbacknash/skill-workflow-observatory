"""Tests for swo.extractor — skill doc parser."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from swo.extractor import (
    _classify_required,
    _extract_artifact,
    _first_paragraph,
    _infer_turn_pattern,
    _parse_skill_md,
    extract_workflow,
)
from swo.schemas import ExpectedWorkflowStage

FIXTURES = Path(__file__).parent / "fixtures"
STUB_SKILL_MD = FIXTURES / "ce-plan-skill-stub.md"


# ---------------------------------------------------------------------------
# Unit: _classify_required
# ---------------------------------------------------------------------------

def test_classify_required_critical():
    assert _classify_required("### Phase 0: Scope", "CRITICAL: Do this first.") == "required"


def test_classify_required_required_keyword():
    assert _classify_required("### Phase 3: Write", "REQUIRED: plan file must exist.") == "required"


def test_classify_required_optional():
    assert _classify_required("### Phase 2: Questions", "Optional: If blocking questions exist, resolve them.") == "conditional"


def test_classify_required_default():
    assert _classify_required("### Phase 1: Context", "Run parallel research agents.") == "optional"


# ---------------------------------------------------------------------------
# Unit: _infer_turn_pattern
# ---------------------------------------------------------------------------

def test_infer_turn_pattern_parallel():
    assert _infer_turn_pattern("Run these agents in parallel: Task ce-repo-research-analyst(...)") == "parallel-research"


def test_infer_turn_pattern_one_question():
    assert _infer_turn_pattern("Ask the user one question at a time.") == "one-question-at-a-time"


def test_infer_turn_pattern_artifact_write():
    assert _infer_turn_pattern("Write the plan to docs/plans/...") == "artifact-write"


def test_infer_turn_pattern_default():
    assert _infer_turn_pattern("Resolve planning questions.") == "prose-reasoning"


# ---------------------------------------------------------------------------
# Unit: _extract_artifact
# ---------------------------------------------------------------------------

def test_extract_artifact_plan_path():
    result = _extract_artifact("Write to docs/plans/2026-05-30-001-feat-thing-plan.md")
    assert "docs/plans/" in result


def test_extract_artifact_strategy():
    result = _extract_artifact("If STRATEGY.md exists, read it.")
    assert result == "STRATEGY.md"


def test_extract_artifact_none():
    result = _extract_artifact("Resolve planning questions and proceed.")
    assert result == ""


# ---------------------------------------------------------------------------
# Unit: _first_paragraph
# ---------------------------------------------------------------------------

def test_first_paragraph_basic():
    text = "First paragraph.\n\nSecond paragraph."
    assert _first_paragraph(text) == "First paragraph."


def test_first_paragraph_truncation():
    long = "x" * 400
    assert len(_first_paragraph(long, max_chars=300)) <= 300


def test_first_paragraph_empty():
    assert _first_paragraph("") == ""


# ---------------------------------------------------------------------------
# Integration: _parse_skill_md using stub fixture
# ---------------------------------------------------------------------------

def test_parse_skill_md_produces_stages():
    stages = _parse_skill_md(STUB_SKILL_MD)
    assert len(stages) == 4


def test_parse_skill_md_stage_ids():
    stages = _parse_skill_md(STUB_SKILL_MD)
    ids = [s["stage_id"] for s in stages]
    assert ids == ["S1", "S2", "S3", "S4"]


def test_parse_skill_md_required_classification():
    stages = _parse_skill_md(STUB_SKILL_MD)
    s0 = stages[0]
    assert s0["required_vs_optional"] == "required"


def test_parse_skill_md_optional_classification():
    stages = _parse_skill_md(STUB_SKILL_MD)
    s2 = stages[2]  # Phase 2 has "Optional:"
    assert s2["required_vs_optional"] == "conditional"


def test_parse_skill_md_artifact_detected():
    stages = _parse_skill_md(STUB_SKILL_MD)
    s3 = stages[3]  # Phase 3 mentions docs/plans/
    assert "docs/plans/" in s3["artifact_expectation"]


def test_parse_skill_md_stage_names():
    stages = _parse_skill_md(STUB_SKILL_MD)
    names = [s["stage_name"] for s in stages]
    assert "Resume, Source, and Scope" in names
    assert "Gather Context" in names


def test_parse_skill_md_source_quote_non_empty():
    stages = _parse_skill_md(STUB_SKILL_MD)
    for s in stages:
        assert len(s["source_quote"]) > 0, f"Empty source_quote for {s['stage_id']}"


# ---------------------------------------------------------------------------
# Integration: extract_workflow — missing SKILL.md raises FileNotFoundError
# ---------------------------------------------------------------------------

def test_extract_workflow_missing_skill_raises():
    with patch.dict(os.environ, {"CE_PLUGIN_PATH": "/nonexistent/path"}):
        with pytest.raises(FileNotFoundError) as exc_info:
            extract_workflow("ce-plan")
    assert "SKILL.md not found" in str(exc_info.value)


def test_extract_workflow_unsupported_skill():
    with pytest.raises(ValueError, match="Unsupported skill"):
        extract_workflow("ce-unknown")


# ---------------------------------------------------------------------------
# Integration: extract_workflow — round-trip JSON serialisation
# ---------------------------------------------------------------------------

def test_extract_workflow_roundtrip():
    """Workflow dict produced from stub must round-trip through JSON cleanly."""
    with patch("swo.extractor._get_plugin_root") as mock_root, \
         patch("swo.extractor._get_plugin_metadata", return_value=("3.9.3", "abc123")):
        # Point plugin root to fixtures dir, stub skill dir structure
        stub_root = FIXTURES.parent / "_fake_plugin_root"
        skill_dir = stub_root / "skills" / "ce-plan"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(STUB_SKILL_MD.read_text())
        mock_root.return_value = stub_root

        try:
            workflow = extract_workflow("ce-plan")
            serialised = json.dumps(workflow)
            restored = json.loads(serialised)
            assert restored["skill_name"] == "ce-plan"
            assert len(restored["stages"]) == 4
        finally:
            import shutil
            shutil.rmtree(stub_root, ignore_errors=True)
