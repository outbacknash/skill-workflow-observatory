"""Tests for swo.trajectory — LLM-assisted turn extraction."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from swo.trajectory import (
    _group_turns,
    _render_turn_for_llm,
    _serialise_session_for_llm,
    extract_trajectory,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(messages):
    return {
        "session_id": "test_001",
        "source": "cli",
        "model": "claude-sonnet",
        "title": "Test",
        "started_at": 1748563200.0,
        "ended_at": None,
        "message_count": len(messages),
        "exported_at": "2026-05-30T00:00:00+00:00",
        "messages": messages,
    }


def _make_workflow(stages=None):
    if stages is None:
        stages = [
            {"stage_id": "S1", "stage_name": "Resume, Source, and Scope",
             "stage_goal": "Determine how to proceed.", "required_vs_optional": "required",
             "turn_pattern": "prose-reasoning", "artifact_expectation": "",
             "source_section": "### Phase 0", "source_quote": "Determine how to proceed."},
            {"stage_id": "S2", "stage_name": "Gather Context",
             "stage_goal": "Run parallel research agents.", "required_vs_optional": "optional",
             "turn_pattern": "parallel-research", "artifact_expectation": "",
             "source_section": "### Phase 1", "source_quote": "Run parallel research."},
        ]
    return {
        "skill_name": "ce-plan",
        "plugin_version": "3.9.3",
        "plugin_git_sha": "abc123",
        "skill_doc_path": "/fake/path",
        "extracted_at": "2026-05-30T00:00:00+00:00",
        "stages": stages,
    }


# ---------------------------------------------------------------------------
# Unit: _group_turns
# ---------------------------------------------------------------------------

def test_group_turns_user_is_own_turn():
    messages = [{"role": "user", "content": "Hello"}]
    groups = _group_turns(messages)
    assert len(groups) == 1
    assert groups[0]["role"] == "user"


def test_group_turns_assistant_grouped_with_tool():
    messages = [
        {"role": "user", "content": "Go"},
        {"role": "assistant", "content": "Calling tool", "tool_calls": None, "tool_name": None},
        {"role": "tool", "content": "result", "tool_name": "terminal", "tool_calls": None},
    ]
    groups = _group_turns(messages)
    assert len(groups) == 2
    assert groups[0]["role"] == "user"
    assert groups[1]["role"] == "agent"
    assert len(groups[1]["messages"]) == 2


def test_group_turns_multiple_user_messages():
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Reply", "tool_calls": None, "tool_name": None},
        {"role": "user", "content": "Second"},
    ]
    groups = _group_turns(messages)
    assert len(groups) == 3
    assert groups[0]["role"] == "user"
    assert groups[1]["role"] == "agent"
    assert groups[2]["role"] == "user"


def test_group_turns_trailing_agent_turn_flushed():
    messages = [
        {"role": "user", "content": "Start"},
        {"role": "assistant", "content": "Working...", "tool_calls": None, "tool_name": None},
    ]
    groups = _group_turns(messages)
    assert len(groups) == 2
    assert groups[1]["role"] == "agent"


def test_group_turns_empty():
    assert _group_turns([]) == []


# ---------------------------------------------------------------------------
# Unit: _render_turn_for_llm
# ---------------------------------------------------------------------------

def test_render_turn_includes_turn_id():
    group = {"role": "user", "messages": [{"role": "user", "content": "Hello", "tool_calls": None, "tool_name": None}]}
    rendered = _render_turn_for_llm(group, "T1")
    assert "T1" in rendered
    assert "USER" in rendered


def test_render_turn_includes_tool_call_name():
    group = {
        "role": "agent",
        "messages": [
            {"role": "assistant", "content": None,
             "tool_calls": [{"function": {"name": "web_search"}}],
             "tool_name": None},
        ]
    }
    rendered = _render_turn_for_llm(group, "T2")
    assert "web_search" in rendered


# ---------------------------------------------------------------------------
# Unit: _serialise_session_for_llm
# ---------------------------------------------------------------------------

def test_serialise_session_not_truncated_for_short_session():
    session = _make_session([
        {"id": 1, "role": "user", "content": "Hello", "tool_calls": None, "tool_name": None, "timestamp": 1.0},
    ])
    text, truncated = _serialise_session_for_llm(session)
    assert not truncated
    assert "T1" in text


def test_serialise_session_truncated_for_long_content():
    long_content = "x" * 200_000
    session = _make_session([
        {"id": 1, "role": "user", "content": long_content, "tool_calls": None, "tool_name": None, "timestamp": 1.0},
    ])
    text, truncated = _serialise_session_for_llm(session)
    assert truncated
    assert len(text) <= 100_000


# ---------------------------------------------------------------------------
# Integration: extract_trajectory — mocked LLM
# ---------------------------------------------------------------------------

_STUB_LLM_TURNS = [
    {
        "turn_id": "T1",
        "role": "user",
        "content_summary": "User requests a plan.",
        "tools_used": [],
        "artifacts_created": [],
        "decision_made": "",
        "inferred_stage_id": None,
        "confidence": 0.3,
        "evidence_quote": "I want to plan a new feature.",
    },
    {
        "turn_id": "T2",
        "role": "agent",
        "content_summary": "Agent clarifies scope with one question.",
        "tools_used": [],
        "artifacts_created": [],
        "decision_made": "Asked for feature description",
        "inferred_stage_id": "S1",
        "confidence": 0.85,
        "evidence_quote": "What would you like to plan?",
    },
]


@patch("swo.trajectory._call_llm", return_value=_STUB_LLM_TURNS)
def test_extract_trajectory_happy_path(mock_llm):
    session = _make_session([
        {"id": 1, "role": "user", "content": "I want to plan a new feature.", "tool_calls": None, "tool_name": None, "timestamp": 1.0},
        {"id": 2, "role": "assistant", "content": "What would you like to plan?", "tool_calls": None, "tool_name": None, "timestamp": 2.0},
    ])
    trajectory = extract_trajectory(session, _make_workflow())
    assert len(trajectory["turns"]) == 2
    assert trajectory["session_id"] == "test_001"
    assert trajectory["skill_name"] == "ce-plan"


@patch("swo.trajectory._call_llm", return_value=_STUB_LLM_TURNS)
def test_extract_trajectory_confidence_clamped(mock_llm):
    # Inject out-of-range confidence values
    bad_turns = [dict(t, confidence=1.5) for t in _STUB_LLM_TURNS]
    with patch("swo.trajectory._call_llm", return_value=bad_turns):
        session = _make_session([
            {"id": 1, "role": "user", "content": "Hello", "tool_calls": None, "tool_name": None, "timestamp": 1.0}
        ])
        trajectory = extract_trajectory(session, _make_workflow())
        for turn in trajectory["turns"]:
            assert 0.0 <= turn["confidence"] <= 1.0


@patch("swo.trajectory._call_llm", return_value=_STUB_LLM_TURNS)
def test_extract_trajectory_truncated_flag(mock_llm):
    long_content = "x" * 200_000
    session = _make_session([
        {"id": 1, "role": "user", "content": long_content, "tool_calls": None, "tool_name": None, "timestamp": 1.0},
    ])
    trajectory = extract_trajectory(session, _make_workflow())
    assert trajectory["truncated"] is True


@patch("swo.trajectory._call_llm", return_value=_STUB_LLM_TURNS)
def test_extract_trajectory_all_turns_have_required_fields(mock_llm):
    session = _make_session([
        {"id": 1, "role": "user", "content": "Plan something", "tool_calls": None, "tool_name": None, "timestamp": 1.0}
    ])
    trajectory = extract_trajectory(session, _make_workflow())
    required_fields = {"turn_id", "role", "content_summary", "tools_used",
                       "artifacts_created", "decision_made", "inferred_stage_id",
                       "confidence", "evidence_quote"}
    for turn in trajectory["turns"]:
        assert required_fields.issubset(set(turn.keys())), f"Missing fields in turn: {turn}"


# ---------------------------------------------------------------------------
# Fixture round-trip
# ---------------------------------------------------------------------------

def test_trajectory_stub_fixture_loads():
    stub = json.loads((FIXTURES / "trajectory-stub.json").read_text())
    assert stub["session_id"] == "test_session_001"
    assert len(stub["turns"]) == 3
    for turn in stub["turns"]:
        assert 0.0 <= turn["confidence"] <= 1.0
