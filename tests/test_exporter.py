"""Tests for swo.exporter — session export from hermes state.db."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from swo.exporter import (
    _build_session_export,
    _parse_tool_calls,
    export_by_id,
    export_by_title,
    list_recent_sessions,
    save_export,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Unit: _parse_tool_calls
# ---------------------------------------------------------------------------

def test_parse_tool_calls_valid_json():
    raw = json.dumps([{"id": "call_1", "function": {"name": "terminal"}}])
    result = _parse_tool_calls(raw)
    assert result is not None
    assert result[0]["id"] == "call_1"


def test_parse_tool_calls_none():
    assert _parse_tool_calls(None) is None


def test_parse_tool_calls_empty_string():
    assert _parse_tool_calls("") is None


def test_parse_tool_calls_invalid_json():
    assert _parse_tool_calls("{broken") is None


def test_parse_tool_calls_non_list_json():
    # JSON object (not list) should return None
    assert _parse_tool_calls('{"key": "value"}') is None


# ---------------------------------------------------------------------------
# Unit: _build_session_export
# ---------------------------------------------------------------------------

def _make_session_dict(session_id="sess_001"):
    return {
        "id": session_id,
        "source": "cli",
        "model": "anthropic/claude-sonnet-4-6",
        "title": "Test Session",
        "started_at": 1748563200.0,
        "ended_at": 1748566800.0,
        "message_count": 2,
    }


def _make_messages():
    return [
        {"id": 1, "role": "user", "content": "Hello", "tool_calls": None, "tool_name": None, "timestamp": 1748563200.0},
        {"id": 2, "role": "assistant", "content": "Hi there", "tool_calls": None, "tool_name": None, "timestamp": 1748563210.0},
    ]


def test_build_session_export_basic():
    export = _build_session_export(_make_session_dict(), _make_messages())
    assert export["session_id"] == "sess_001"
    assert export["message_count"] == 2
    assert len(export["messages"]) == 2


def test_build_session_export_tool_calls_parsed():
    raw_calls = json.dumps([{"id": "c1", "function": {"name": "web_search"}}])
    messages = [
        {"id": 1, "role": "assistant", "content": None, "tool_calls": raw_calls, "tool_name": "web_search", "timestamp": 1.0}
    ]
    export = _build_session_export(_make_session_dict(), messages)
    assert export["messages"][0]["tool_calls"] == [{"id": "c1", "function": {"name": "web_search"}}]


def test_build_session_export_exported_at_present():
    export = _build_session_export(_make_session_dict(), [])
    assert "exported_at" in export
    assert "2026" in export["exported_at"]


# ---------------------------------------------------------------------------
# Integration: export_by_id — mocked SessionDB
# ---------------------------------------------------------------------------

def _make_mock_db(session_data=None, messages=None):
    mock_db = MagicMock()
    if session_data is None:
        session_data = {**_make_session_dict(), "messages": messages or _make_messages()}
    mock_db.export_session.return_value = session_data
    return mock_db


@patch("swo.exporter.get_db")
def test_export_by_id_happy_path(mock_get_db):
    mock_db = _make_mock_db()
    mock_get_db.return_value = mock_db

    export = export_by_id("sess_001")
    assert export["session_id"] == "sess_001"
    assert export["message_count"] == 2
    mock_db.export_session.assert_called_once_with("sess_001")


@patch("swo.exporter.get_db")
def test_export_by_id_not_found(mock_get_db):
    mock_db = MagicMock()
    mock_db.export_session.return_value = None
    mock_get_db.return_value = mock_db

    with pytest.raises(ValueError, match="Session not found"):
        export_by_id("missing_session")


# ---------------------------------------------------------------------------
# Integration: export_by_title — mocked SessionDB
# ---------------------------------------------------------------------------

@patch("swo.exporter.get_db")
def test_export_by_title_resolves(mock_get_db):
    mock_db = MagicMock()
    mock_db.resolve_session_by_title.return_value = "sess_001"
    mock_db.export_session.return_value = {**_make_session_dict(), "messages": _make_messages()}
    mock_get_db.return_value = mock_db

    export = export_by_title("Test Session")
    assert export["session_id"] == "sess_001"


@patch("swo.exporter.get_db")
def test_export_by_title_not_found(mock_get_db):
    mock_db = MagicMock()
    mock_db.resolve_session_by_title.return_value = None
    mock_get_db.return_value = mock_db

    with pytest.raises(ValueError, match="No session found"):
        export_by_title("Nonexistent Title")


# ---------------------------------------------------------------------------
# Integration: list_recent_sessions
# ---------------------------------------------------------------------------

@patch("swo.exporter.get_db")
def test_list_recent_sessions_happy_path(mock_get_db):
    mock_db = MagicMock()
    mock_db.list_sessions_rich.return_value = [
        {"id": "s1", "title": "Session A", "started_at": 1748563200.0, "message_count": 5, "model": "claude-sonnet"},
        {"id": "s2", "title": None, "started_at": 1748563300.0, "message_count": 2, "model": ""},
    ]
    mock_get_db.return_value = mock_db

    sessions = list_recent_sessions()
    assert len(sessions) == 2
    assert sessions[0]["id"] == "s1"
    assert sessions[1]["title"] == "(untitled)"


@patch("swo.exporter.get_db")
def test_list_recent_sessions_empty(mock_get_db):
    mock_db = MagicMock()
    mock_db.list_sessions_rich.return_value = []
    mock_get_db.return_value = mock_db

    sessions = list_recent_sessions()
    assert sessions == []


# ---------------------------------------------------------------------------
# Integration: save_export
# ---------------------------------------------------------------------------

def test_save_export_writes_file(tmp_path):
    stub = json.loads((FIXTURES / "session-stub.json").read_text())
    out_path = save_export(stub, tmp_path)
    assert out_path.exists()
    restored = json.loads(out_path.read_text())
    assert restored["session_id"] == "test_session_001"
    assert len(restored["messages"]) == 3


def test_save_export_filename_matches_session_id(tmp_path):
    stub = json.loads((FIXTURES / "session-stub.json").read_text())
    out_path = save_export(stub, tmp_path)
    assert out_path.name == "test_session_001.json"
