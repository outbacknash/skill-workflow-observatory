"""U3: Session Exporter — read hermes state.db via SessionDB Python API."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path injection — make hermes_state importable from the editable install
# ---------------------------------------------------------------------------

_HERMES_AGENT_DIR = Path.home() / ".hermes/hermes-agent"
if str(_HERMES_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_HERMES_AGENT_DIR))

try:
    from hermes_state import SessionDB  # type: ignore[import]
except ImportError as e:
    raise ImportError(
        f"Could not import hermes_state from {_HERMES_AGENT_DIR}. "
        f"Activate the hermes venv: source ~/.hermes/hermes-agent/venv/bin/activate\n"
        f"Original error: {e}"
    ) from e

from swo.schemas import SessionExport, SessionMessage

_DEFAULT_DB_PATH = Path.home() / ".hermes/state.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tool_calls(raw: str | None) -> list[dict] | None:
    """Parse the JSON-encoded tool_calls string from the DB."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _build_session_export(session_dict: dict, messages: list[dict]) -> SessionExport:
    session_messages: list[SessionMessage] = []
    for m in messages:
        # Skip reasoning-only and internal metadata fields
        session_messages.append(
            SessionMessage(
                id=m.get("id", 0),
                role=m.get("role", ""),
                content=m.get("content"),
                tool_calls=_parse_tool_calls(m.get("tool_calls")),
                tool_name=m.get("tool_name"),
                timestamp=m.get("timestamp", 0.0),
            )
        )

    return SessionExport(
        session_id=session_dict["id"],
        source=session_dict.get("source", "cli"),
        model=session_dict.get("model"),
        title=session_dict.get("title"),
        started_at=session_dict.get("started_at", 0.0),
        ended_at=session_dict.get("ended_at"),
        message_count=len(session_messages),
        exported_at=datetime.now(timezone.utc).isoformat(),
        messages=session_messages,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_db(db_path: Path | None = None) -> SessionDB:
    return SessionDB(db_path or _DEFAULT_DB_PATH)


def export_by_id(session_id: str, db_path: Path | None = None) -> SessionExport:
    """Export a single session by ID."""
    db = get_db(db_path)
    raw = db.export_session(session_id)
    if raw is None:
        raise ValueError(f"Session not found: {session_id!r}")
    messages = raw.get("messages", [])
    return _build_session_export(raw, messages)


def export_by_title(title: str, db_path: Path | None = None) -> SessionExport:
    """Export a session by its title (resolves to most recent in lineage)."""
    db = get_db(db_path)
    session_id = db.resolve_session_by_title(title)
    if session_id is None:
        raise ValueError(
            f"No session found with title {title!r}. "
            f"Run with --list to see available sessions."
        )
    return export_by_id(session_id, db_path)


def list_recent_sessions(limit: int = 20, db_path: Path | None = None) -> list[dict]:
    """Return recent CLI sessions with id, title, started_at, message_count."""
    db = get_db(db_path)
    sessions = db.list_sessions_rich(limit=limit, source="cli")
    return [
        {
            "id": s["id"],
            "title": s.get("title") or "(untitled)",
            "started_at": s.get("started_at", 0),
            "message_count": s.get("message_count", 0),
            "model": s.get("model", ""),
        }
        for s in sessions
    ]


def save_export(export: SessionExport, output_dir: Path) -> Path:
    """Write session JSON to output_dir/{session_id}.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{export['session_id']}.json"
    out_path.write_text(json.dumps(export, indent=2), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export a hermes session from state.db to JSON."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--session-id", help="Export session by ID")
    group.add_argument("--session-title", help="Export session by title")
    group.add_argument("--list", action="store_true", help="List recent CLI sessions")
    parser.add_argument(
        "--output-dir",
        default="data/sessions",
        help="Directory to write session JSON (default: data/sessions)",
    )
    parser.add_argument("--db", help="Path to state.db (default: ~/.hermes/state.db)")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else None

    try:
        if args.list:
            sessions = list_recent_sessions(db_path=db_path)
            if not sessions:
                print("No CLI sessions found in state.db.")
                return
            print(f"{'ID':<30} {'Title':<45} {'Messages':>8}")
            print("-" * 85)
            for s in sessions:
                from datetime import datetime as dt
                started = dt.fromtimestamp(s["started_at"]).strftime("%Y-%m-%d %H:%M")
                title = (s["title"] or "(untitled)")[:44]
                print(f"{s['id']:<30} {title:<45} {s['message_count']:>8}  {started}")
            return

        if args.session_id:
            export = export_by_id(args.session_id, db_path)
        elif args.session_title:
            export = export_by_title(args.session_title, db_path)
        else:
            parser.print_help()
            sys.exit(1)

        out_path = save_export(export, Path(args.output_dir))
        print(f"Exported {export['message_count']} messages → {out_path}")

    except (ValueError, ImportError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
