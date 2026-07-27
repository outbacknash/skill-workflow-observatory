"""U4: Trajectory Extractor — LLM-assisted per-turn session timeline."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from swo.schemas import ExpectedWorkflow, SessionExport, SessionTrajectory, SessionTurn

_MAX_SESSION_CHARS = 100_000
_MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Turn grouping helpers
# ---------------------------------------------------------------------------

def _group_turns(messages: list[dict]) -> list[dict]:
    """
    Group raw session messages into logical turns:
    - Each user message is its own turn (role=user)
    - Consecutive assistant + tool_result messages collapse into one agent turn (role=agent)
    """
    grouped: list[dict] = []
    current_agent: list[dict] | None = None

    for msg in messages:
        role = msg.get("role", "")

        if role == "user":
            # Flush any pending agent turn
            if current_agent is not None:
                grouped.append({"role": "agent", "messages": current_agent})
                current_agent = None
            grouped.append({"role": "user", "messages": [msg]})

        elif role in ("assistant", "tool"):
            if current_agent is None:
                current_agent = []
            current_agent.append(msg)

    # Flush trailing agent turn
    if current_agent is not None:
        grouped.append({"role": "agent", "messages": current_agent})

    return grouped


def _render_turn_for_llm(turn_group: dict, turn_id: str) -> str:
    """Render a grouped turn as readable text for the LLM."""
    role = turn_group["role"]
    lines = [f"[{turn_id} | {role.upper()}]"]

    for msg in turn_group["messages"]:
        msg_role = msg.get("role", "")
        content = msg.get("content") or ""
        tool_name = msg.get("tool_name") or ""
        tool_calls = msg.get("tool_calls") or []

        if msg_role == "user" and content:
            lines.append(f"  User: {content[:500]}")
        elif msg_role == "assistant":
            if content:
                lines.append(f"  Assistant: {content[:500]}")
            for call in (tool_calls if isinstance(tool_calls, list) else []):
                fn = call.get("function", {})
                lines.append(f"  [TOOL CALL] {fn.get('name', '?')}")
        elif msg_role == "tool":
            lines.append(f"  [TOOL RESULT] {tool_name}: {content[:200]}")

    return "\n".join(lines)


def _serialise_session_for_llm(session: SessionExport) -> tuple[str, bool]:
    """Return (session_text, was_truncated).

    Truncation is detected against the *raw* session content size so that large
    messages that get per-field truncated during rendering still set the flag.
    """
    raw_size = sum(len(m.get("content") or "") for m in session["messages"])
    truncated = raw_size > _MAX_SESSION_CHARS

    groups = _group_turns(session["messages"])
    parts: list[str] = []
    for i, group in enumerate(groups, 1):
        parts.append(_render_turn_for_llm(group, f"T{i}"))

    full_text = "\n\n".join(parts)
    return full_text[:_MAX_SESSION_CHARS], truncated


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

_EXTRACTION_TOOL = {
    "name": "record_trajectory",
    "description": "Record the session trajectory as a structured list of turns.",
    "input_schema": {
        "type": "object",
        "properties": {
            "turns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "turn_id": {"type": "string", "description": "e.g. T1, T2"},
                        "role": {"type": "string", "enum": ["user", "agent"]},
                        "content_summary": {"type": "string", "description": "≤50 words summarising what this turn did"},
                        "tools_used": {"type": "array", "items": {"type": "string"}},
                        "artifacts_created": {"type": "array", "items": {"type": "string"}},
                        "decision_made": {"type": "string", "description": "Key decision made in this turn, or empty string"},
                        "inferred_stage_id": {
                            "type": ["string", "null"],
                            "description": "Stage ID from expected workflow (e.g. S1), or null if unclear",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence in the stage inference",
                        },
                        "evidence_quote": {"type": "string", "description": "1-3 sentence verbatim extract supporting the stage inference"},
                    },
                    "required": ["turn_id", "role", "content_summary", "tools_used",
                                 "artifacts_created", "decision_made", "inferred_stage_id",
                                 "confidence", "evidence_quote"],
                },
            }
        },
        "required": ["turns"],
    },
}


def _call_llm(session_text: str, skill_name: str, stages: list[dict]) -> list[dict]:
    """Call Anthropic API and return the raw turns list."""
    stage_summary = "\n".join(
        f"  {s['stage_id']}: {s['stage_name']} — {s['stage_goal'][:100]}"
        for s in stages
    )

    system = (
        "You are an expert at analysing AI agent session transcripts. "
        "Your job is to convert a raw session into a structured trajectory. "
        "Be precise: use evidence quotes verbatim from the session text. "
        "Never invent content that is not in the session."
    )

    user_prompt = f"""Analyse this session transcript for skill: {skill_name}

## Expected workflow stages for {skill_name}:
{stage_summary}

## Session transcript:
{session_text}

Map each logical turn to the closest expected stage. For each turn:
- Summarise what happened in ≤50 words
- List any tool names called
- Note any files or artefacts created
- Identify the most important decision made (empty string if none)
- Assign the closest expected stage_id (e.g. S1) or null if the turn doesn't map clearly
- Give a confidence score 0.0–1.0 for your stage assignment
- Quote 1–3 sentences verbatim from the turn as evidence

Record all turns via the record_trajectory tool."""

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        system=system,
        tools=[_EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "record_trajectory"},
        messages=[{"role": "user", "content": user_prompt}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "record_trajectory":
            return block.input.get("turns", [])

    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_trajectory(
    session: SessionExport,
    workflow: ExpectedWorkflow,
) -> SessionTrajectory:
    """Extract a structured session trajectory using LLM analysis."""
    session_text, truncated = _serialise_session_for_llm(session)
    raw_turns = _call_llm(session_text, workflow["skill_name"], workflow["stages"])

    turns: list[SessionTurn] = []
    for t in raw_turns:
        confidence = float(t.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        turns.append(
            SessionTurn(
                turn_id=t.get("turn_id", f"T{len(turns)+1}"),
                role=t.get("role", "user"),
                content_summary=t.get("content_summary", ""),
                tools_used=t.get("tools_used", []),
                artifacts_created=t.get("artifacts_created", []),
                decision_made=t.get("decision_made", ""),
                inferred_stage_id=t.get("inferred_stage_id"),
                confidence=confidence,
                evidence_quote=t.get("evidence_quote", ""),
            )
        )

    return SessionTrajectory(
        session_id=session["session_id"],
        skill_name=workflow["skill_name"],
        extracted_at=datetime.now(timezone.utc).isoformat(),
        truncated=truncated,
        turns=turns,
    )


def extract_and_save(
    session: SessionExport,
    workflow: ExpectedWorkflow,
    output_dir: Path,
) -> Path:
    trajectory = extract_trajectory(session, workflow)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{session['session_id']}_trajectory.json"
    out_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract trajectory from a session JSON using LLM analysis."
    )
    parser.add_argument("--session-id", required=True, help="Session ID (matches data/sessions/{id}.json)")
    parser.add_argument("--skill", required=True, choices=["ce-brainstorm", "ce-plan"])
    parser.add_argument("--sessions-dir", default="data/sessions")
    parser.add_argument("--workflows-dir", default="data/expected_workflows")
    parser.add_argument("--output-dir", default="data/trajectories")
    args = parser.parse_args()

    session_path = Path(args.sessions_dir) / f"{args.session_id}.json"
    workflow_path = Path(args.workflows_dir) / f"{args.skill}_workflow.json"

    if not session_path.exists():
        print(f"Error: session not found at {session_path}", file=sys.stderr)
        sys.exit(1)
    if not workflow_path.exists():
        print(f"Error: workflow not found at {workflow_path}", file=sys.stderr)
        sys.exit(1)

    session: SessionExport = json.loads(session_path.read_text())
    workflow: ExpectedWorkflow = json.loads(workflow_path.read_text())

    out_path = extract_and_save(session, workflow, Path(args.output_dir))
    trajectory: SessionTrajectory = json.loads(out_path.read_text())
    print(f"Extracted {len(trajectory['turns'])} turns → {out_path}")
    if trajectory["truncated"]:
        print("  (session was truncated at 100k chars)")


if __name__ == "__main__":
    main()
