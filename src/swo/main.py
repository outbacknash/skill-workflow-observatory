"""U7: Main Orchestrator — single CLI command ties the full pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    _RICH = True
    console = Console()
except ImportError:
    _RICH = False
    console = None  # type: ignore[assignment]

from swo.analyzer import analyse_and_save
from swo.extractor import _SUPPORTED_SKILLS, extract_and_save as extract_workflow
from swo.exporter import export_by_id, export_by_title, list_recent_sessions, save_export
from swo.renderer import build_report_model, render_and_save
from swo.trajectory import extract_and_save as extract_trajectory


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent  # /Users/hermes/skill-workflow-observatory


def _data_dir(subdir: str) -> Path:
    return _PROJECT_ROOT / "data" / subdir


def _reports_dir() -> Path:
    return _PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    if _RICH:
        console.print(f"  [dim]{msg}[/dim]")
    else:
        print(f"  {msg}")


def _ok(msg: str) -> None:
    if _RICH:
        console.print(f"  [green]✓[/green] {msg}")
    else:
        print(f"  ✓ {msg}")


def _skip(msg: str) -> None:
    if _RICH:
        console.print(f"  [yellow]↩[/yellow] {msg} [dim](cached)[/dim]")
    else:
        print(f"  ↩ {msg} (cached)")


def _step(label: str) -> None:
    if _RICH:
        console.rule(f"[bold]{label}[/bold]", style="dim")
    else:
        print(f"\n── {label} ──")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _step_extract_workflow(skill: str) -> dict:
    out_path = _data_dir("expected_workflows") / f"{skill}_workflow.json"
    if out_path.exists():
        _skip(f"Workflow for {skill}")
        return json.loads(out_path.read_text())
    _log(f"Extracting expected workflow for {skill}…")
    extract_workflow(skill, _data_dir("expected_workflows"))
    _ok(f"Extracted {skill} workflow → {out_path.name}")
    return json.loads(out_path.read_text())


def _step_export_session(session_id: str | None, session_title: str | None) -> tuple[str, dict]:
    if session_id:
        out_path = _data_dir("sessions") / f"{session_id}.json"
    else:
        # We need to resolve the title to an ID first
        export = export_by_title(session_title)
        session_id = export["session_id"]
        out_path = _data_dir("sessions") / f"{session_id}.json"
        if not out_path.exists():
            save_export(export, _data_dir("sessions"))
            _ok(f"Exported session → {out_path.name}")
        else:
            _skip(f"Session {session_id}")
        return session_id, json.loads(out_path.read_text())

    if out_path.exists():
        _skip(f"Session {session_id}")
        return session_id, json.loads(out_path.read_text())

    _log(f"Exporting session {session_id}…")
    export = export_by_id(session_id)
    save_export(export, _data_dir("sessions"))
    _ok(f"Exported session → {out_path.name}")
    return session_id, json.loads(out_path.read_text())


def _step_extract_trajectory(session_id: str, session: dict, workflow: dict) -> dict:
    out_path = _data_dir("trajectories") / f"{session_id}_trajectory.json"
    if out_path.exists():
        _skip("Trajectory")
        return json.loads(out_path.read_text())
    _log("Extracting session trajectory (LLM)…")
    extract_trajectory(session, workflow, _data_dir("trajectories"))
    _ok(f"Trajectory extracted → {out_path.name}")
    return json.loads(out_path.read_text())


def _step_analyse_deviations(session_id: str, workflow: dict, trajectory: dict) -> dict:
    out_path = _data_dir("analyses") / f"{session_id}_deviations.json"
    if out_path.exists():
        _skip("Deviation analysis")
        return json.loads(out_path.read_text())
    _log("Analysing deviations (LLM)…")
    analyse_and_save(workflow, trajectory, _data_dir("analyses"))
    _ok(f"Deviations analysed → {out_path.name}")
    return json.loads(out_path.read_text())


def _step_render_report(
    session_id: str,
    session: dict,
    skill: str,
    workflow: dict,
    trajectory: dict,
    deviations: dict,
) -> Path:
    model = build_report_model(
        session_id=session_id,
        session_title=session.get("title"),
        skill_name=skill,
        workflow=workflow,
        trajectory=trajectory,
        deviations=deviations,
    )
    out_path = render_and_save(model, _reports_dir())
    _ok(f"Report rendered → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    skill: str,
    session_id: str | None = None,
    session_title: str | None = None,
    open_browser: bool = False,
) -> Path:
    if _RICH:
        console.print(f"\n[bold]Skill Workflow Observatory[/bold]  skill=[cyan]{skill}[/cyan]\n")

    _step("Step 1: Expected Workflow")
    workflow = _step_extract_workflow(skill)

    _step("Step 2: Session Export")
    session_id, session = _step_export_session(session_id, session_title)

    _step("Step 3: Trajectory Extraction")
    trajectory = _step_extract_trajectory(session_id, session, workflow)

    _step("Step 4: Deviation Analysis")
    deviations = _step_analyse_deviations(session_id, workflow, trajectory)

    _step("Step 5: Render HTML Report")
    report_path = _step_render_report(
        session_id, session, skill, workflow, trajectory, deviations
    )

    if _RICH:
        console.print(f"\n[bold green]Done![/bold green]  {report_path.absolute()}\n")
    else:
        print(f"\nDone! {report_path.absolute()}")

    if open_browser:
        try:
            subprocess.call(["open", str(report_path.absolute())])
        except FileNotFoundError:
            print(f"Could not open browser automatically. Open manually: {report_path.absolute()}")

    return report_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Skill Workflow Observatory — generate an HTML report for a compound-engineering session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m swo.main --skill ce-plan --session-title 'My Planning Session' --open\n"
            "  python -m swo.main --skill ce-brainstorm --session-id sess_abc123\n"
            "  python -m swo.main --list\n"
        ),
    )
    parser.add_argument(
        "--skill",
        choices=list(_SUPPORTED_SKILLS),
        help="Skill to analyse",
    )
    id_group = parser.add_mutually_exclusive_group()
    id_group.add_argument("--session-id", help="Session ID from state.db")
    id_group.add_argument("--session-title", help="Session title from state.db")
    id_group.add_argument(
        "--list",
        action="store_true",
        help="List recent CLI sessions and exit",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the HTML report in the default browser after generation",
    )
    args = parser.parse_args()

    if args.list:
        sessions = list_recent_sessions()
        if not sessions:
            print("No CLI sessions found.")
            return
        from datetime import datetime as dt
        print(f"{'ID':<30} {'Title':<45} {'Messages':>8}")
        print("-" * 85)
        for s in sessions:
            started = dt.fromtimestamp(s["started_at"]).strftime("%Y-%m-%d %H:%M")
            title = (s["title"] or "(untitled)")[:44]
            print(f"{s['id']:<30} {title:<45} {s['message_count']:>8}  {started}")
        return

    if not args.skill:
        parser.error("--skill is required (ce-brainstorm or ce-plan)")
    if not args.session_id and not args.session_title:
        parser.error("Either --session-id or --session-title is required")

    try:
        run_pipeline(
            skill=args.skill,
            session_id=args.session_id,
            session_title=args.session_title,
            open_browser=args.open,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Run with --list to see available sessions.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
