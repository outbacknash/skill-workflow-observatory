"""U5: Deviation Analyzer — map expected workflow stages to actual trajectory."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from swo.schemas import (
    DeviationAssessment,
    DeviationReport,
    ExpectedWorkflow,
    SessionTrajectory,
)

_MODEL = "claude-sonnet-4-6"
_RUBRIC_PATH = Path(__file__).parent.parent.parent / "data/analyses/deviation_rubric.md"

# ---------------------------------------------------------------------------
# LLM tool schema
# ---------------------------------------------------------------------------

_ANALYSIS_TOOL = {
    "name": "record_deviation_analysis",
    "description": "Record deviation assessments for each expected workflow stage.",
    "input_schema": {
        "type": "object",
        "properties": {
            "executive_verdict": {
                "type": "string",
                "description": (
                    "One paragraph assessing whether the skill served this session well overall, "
                    "highlighting the 2–3 most important adaptations."
                ),
            },
            "assessments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "expected_stage_id": {
                            "type": ["string", "null"],
                            "description": "Stage ID from expected workflow (null for invented turns)",
                        },
                        "actual_turn_range": {
                            "type": "string",
                            "description": "e.g. 'T3–T5', or empty string if stage was skipped",
                        },
                        "deviation_type": {
                            "type": "string",
                            "enum": [
                                "followed", "skipped", "compressed", "reordered",
                                "over-applied", "under-applied", "invented", "misclassified",
                            ],
                        },
                        "fit_for_knowledge_work": {
                            "type": "string",
                            "enum": ["helpful", "harmful", "neutral", "unclear"],
                        },
                        "rationale": {
                            "type": "string",
                            "description": "1–3 sentence explanation referencing evidence from the session",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "possible_skill_change": {
                            "type": "string",
                            "description": "One of: patch, wrapper, invocation-guide, none, or empty string",
                        },
                    },
                    "required": [
                        "expected_stage_id", "actual_turn_range", "deviation_type",
                        "fit_for_knowledge_work", "rationale", "confidence",
                        "possible_skill_change",
                    ],
                },
            },
        },
        "required": ["executive_verdict", "assessments"],
    },
}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _load_rubric() -> str:
    rubric_path = _RUBRIC_PATH
    if rubric_path.exists():
        return rubric_path.read_text(encoding="utf-8")
    return "(rubric not found — use general judgment)"


def _call_llm(
    workflow: ExpectedWorkflow,
    trajectory: SessionTrajectory,
    rubric: str,
) -> tuple[str, list[dict]]:
    """Call Anthropic API. Returns (executive_verdict, raw_assessments)."""
    stages_text = "\n".join(
        f"  {s['stage_id']}: {s['stage_name']} [{s['required_vs_optional']}] — {s['stage_goal'][:120]}"
        for s in workflow["stages"]
    )

    turns_text = "\n\n".join(
        f"[{t['turn_id']} | {t['role'].upper()}] stage={t['inferred_stage_id'] or '?'} "
        f"confidence={t['confidence']:.2f}\n"
        f"  Summary: {t['content_summary']}\n"
        f"  Tools: {', '.join(t['tools_used']) or 'none'}\n"
        f"  Artifacts: {', '.join(t['artifacts_created']) or 'none'}\n"
        f"  Quote: {t['evidence_quote']}"
        for t in trajectory["turns"]
    )

    system = (
        "You are an expert workflow analyst specialising in AI agent behaviour. "
        "You assess how well an AI agent followed a prescribed skill workflow and "
        "whether deviations helped or hurt the user's goals. "
        "Be precise, evidence-based, and calibrated. Reward intelligent adaptation."
    )

    user_prompt = f"""Analyse how the agent followed the {workflow['skill_name']} workflow in this session.

## Deviation Rubric:
{rubric[:3000]}

## Expected workflow stages for {workflow['skill_name']}:
{stages_text}

## Actual session trajectory:
{turns_text}

For EACH expected stage, produce a deviation assessment. Also add assessments for
any "invented" turns (agent behaviour not in any expected stage).

Remember: skipping a stage can be helpful if the context warranted it. Reward
intelligent adaptation, not blind compliance.

Record your analysis via the record_deviation_analysis tool."""

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        system=system,
        tools=[_ANALYSIS_TOOL],
        tool_choice={"type": "tool", "name": "record_deviation_analysis"},
        messages=[{"role": "user", "content": user_prompt}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "record_deviation_analysis":
            verdict = block.input.get("executive_verdict", "")
            raw = block.input.get("assessments", [])
            return verdict, raw

    return "", []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_deviations(
    workflow: ExpectedWorkflow,
    trajectory: SessionTrajectory,
) -> DeviationReport:
    rubric = _load_rubric()
    executive_verdict, raw_assessments = _call_llm(workflow, trajectory, rubric)

    assessments: list[DeviationAssessment] = []
    for a in raw_assessments:
        confidence = max(0.0, min(1.0, float(a.get("confidence", 0.5))))
        assessments.append(
            DeviationAssessment(
                expected_stage_id=a.get("expected_stage_id"),
                actual_turn_range=a.get("actual_turn_range", ""),
                deviation_type=a.get("deviation_type", "skipped"),
                fit_for_knowledge_work=a.get("fit_for_knowledge_work", "unclear"),
                rationale=a.get("rationale", ""),
                confidence=confidence,
                possible_skill_change=a.get("possible_skill_change", "none"),
            )
        )

    return DeviationReport(
        session_id=trajectory["session_id"],
        skill_name=workflow["skill_name"],
        analyzed_at=datetime.now(timezone.utc).isoformat(),
        executive_verdict=executive_verdict,
        assessments=assessments,
    )


def analyse_and_save(
    workflow: ExpectedWorkflow,
    trajectory: SessionTrajectory,
    output_dir: Path,
) -> Path:
    report = analyse_deviations(workflow, trajectory)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{trajectory['session_id']}_deviations.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyse workflow deviations between expected stages and actual trajectory."
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--skill", required=True, choices=["ce-brainstorm", "ce-plan"])
    parser.add_argument("--workflows-dir", default="data/expected_workflows")
    parser.add_argument("--trajectories-dir", default="data/trajectories")
    parser.add_argument("--output-dir", default="data/analyses")
    args = parser.parse_args()

    workflow_path = Path(args.workflows_dir) / f"{args.skill}_workflow.json"
    trajectory_path = Path(args.trajectories_dir) / f"{args.session_id}_trajectory.json"

    for p in (workflow_path, trajectory_path):
        if not p.exists():
            print(f"Error: not found: {p}", file=sys.stderr)
            sys.exit(1)

    workflow: ExpectedWorkflow = json.loads(workflow_path.read_text())
    trajectory: SessionTrajectory = json.loads(trajectory_path.read_text())

    out_path = analyse_and_save(workflow, trajectory, Path(args.output_dir))
    report: DeviationReport = json.loads(out_path.read_text())
    print(f"Analysed {len(report['assessments'])} deviations → {out_path}")
    print(f"\nVerdict: {report['executive_verdict'][:200]}...")


if __name__ == "__main__":
    main()
