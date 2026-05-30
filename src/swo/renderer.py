"""U6: HTML Report Renderer — self-contained static HTML via Jinja2."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from swo.schemas import ReportModel

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _fit_counts(assessments: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"helpful": 0, "harmful": 0, "neutral": 0, "unclear": 0}
    for a in assessments:
        key = a.get("fit_for_knowledge_work", "unclear")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _followed_pct(assessments: list[dict]) -> int:
    if not assessments:
        return 0
    followed = sum(1 for a in assessments if a.get("deviation_type") == "followed")
    return round(100 * followed / len(assessments))


def render_report(model: ReportModel) -> str:
    """Render the HTML report and return the HTML string."""
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html.j2")

    assessments = model["deviations"]["assessments"]
    stage_map = {
        s["stage_id"]: s
        for s in model["expected_workflow"]["stages"]
    }

    return template.render(
        model=model,
        assessments=assessments,
        stage_map=stage_map,
        fit_counts=_fit_counts(assessments),
        followed_pct=_followed_pct(assessments),
        generated_at=model["generated_at"],
    )


def render_and_save(model: ReportModel, output_dir: Path) -> Path:
    html = render_report(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model['session_id']}_report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def build_report_model(
    session_id: str,
    session_title: str | None,
    skill_name: str,
    workflow: dict,
    trajectory: dict,
    deviations: dict,
) -> ReportModel:
    return ReportModel(
        session_id=session_id,
        skill_name=skill_name,
        session_title=session_title,
        generated_at=datetime.now(timezone.utc).isoformat(),
        expected_workflow=workflow,
        trajectory=trajectory,
        deviations=deviations,
    )
