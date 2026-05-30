"""U2: Skill Doc Extractor — parse SKILL.md into ExpectedWorkflow JSON."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from swo.schemas import ExpectedWorkflow, ExpectedWorkflowStage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PLUGIN_ROOT = (
    Path.home()
    / ".claude/plugins/cache/compound-engineering-plugin/compound-engineering/3.9.3"
)
_INSTALLED_PLUGINS_JSON = Path.home() / ".claude/plugins/installed_plugins.json"
_SUPPORTED_SKILLS = ("ce-brainstorm", "ce-plan")

_REQUIRED_KEYWORDS = re.compile(
    r"\b(CRITICAL|REQUIRED|STOP\.|STOP —|non-optional|mandatory|must)\b"
)
_OPTIONAL_KEYWORDS = re.compile(r"\b(optional|Conditional|skip when|only when|when applicable)\b", re.IGNORECASE)

# Artifact patterns: file paths (.md, .html, .json) or named artifacts
_ARTIFACT_RE = re.compile(
    r"(?:"
    r"docs/(?:plans|brainstorms)/[^\s\)\"\'`]+"
    r"|STRATEGY\.md|requirements\.md|[A-Z_]+\.md"
    r"|plan file|requirements doc|report\.html|workflow_trace\.json"
    r")"
)


# ---------------------------------------------------------------------------
# Plugin metadata helpers
# ---------------------------------------------------------------------------

def _get_plugin_metadata() -> tuple[str, str]:
    """Return (version, git_sha) from installed_plugins.json."""
    try:
        data = json.loads(_INSTALLED_PLUGINS_JSON.read_text())
        entry = data["plugins"]["compound-engineering@compound-engineering-plugin"][0]
        return entry["version"], entry["gitCommitSha"]
    except Exception:
        return "unknown", "unknown"


def _get_plugin_root() -> Path:
    env_override = os.environ.get("CE_PLUGIN_PATH")
    return Path(env_override) if env_override else _DEFAULT_PLUGIN_ROOT


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _classify_required(heading: str, body: str) -> str:
    combined = heading + "\n" + body[:500]
    if _REQUIRED_KEYWORDS.search(combined):
        return "required"
    if _OPTIONAL_KEYWORDS.search(combined):
        return "conditional"
    return "optional"


def _infer_turn_pattern(body: str) -> str:
    body_lower = body.lower()
    if "parallel" in body_lower and ("task " in body_lower or "agent" in body_lower):
        return "parallel-research"
    if "ask" in body_lower and "one" in body_lower and "question" in body_lower:
        return "one-question-at-a-time"
    if "synthesi" in body_lower or "consolidat" in body_lower:
        return "synthesis-summary"
    if "confidence" in body_lower and "check" in body_lower:
        return "confidence-check"
    if "write" in body_lower and ("plan" in body_lower or "file" in body_lower):
        return "artifact-write"
    return "prose-reasoning"


def _extract_artifact(body: str) -> str:
    match = _ARTIFACT_RE.search(body)
    return match.group(0) if match else ""


def _first_paragraph(text: str, max_chars: int = 300) -> str:
    """Return the first non-empty paragraph of text, truncated."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ""
    return paragraphs[0][:max_chars]


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def _parse_skill_md(skill_md_path: Path) -> list[ExpectedWorkflowStage]:
    """Parse SKILL.md and return a list of ExpectedWorkflowStage dicts."""
    content = skill_md_path.read_text(encoding="utf-8")

    # Split on top-level ### Phase headings (### Phase N: Name)
    phase_pattern = re.compile(r"^(### Phase [^\n]+)", re.MULTILINE)
    parts = phase_pattern.split(content)

    # parts is: [preamble, heading1, body1, heading2, body2, ...]
    stages: list[ExpectedWorkflowStage] = []
    stage_counter = 0

    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""

        # Strip heading prefix for clean name
        stage_name = re.sub(r"^### Phase \d+[.\d]*:\s*", "", heading)

        stage_counter += 1
        stage_id = f"S{stage_counter}"

        stages.append(
            ExpectedWorkflowStage(
                stage_id=stage_id,
                stage_name=stage_name,
                stage_goal=_first_paragraph(body),
                required_vs_optional=_classify_required(heading, body),
                turn_pattern=_infer_turn_pattern(body),
                artifact_expectation=_extract_artifact(body),
                source_section=heading,
                source_quote=body[:200].strip(),
            )
        )

    return stages


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_workflow(skill_name: str) -> ExpectedWorkflow:
    """
    Extract the expected workflow for a supported skill.

    Returns an ExpectedWorkflow dict ready for JSON serialisation.
    Raises FileNotFoundError if the SKILL.md cannot be found.
    """
    if skill_name not in _SUPPORTED_SKILLS:
        raise ValueError(
            f"Unsupported skill '{skill_name}'. Supported: {_SUPPORTED_SKILLS}"
        )

    plugin_root = _get_plugin_root()
    skill_md_path = plugin_root / "skills" / skill_name / "SKILL.md"

    if not skill_md_path.exists():
        raise FileNotFoundError(
            f"SKILL.md not found at expected path: {skill_md_path}\n"
            f"Set CE_PLUGIN_PATH env var to override the plugin root."
        )

    version, git_sha = _get_plugin_metadata()
    stages = _parse_skill_md(skill_md_path)

    return ExpectedWorkflow(
        skill_name=skill_name,
        plugin_version=version,
        plugin_git_sha=git_sha,
        skill_doc_path=str(skill_md_path),
        extracted_at=datetime.now(timezone.utc).isoformat(),
        stages=stages,
    )


def extract_and_save(skill_name: str, output_dir: Path) -> Path:
    """Extract workflow and write JSON to output_dir/{skill_name}_workflow.json."""
    workflow = extract_workflow(skill_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{skill_name}_workflow.json"
    out_path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract expected workflow stages from a compound-engineering skill doc."
    )
    parser.add_argument(
        "skill",
        choices=list(_SUPPORTED_SKILLS),
        help="Skill name to extract (e.g. ce-plan)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/expected_workflows",
        help="Directory to write the workflow JSON (default: data/expected_workflows)",
    )
    args = parser.parse_args()

    try:
        out_path = extract_and_save(args.skill, Path(args.output_dir))
        workflow = json.loads(out_path.read_text())
        print(f"Extracted {len(workflow['stages'])} stages → {out_path}")
        for stage in workflow["stages"]:
            req = stage["required_vs_optional"]
            print(f"  {stage['stage_id']}: {stage['stage_name']} [{req}]")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
