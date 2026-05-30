"""Shared data model definitions for the Skill Workflow Observatory."""

from typing import Literal, TypedDict


# ---------------------------------------------------------------------------
# U2 — Expected workflow model
# ---------------------------------------------------------------------------

class ExpectedWorkflowStage(TypedDict):
    stage_id: str                    # Sequential: "S1", "S2", ...
    stage_name: str                  # Heading text
    stage_goal: str                  # First paragraph after heading (≤300 chars)
    required_vs_optional: Literal["required", "optional", "conditional"]
    turn_pattern: str                # e.g. "one-question-at-a-time", "parallel-research"
    artifact_expectation: str        # Named output file or artifact (empty string if none)
    source_section: str              # Original heading text (may differ from stage_name after normalization)
    source_quote: str                # First 200 chars of section body


class ExpectedWorkflow(TypedDict):
    skill_name: str
    plugin_version: str
    plugin_git_sha: str
    skill_doc_path: str
    extracted_at: str                # ISO 8601
    stages: list[ExpectedWorkflowStage]


# ---------------------------------------------------------------------------
# U3 — Session export
# ---------------------------------------------------------------------------

class SessionMessage(TypedDict):
    id: int
    role: str                        # "user" | "assistant" | "tool"
    content: str | None
    tool_calls: list[dict] | None    # Parsed from JSON string
    tool_name: str | None
    timestamp: float


class SessionExport(TypedDict):
    session_id: str
    source: str
    model: str | None
    title: str | None
    started_at: float
    ended_at: float | None
    message_count: int
    exported_at: str                 # ISO 8601
    messages: list[SessionMessage]


# ---------------------------------------------------------------------------
# U4 — Session trajectory
# ---------------------------------------------------------------------------

class SessionTurn(TypedDict):
    turn_id: str                     # "T1", "T2", ...
    role: str                        # "user" | "agent"
    content_summary: str             # ≤50 words
    tools_used: list[str]            # Tool names called in this turn
    artifacts_created: list[str]     # File paths or artifact names mentioned
    decision_made: str               # Key decision, or empty string
    inferred_stage_id: str | None    # Matches ExpectedWorkflowStage.stage_id, or null
    confidence: float                # 0.0–1.0
    evidence_quote: str              # 1–3 sentence verbatim extract


class SessionTrajectory(TypedDict):
    session_id: str
    skill_name: str
    extracted_at: str
    truncated: bool                  # True if session was truncated before extraction
    turns: list[SessionTurn]


# ---------------------------------------------------------------------------
# U5 — Deviation assessment
# ---------------------------------------------------------------------------

DeviationType = Literal[
    "followed",
    "skipped",
    "compressed",
    "reordered",
    "over-applied",
    "under-applied",
    "invented",
    "misclassified",
]

FitScore = Literal["helpful", "harmful", "neutral", "unclear"]


class DeviationAssessment(TypedDict):
    expected_stage_id: str | None    # null for "invented" entries
    actual_turn_range: str           # e.g. "T3–T5" or "" if skipped
    deviation_type: DeviationType
    fit_for_knowledge_work: FitScore
    rationale: str
    confidence: float
    possible_skill_change: str       # "patch" | "wrapper" | "invocation-guide" | "none" | ""


class DeviationReport(TypedDict):
    session_id: str
    skill_name: str
    analyzed_at: str
    executive_verdict: str
    assessments: list[DeviationAssessment]


# ---------------------------------------------------------------------------
# U6 — Report model (aggregated input to renderer)
# ---------------------------------------------------------------------------

class ReportModel(TypedDict):
    session_id: str
    skill_name: str
    session_title: str | None
    generated_at: str
    expected_workflow: ExpectedWorkflow
    trajectory: SessionTrajectory
    deviations: DeviationReport
