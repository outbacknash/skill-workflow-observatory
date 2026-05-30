# Skill Workflow Observatory

A Python pipeline that produces static HTML reports comparing expected vs actual compound-engineering skill workflow execution in hermes/Claude sessions.

## Purpose

Given one compound-engineering skill (ce-brainstorm or ce-plan) and one real hermes session, the observatory generates a self-contained HTML report showing:
- What the skill intended (expected workflow stages)
- What the agent actually did (session trajectory)
- Where and how the session diverged (deviation assessments)
- Whether deviations helped or hurt knowledge-work goals
- Adaptation recommendations

## Environment Setup

This project uses the hermes venv to access `hermes_state.SessionDB`, `jinja2`, and `anthropic`:

```bash
source ~/.hermes/hermes-agent/venv/bin/activate
```

After activation, run from the project root:

```bash
python -m swo.main --skill ce-plan --session-title "My Session Title"
```

## Quick Reference

```bash
# List recent hermes CLI sessions
python -m swo.exporter --list

# Export a specific session
python -m swo.exporter --session-title "Session Title"

# Extract expected workflow for a skill
python -m swo.extractor ce-plan

# Run full pipeline (produces reports/{session_id}_report.html)
python -m swo.main --skill ce-plan --session-title "Session Title" --open
```

## Project Structure

```
src/swo/
├── schemas.py      # TypedDict definitions for all data models
├── extractor.py    # Skill doc → ExpectedWorkflow JSON (U2)
├── exporter.py     # hermes state.db → session JSON (U3)
├── trajectory.py   # session JSON → turn-by-turn timeline (U4)
├── analyzer.py     # expected + trajectory → deviation assessments (U5)
├── renderer.py     # all models → static HTML report (U6)
├── main.py         # CLI orchestrator (U7)
└── templates/
    └── report.html.j2   # Jinja2 HTML template
```

## Key Design Decisions

- **Local skill docs**: Read from `~/.claude/plugins/cache/compound-engineering-plugin/` — no network requests
- **SessionDB access**: `sys.path` injection to import `hermes_state` from the hermes editable install
- **LLM tasks**: Anthropic SDK with Claude Sonnet 4.6, structured JSON output mode
- **v1 scope**: ce-brainstorm and ce-plan only; one session at a time; static HTML only

## Running Tests

```bash
source ~/.hermes/hermes-agent/venv/bin/activate
cd /Users/hermes/skill-workflow-observatory
python -m pytest tests/ -v
```
