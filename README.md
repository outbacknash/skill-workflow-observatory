# Skill Workflow Observatory

A static HTML report generator that makes compound-engineering skill workflow execution visible.

Given one skill (ce-brainstorm or ce-plan) and one real hermes session, it produces a self-contained HTML report showing what the skill intended, what actually happened, where sessions diverged, and whether those deviations helped or hurt knowledge work.

## Setup

```bash
source ~/.hermes/hermes-agent/venv/bin/activate
cd /Users/hermes/skill-workflow-observatory
```

## Usage

```bash
# List recent sessions
python -m swo.exporter --list

# Generate a report
python -m swo.main --skill ce-plan --session-title "Session Title" --open
```

See [CLAUDE.md](CLAUDE.md) for full documentation.
