## Residual Review Findings

Source: ce-code-review autofix run `/tmp/compound-engineering/ce-code-review/20260530-144938-347d989f/`  
HEAD at review time: `5e6aaba`  
Applied safe_auto fixes: 5 (pyproject.toml pins, pythonpath config, _SUPPORTED_SKILLS import, FileNotFoundError guard, rubric warnings.warn)

### P1 — Critical (should fix before first production use)

- **P1 `src/swo/extractor.py:109`** — Code-block phase headings extracted as real stages: `re.MULTILINE` regex matches `### Phase N:` inside triple-backtick fenced code blocks, silently corrupting the stage list. Fix: strip fenced-code-block regions before applying the heading regex. Class: `gated_auto`.

- **P1 `src/swo/trajectory.py:173`** — No try/except around `client.messages.create()`: `RateLimitError`, `APIConnectionError`, `APIStatusError` crash with raw traceback. Fix: wrap in `except anthropic.APIError as e: raise RuntimeError(...)`. Class: `gated_auto`.

- **P1 `src/swo/analyzer.py:149`** — Same as above in `analyzer._call_llm()`. Class: `gated_auto`.

- **P1 `tests/test_trajectory.py` + `tests/test_analyzer.py`** — LLM no-tool-use response path and API exception paths completely untested. The `for block in response.content` loop is never exercised under test. Fix: add tests mocking `client.messages.create` to return a no-tool-use response and raising `anthropic.APIError`. Class: `manual`.

### P2 — High (quality/correctness improvements)

- **P2 `src/swo/main.py:94`** — Title branch calls `export_by_title()` before cache check: if session is deleted from DB after first export, `ValueError` is raised even though cache exists on disk. Fix: resolve session ID from cache filename first, fall back to DB only when cache is absent. Class: `gated_auto`.

- **P2 `src/swo/trajectory.py:83`** — Truncation false-negative: `raw_size` sums only `content` field lengths, ignoring `tool_calls` JSON bulk — `truncated=False` when actual rendered text silently exceeds 100k. Fix: include `len(json.dumps(msg.get("tool_calls", [])))` in the size sum. Class: `gated_auto`.

- **P2 `src/swo/main.py:117`** — Partial-write cache poisoning: if LLM call raises mid-write, corrupt JSON is cached and read as valid on next run. Fix: write to a `.tmp` file then `rename()` atomically. Class: `gated_auto`.

- **P2 `src/swo/exporter.py:14`** — `sys.path` mutated at module import time: any importer of `exporter` gets `~/.hermes/hermes-agent` injected regardless of whether `get_db()` is called. Fix: move `sys.path.insert` inside `get_db()` behind a `_db_initialised` guard. Class: `gated_auto`.

- **P2 `src/swo/main.py:162`** — `run_pipeline()` returns a bare `Path` and uses hardcoded I/O paths, preventing programmatic or parallel invocation. Fix: return a structured `PipelineResult` TypedDict (`report_path`, `session_id`, `skill`, `deviations_count`); accept optional `data_dir` / `reports_dir` overrides. Class: `manual`.

- **P2 `src/swo/extractor.py:18`** — `_DEFAULT_PLUGIN_ROOT` hardcodes plugin version literal `3.9.3`: silently reads stale SKILL.md after `hermes update`. Fix: derive version dynamically from `~/.claude/plugins/installed_plugins.json`. Class: `gated_auto`.
