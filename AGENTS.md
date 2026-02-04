# AGENTS.md

Guide for coding agents operating inside the LangGraph Plan-Act-Review workspace.
Keep this file handy before editing code or running commands.

## Purpose & High-Level Expectations
- Follow the Plan-Act-Review workflow conventions described in `CLAUDE.md` and `agent_structure.md`.
- Assume multi-turn conversations; preserve planner/actor/reviewer separation and do not bypass nodes.
- Always pass LangChain `RunnableConfig` objects down every LLM call so Langfuse callbacks stay wired.
- Normalize observability metadata: `skill` should be "none" when no skill is active and `auto_select` stored as a lowercase string.
- Flush Langfuse traces via `src/observability.flush()` whenever a workflow run completes, especially in CLI scripts or tests.
- UI copy is Korean; keep new user-facing strings Korean unless a feature explicitly targets another locale.

## Repository & Environment Setup
- Python 3.11+; dependency graph managed with `uv` (see `pyproject.toml` and `uv.lock`).
- Install or refresh dependencies after pulling: `uv sync --extra dev` (or `--extra eval` for evaluation suites).
- Config lives in `config.yaml`; copy from `config.yaml.example` and keep secrets in `.env` (auto-loaded via `python-dotenv` inside `app.py`).
- LiteLLM proxy must be running before the Streamlit app: `litellm --model gpt-4 --port 4000`.
- Launch the UI with `uv run streamlit run app.py`; Streamlit session state stores `WorkflowComponents` and conversation history.
- Generate the workflow graph image with `python visualize_graph.py` when docs or diagrams need refreshing.
- Use `uv add <package>` / `uv remove <package>` instead of pip to keep `uv.lock` canonical.
- Avoid committing `config.yaml`, `.env`, or secrets; they are gitignored on purpose.

## Build, Lint, Format, and Runtime Commands
- **Install deps:** `uv sync` (append `--extra dev` in CI parity).
- **Run app:** `litellm --model gpt-4 --port 4000` (Terminal 1) and `uv run streamlit run app.py` (Terminal 2).
- **Lint:** `uv run ruff check .` (Ruff uses target `py311`, rules `E,F,I,W`, line-length 100).
- **Format:** `uv run ruff format .` for auto-formatting; CI runs `uv run ruff format --check .`.
- **Type-ish expectations:** no mypy yet, but add type hints consistent with existing Optional/list[str]/Literal usage.
- **Graph viz:** `python visualize_graph.py` produces `workflow_graph.png` for architecture docs.
- **Observability sanity:** call `create_callback()` before LangGraph runs and `src/observability.flush()` afterward.

## Testing Guide (Pytest via uv)
- Default all tests through uv: `uv run pytest`.
- Unit tests live under `tests/unit`; they do not require external LLMs.
- Evaluation tests under `tests/eval` invoke LiteLLM + DeepEval; export `DEEPEVAL_API_KEY` (usually via `.env`).
- Quick unit pass: `uv run pytest tests/unit -v`.
- Eval-only run: `uv run pytest tests/eval -v --tb=short`.
- Markers defined in `pyproject.toml`: `@pytest.mark.unit` and `@pytest.mark.eval`; combine with `-m "unit"` to narrow scope.
- To run a single test function: `uv run pytest tests/unit/test_routing.py::test_should_continue_plan -vv`.
- Pattern-based filtering works too: `uv run pytest -k "plan_node and not eval"`.
- CI workflow (`.github/workflows/test.yml`) installs uv, syncs dev deps, runs Ruff lint+format check, then executes `uv run pytest tests/unit/ -v`; evaluation suite only runs on main or manual dispatch.
- Golden datasets for plan/act/review/e2e validations live in `tests/datasets/*.json`; keep them deterministic.

## Observability, LLM, and Streaming Stack
- `src/llm/client.py` wraps `langchain-openai.ChatOpenAI` and automatically normalizes Base URLs to `/v1`; always pass `config` through to keep tracing.
- Use `WorkflowComponents` to bundle `LLMClient`, `PromptRegistry`, `SkillRegistry`, `ToolRegistry`, optional `initial_skill`, and a `StreamCallback`.
- When you add nodes or custom flows, always emit `StreamCallback` events so the Streamlit UI stays reactive (plan_ready, step_start, tool_call, review_ready, error).
- Node classes must satisfy the `BaseNode` protocol (`__call__(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]`).
- Keep the Plan-Act-Review loop intact: `should_continue()` gates plan/act/review/end transitions, and `after_review()` either loops back to plan or terminates.
- Tool execution goes through `execute_tool_calls()` which handles the special `use_skill` tool; include that definition whenever you expose tools to the LLM.
- When sending observability metadata, trim user input to ~100 chars (see `app.py`) and attach `skill`/`auto_select` normalized values.

## Workflow Architecture Cliff Notes
- Planner builds a list of `PlanStep` dictionaries; `NodeMixin._build_skill_selection_block()` only shows up when auto-select is active and no skill is chosen.
- Actor marks steps `in_progress`, streams tokens via `StreamCallback.token`, executes tools, and writes `result` plus `status = "completed"` back onto the plan.
- Reviewer ingests `_format_plan_for_review()` summaries, returns `is_complete`, `final_answer`, `error`, and `key_facts`; those `key_facts` feed the next turn's actor/reviewer context.
- `ConversationHistoryManager` maintains sliding windows (full detail for the latest turn, summaries for older ones); `ConversationContextBuilder` formats role-specific history blocks.
- Skills come from YAML templates under `src/skills/templates`; each skill can constrain tools and add prompt context.
- Prompts are versioned YAML files under `src/prompts/templates`; use `{role}_v{N}.yaml` naming so `PromptRegistry` can parse versions.
- Any workflow extension should stick to the `WorkflowComponents` bundle to avoid leaking individual dependencies everywhere.
- `StreamCallback` should only issue events when a handler is registered; guard for `is_active` before streaming tokens.
- Remember to update `workflow_graph.png` and `agent_structure.md` when you modify node wiring significantly.

## Runtime Configuration & Secrets
- `config.yaml` controls LLM settings, prompt directories, workflow iteration limits, and conversation history windows.
- `config.yaml` is gitignored; keep developer-specific overrides there, not in committed code.
- Observability config sits under `observability` in `config.yaml`; `create_callback()` resolves providers (`langfuse` currently) and returns both handler + tracing context.
- Always call `flush()` after runs to persist traces and close propagation contexts.
- `.env` is loaded at `app.py` startup via `load_dotenv()` so agents can assume environment variables are ready; never hardcode keys.
- Workflow metadata going to observability must use normalized strings: `skill="none"` and `auto_select="true"/"false"`.
- Keep `Langfuse` keys (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`) out of history and logs.

## Code Style – General
- Ruff enforces 100 character lines and import sorting (E,F,I,W); match existing grouping: stdlib, third-party, local, each separated by one blank line.
- Prefer dataclasses (`@dataclass`) for structured data (see `Skill`, `WorkflowComponents`, `StreamEvent`).
- Every module begins with a concise triple-double-quoted docstring describing its intent.
- Stick with f-strings for string interpolation; avoid `%` formatting.
- Use `Path` from `pathlib` for filesystem paths; open files with explicit UTF-8 encoding.
- Avoid top-level side effects; expose functions/classes and guard runtime code under `if __name__ == "__main__":` if needed.
- Keep UI messaging (Streamlit) inside `app.py`; new workflow logic belongs in `src/workflow`.

## Code Style – Typing & Interfaces
- Type everything: function signatures, return types, and class attributes; prefer `list[str]`/`dict[str, Any]` literal generics (PEP 585).
- Use `Optional[...]` when `None` appears in the data flow and annotate dictionaries with `TypedDict` or dataclasses when persistent.
- Protocols (`BaseNode`) and mixins should stay lightweight; use `runtime_checkable` when other modules need `isinstance` checks.
- When you need callables for LangChain, keep method signatures consistent and ensure `config: RunnableConfig` is optional but passed through.
- Use `Literal` for finite return values (`Literal["act", "review", ...]` in `should_continue`).
- Dataclasses should default to `field(default_factory=list)` or similar to avoid mutable default arguments.

## Code Style – Imports & Module Layout
- Module-level `logger = logging.getLogger(__name__)` appears near the top; never instantiate loggers inside functions.
- Keep import groups ordered: stdlib → third-party → first-party; alphabetical inside each block.
- Avoid wildcard imports; only import what you use.
- Re-export symbols via `__all__` only when needed (see `src/observability/__init__.py`).

## Code Style – Naming & Structure
- Classes use `CamelCase`, functions/variables use `snake_case`, constants are `UPPER_SNAKE_CASE`.
- Workflow state keys (`plan`, `current_step_index`, `review_key_facts`) should stay consistent; reuse the same keys when extending state.
- Keep prompt placeholders descriptive (`{skill_selection_block}`) and document them in `NodeMixin` helpers.
- Tools should have descriptive names and include JSON schemas describing arguments (OpenAI function format).
- Tests follow `test_<feature>.py` naming; prefer descriptive test function names over numeric suffixes.

## Code Style – Error Handling & Logging
- Prefer raising `ValueError`/`RuntimeError` for programmer errors (see `PromptRegistry.get`); return structured dicts for user-facing tool errors.
- Wrap risky sections in `try/except` and log via `logger.exception` before surfacing a user-visible error (see `_run_workflow_streaming`).
- Avoid bare `except`; catch specific exceptions when possible or re-raise after logging.
- Never `print()` for diagnostics; rely on logging and Streamlit status containers.
- When executing tools, return JSON-serializable payloads and include `success`/`error` fields so the UI can render them.
- Propagate errors through the workflow state (`state["error"]`) so `should_continue` can terminate gracefully.

## Code Style – Streamlit & UI Conventions
- Use `st.status`, `st.info`, `st.caption`, etc., mirroring the current UI patterns; respect Korean copy guidelines.
- Store mutable UI state in dictionaries, as `_run_workflow_streaming` does for containers/token buffers.
- Truncate large tool results before rendering (see `app.py`: limit to 500 chars).
- Expose skill/tool metadata in the sidebar; when adding new registries, surface them via Streamlit controls consistent with existing sections.
- When adding settings, keep them inside the sidebar and persist them in `st.session_state`.

## Code Style – Tests
- Structure fixtures inside `tests/conftest.py` (root) or scoped `tests/eval/conftest.py`; keep them deterministic.
- Mark integration tests that call LLMs with `@pytest.mark.eval` to avoid accidental CI cost.
- Use DeepEval utilities inside `tests/eval` only when `DEEPEVAL_API_KEY` is configured; otherwise skip gracefully.
- Prefer asserting on structured data (plans, tool outputs) rather than raw strings to keep tests stable.
- Keep golden JSON fixtures (`tests/datasets/*.json`) human-readable and sorted; document schema inside the fixture file header comment if you add new fields.

## Registries, Skills, and Prompts
- Add prompts under `src/prompts/templates/{role}_v{N}.yaml`; set the active version via `PromptRegistry.set_active_version` or the Streamlit UI.
- Skills belong in `src/skills/templates/*.yaml`; include `name`, `description`, `trigger`, optional `prompt`, and `tools` arrays.
- When new skills filter tools, ensure the corresponding tool names exist in `ToolRegistry`; missing definitions resolve to `None` and should be avoided.
- `SkillRegistry.parse_input` strips the trigger from the user message; keep triggers unique and start them with `/`.
- Reload registries with `.reload()` after modifying YAML files during runtime (buttons already exist in the UI).
- Expand tool coverage by editing `src/tools/base.py` and registering functions plus OpenAI-compatible JSON schemas.

## Git & Workflow Hygiene
- Never force-reset or discard user changes; keep the working tree clean and commit only what the task requires.
- Match the formatting of existing files instead of restyling entire modules.
- Run `uv run ruff format .` before committing to avoid CI failures.
- If you add dependencies, update both `pyproject.toml` and `uv.lock` using `uv sync`.
- Document notable workflow changes in `future_plan.md` or README sections when appropriate.

## Cursor / Copilot Rules
- No `.cursor/rules` or `.cursorrules` files are present in this repo as of 2026-02-04.
- No `.github/copilot-instructions.md` file exists either; follow this AGENTS guide plus `CLAUDE.md` instead.

## Final Checklist Before You Ship
- Run Ruff lint + format, invoke the appropriate pytest subset, and (if applicable) `python visualize_graph.py`.
- Confirm LiteLLM proxy + Streamlit combo still works when your change touches workflow or UI code.
- Ensure observability callbacks still fire (pass `RunnableConfig`, call `flush()` after tests that invoke the workflow).
- Keep AGENTS.md updated if you introduce new commands, skills, prompt directories, or style conventions.
- Prefer smaller commits with clear messages; CI expects green lint and unit tests before merging.
