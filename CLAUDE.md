# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose & High-Level Expectations

- Follow the Plan-Act-Review workflow conventions; preserve planner/actor/reviewer separation and do not bypass nodes.
- Assume multi-turn conversations.
- Always pass LangChain `RunnableConfig` objects down every LLM call so Langfuse callbacks stay wired.
- Normalize observability metadata: `skill` should be `"none"` when no skill is active and `auto_select` stored as a lowercase string.
- Flush Langfuse traces via `src/observability.flush()` whenever a workflow run completes, especially in CLI scripts or tests.
- UI copy is Korean; keep new user-facing strings Korean unless a feature explicitly targets another locale.

## Commands

```bash
# Install dependencies (append --extra dev in CI parity, --extra eval for evaluation suites)
uv sync

# Run the app (requires LiteLLM proxy running first)
litellm --model gpt-4 --port 4000    # Terminal 1: start LLM proxy
uv run streamlit run app.py           # Terminal 2: start Streamlit UI

# Lint and format
uv run ruff check .
uv run ruff format .                  # CI runs: uv run ruff format --check .

# Tests
uv run pytest                         # all tests
uv run pytest tests/unit -v           # unit only
uv run pytest tests/eval -v --tb=short  # eval only (requires DEEPEVAL_API_KEY)
uv run pytest tests/unit/test_routing.py::test_should_continue_plan -vv  # single test
uv run pytest -k "plan_node and not eval"  # pattern filtering

# Generate workflow graph visualization
python visualize_graph.py
```

## Repository & Environment Setup

- Python 3.11+; dependency graph managed with `uv` (see `pyproject.toml` and `uv.lock`).
- Install or refresh dependencies after pulling: `uv sync --extra dev`.
- Use `uv add <package>` / `uv remove <package>` instead of pip to keep `uv.lock` canonical.
- Config lives in `config.yaml`; copy from `config.yaml.example` and keep secrets in `.env` (auto-loaded via `python-dotenv` inside `app.py`).
- LiteLLM proxy must be running before the Streamlit app.
- Streamlit session state stores `WorkflowComponents` and conversation history.
- Generate the workflow graph image with `python visualize_graph.py` when docs or diagrams need refreshing.
- Avoid committing `config.yaml`, `.env`, or secrets; they are gitignored.

## Recent Changes (2026-02-04)

- LLM stack now uses `langchain-openai`'s `ChatOpenAI` (via LiteLLM proxy). Always pass the `RunnableConfig` from LangGraph nodes down to `LLMClient` so Langfuse callbacks capture every call.
- Added official Langfuse v3 integration. Use `create_callback()` to get both the callback handler and tracing context, and call `src/observability.flush()` once a workflow run completes to ensure traces are persisted.
- `python-dotenv` is loaded at `app.py` startup so secrets can live in `.env`. `config.yaml` is gitignored—copy from `config.yaml.example` when bootstrapping local envs.
- Workflow metadata sent to observability is now normalized (`skill="none"`, `auto_select` as lowercase string) to keep Langfuse dashboards clean.
- Dependencies updated (`langchain`, `langchain-openai`, `python-dotenv`), so run `uv sync` after pulling.

## Architecture

This is a **Plan-Act-Review** AI workflow built on LangGraph and LiteLLM with a Streamlit chat UI supporting **multi-turn conversations**.

### Workflow Node Architecture

Nodes are independent classes satisfying the `BaseNode` protocol (`src/workflow/nodes/base.py`). Each node receives a `WorkflowComponents` dataclass that bundles all shared dependencies (LLM client, registries, callback).

```
src/workflow/
├── components.py        # WorkflowComponents dataclass
├── parsing.py           # JSON/plan/review parsing utilities
├── tool_executor.py     # Reusable tool execution loop
├── graph.py             # LangGraph workflow definition & execution
├── state.py             # AgentState TypedDict
├── streaming.py         # StreamCallback for UI events
└── nodes/
    ├── base.py          # BaseNode Protocol + NodeMixin (shared helpers)
    ├── plan.py          # PlanNode — creates execution plans
    ├── act.py           # ActNode — executes plan steps with tools
    └── review.py        # ReviewNode — evaluates results
```

**Key types:**
- **`BaseNode`** (Protocol) — Any callable `(AgentState, RunnableConfig) -> dict[str, Any]` satisfies this. LangGraph auto-passes `RunnableConfig` (with callback handlers) to each node.
- **`NodeMixin`** — Provides shared helpers (`_get_prompt`, `_get_skill`, `_get_tools`) used by all built-in nodes.
- **`WorkflowComponents`** — Bundles `LLMClient`, `PromptRegistry`, `ToolRegistry`, `SkillRegistry`, `StreamCallback`, and `ConversationContextBuilder`.

### Workflow Loop

The core loop is a LangGraph state machine defined in `src/workflow/graph.py`:

1. **Plan** (`PlanNode`) — LLM analyzes the user request and produces a structured execution plan (list of `PlanStep`). `NodeMixin._build_skill_selection_block()` only shows up when auto-select is active and no skill is chosen.
2. **Act** (`ActNode`) — Executes one step at a time, calling tools via `execute_tool_calls()` from `tool_executor.py`. Marks steps `in_progress`, builds a HumanMessage with the original `user_request`, prior completed step results (`_build_prior_context`), and the current step instruction. Streams tokens via `StreamCallback.token` and supports multi-round tool call loops (up to `MAX_TOOL_ROUNDS=3`); on the final round tools are omitted to force a text response. Failures set `status="failed"` and propagate an `error` field so `should_continue` terminates the run.
3. **Review** (`ReviewNode`) — LLM evaluates results via `_format_plan_for_review()` summaries and decides: complete → END, or needs replan → back to Plan. Returns `is_complete`, `final_answer`, `error`, and `key_facts`. Those `key_facts` feed the next turn's actor/reviewer context.

Routing between nodes is handled by `should_continue()` and `after_review()` conditional edge functions. The workflow state (`AgentState` TypedDict in `src/workflow/state.py`) carries messages, plan steps, iteration count, active skill context, and conversation history through the graph.

### Multi-Turn Conversation

`src/conversation/` manages cross-turn context with a hybrid sliding window strategy:

- **`ConversationHistoryManager`** (`src/conversation/history.py`) — Tracks `ConversationTurn` summaries across workflow invocations. Applies a sliding window: recent turns retain full plan results, older turns keep only summaries. Configurable via `config.yaml` (`max_history_turns`, `include_plan_detail_turns`).
- **`ConversationContextBuilder`** (`src/conversation/context.py`) — Builds role-specific context strings from history. Uses `RoleContextFormatter` subclasses for each role:
  - **Planner**: Sees request, answer, recent plan structure, skill used (most detail)
  - **Actor**: Sees only `key_facts` (minimal, avoids noise)
  - **Reviewer**: Sees request, answer, key_facts (medium detail)
- Custom formatters can be registered via `ConversationContextBuilder.register_formatter()`.

Context is injected into each node's system prompt via `NodeMixin._get_prompt()`. The `ConversationHistoryManager` lives in Streamlit session state and persists across reruns.

### Registry Pattern

Three registries manage extensible components:

- **PromptRegistry** (`src/prompts/registry.py`) — Loads versioned YAML prompt templates from `src/prompts/templates/`. Each role (planner, actor, reviewer) can have multiple versions (v1, v2). Active version is switchable at runtime via `set_active_version` or the Streamlit UI.
- **ToolRegistry** (`src/tools/registry.py`) — Registers tool definitions (OpenAI function format) and their Python implementations. Tools are defined in `src/tools/base.py`.
- **SkillRegistry** (`src/skills/registry.py`) — Loads skill YAML files from `src/skills/templates/`. Skills are triggered via user input prefixes (e.g., `/research`, `/calc`) or auto-selected by the LLM. `parse_input` strips the trigger from the user message; keep triggers unique and start them with `/`. Skills inject additional prompt context and can restrict available tools.

Reload registries with `.reload()` after modifying YAML files during runtime (buttons already exist in the UI).

### LLM Client

`src/llm/client.py` wraps `langchain-openai`'s `ChatOpenAI` pointed at a LiteLLM proxy. Automatically normalizes Base URLs to `/v1`. All LLM calls are proper LangChain Runnable invocations, so `RunnableConfig` (containing Langfuse callback handlers) automatically captures full input/output. Configured via `config.yaml` (default `localhost:4000`).

### Observability

`src/observability/` provides Langfuse v3 integration via the official `langfuse.langchain.CallbackHandler`. Custom attributes (trace name, metadata, session/user IDs) are propagated via `langfuse.propagate_attributes`. The module-level `flush()` function handles trace flushing and propagation context cleanup.

- Call `create_callback()` before LangGraph runs and `flush()` afterward.
- When sending observability metadata, trim user input to ~100 chars (see `app.py`) and attach `skill`/`auto_select` normalized values.
- `StreamCallback` should only issue events when a handler is registered; guard for `is_active` before streaming tokens.
- When adding nodes or custom flows, always emit `StreamCallback` events so the Streamlit UI stays reactive (plan_ready, step_start, tool_call, review_ready, error).

### Adding New Components

- **Workflow node**: Implement `__call__(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]` (satisfies `BaseNode` protocol). Pass `config` to LLM calls for callback propagation. Use `NodeMixin` for shared helpers. Initialize with `WorkflowComponents`.
- **Prompt version**: Add a YAML file to `src/prompts/templates/` named `{role}_v{N}.yaml`.
- **Tool**: Add implementation to `src/tools/base.py` and register in `src/tools/registry.py`. Include JSON schemas describing arguments (OpenAI function format).
- **Skill**: Add a YAML file to `src/skills/templates/` with `name`, `description`, `trigger`, optional `prompt`, and `tools` arrays. Ensure tool names exist in `ToolRegistry`.
- **Context formatter**: Subclass `RoleContextFormatter` in `src/conversation/context.py` and register via `ConversationContextBuilder.register_formatter()`.
- Tool execution goes through `execute_tool_calls()` which handles the special `use_skill` tool; include that definition whenever you expose tools to the LLM.
- Any workflow extension should stick to the `WorkflowComponents` bundle to avoid leaking individual dependencies.
- Remember to update `workflow_graph.png` and `agent_structure.md` when you modify node wiring significantly.

## Configuration

Copy `config.yaml.example` to `config.yaml` and edit. It holds LLM connection settings (base_url, model, api_key), default prompt version, workflow limits (max_iterations), conversation history settings (max_history_turns, include_plan_detail_turns), and observability settings. `config.yaml` is gitignored to protect secrets.

Environment variables can also be loaded from a `.env` file (via `python-dotenv`). Langfuse credentials (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`) can be set via env vars or `config.yaml`. The Streamlit sidebar also allows runtime overrides.

- Keep Langfuse keys out of history and logs; never hardcode keys.
- `.env` is loaded at `app.py` startup via `load_dotenv()` so agents can assume environment variables are ready.
- Workflow metadata going to observability must use normalized strings: `skill="none"` and `auto_select="true"/"false"`.

## Testing

- Default all tests through uv: `uv run pytest`.
- Unit tests live under `tests/unit`; they do not require external LLMs.
- Evaluation tests under `tests/eval` invoke LiteLLM + DeepEval; export `DEEPEVAL_API_KEY` (usually via `.env`).
- Markers defined in `pyproject.toml`: `@pytest.mark.unit` and `@pytest.mark.eval`; combine with `-m "unit"` to narrow scope.
- CI workflow (`.github/workflows/test.yml`) installs uv, syncs dev deps, runs Ruff lint+format check, then executes `uv run pytest tests/unit/ -v`; evaluation suite only runs on main or manual dispatch.
- Golden datasets for plan/act/review/e2e validations live in `tests/datasets/*.json`; keep them deterministic. Multi-step goldens use `metadata.is_multistep: true` with a `steps` array containing per-step description, status, result, and expected_tools.
- Structure fixtures inside `tests/conftest.py` (root) or scoped `tests/eval/conftest.py`; keep them deterministic.
- Mark integration tests that call LLMs with `@pytest.mark.eval` to avoid accidental CI cost.
- Use DeepEval utilities inside `tests/eval` only when `DEEPEVAL_API_KEY` is configured; otherwise skip gracefully.
- Prefer asserting on structured data (plans, tool outputs) rather than raw strings to keep tests stable.
- Keep golden JSON fixtures human-readable and sorted.

## Code Style

### General

- Python 3.11+; Ruff with line-length 100, rules: E, F, I, W.
- UI text is in Korean; code and variable names are in English.
- Prefer dataclasses (`@dataclass`) for structured data (see `Skill`, `WorkflowComponents`, `StreamEvent`).
- Every module begins with a concise triple-double-quoted docstring describing its intent.
- Stick with f-strings for string interpolation; avoid `%` formatting.
- Use `Path` from `pathlib` for filesystem paths; open files with explicit UTF-8 encoding.
- Avoid top-level side effects; guard runtime code under `if __name__ == "__main__":`.
- Keep UI messaging (Streamlit) inside `app.py`; new workflow logic belongs in `src/workflow`.
- No mypy yet, but add type hints consistent with existing usage.

### Typing & Interfaces

- Type everything: function signatures, return types, and class attributes; prefer `list[str]`/`dict[str, Any]` literal generics (PEP 585).
- Use `Optional[...]` when `None` appears in the data flow and annotate dictionaries with `TypedDict` or dataclasses when persistent.
- Protocols (`BaseNode`) and mixins should stay lightweight; use `runtime_checkable` when other modules need `isinstance` checks.
- When you need callables for LangChain, keep method signatures consistent and ensure `config: RunnableConfig` is optional but passed through.
- Use `Literal` for finite return values (`Literal["act", "review", ...]` in `should_continue`).
- Dataclasses should default to `field(default_factory=list)` or similar to avoid mutable default arguments.

### Imports & Module Layout

- Module-level `logger = logging.getLogger(__name__)` appears near the top; never instantiate loggers inside functions.
- Keep import groups ordered: stdlib → third-party → first-party; alphabetical inside each block.
- Avoid wildcard imports; only import what you use.
- Re-export symbols via `__all__` only when needed (see `src/observability/__init__.py`).

### Naming & Structure

- Classes use `CamelCase`, functions/variables use `snake_case`, constants are `UPPER_SNAKE_CASE`.
- Workflow state keys (`plan`, `current_step_index`, `review_key_facts`) should stay consistent; reuse the same keys when extending state.
- Keep prompt placeholders descriptive (`{skill_selection_block}`) and document them in `NodeMixin` helpers.
- Tests follow `test_<feature>.py` naming; prefer descriptive test function names over numeric suffixes.

### Error Handling & Logging

- Prefer raising `ValueError`/`RuntimeError` for programmer errors (see `PromptRegistry.get`); return structured dicts for user-facing tool errors.
- Wrap risky sections in `try/except` and log via `logger.exception` before surfacing a user-visible error.
- Avoid bare `except`; catch specific exceptions when possible or re-raise after logging.
- Never `print()` for diagnostics; rely on logging and Streamlit status containers.
- When executing tools, return JSON-serializable payloads and include `success`/`error` fields so the UI can render them.
- Propagate errors through the workflow state (`state["error"]`) so `should_continue` can terminate gracefully.

### Streamlit & UI Conventions

- Use `st.status`, `st.info`, `st.caption`, etc., mirroring current UI patterns; respect Korean copy guidelines.
- Store mutable UI state in dictionaries, as `_run_workflow_streaming` does for containers/token buffers.
- Truncate large tool results before rendering (see `app.py`: limit to 500 chars).
- Expose skill/tool metadata in the sidebar; surface new registries via Streamlit controls consistent with existing sections.
- When adding settings, keep them inside the sidebar and persist them in `st.session_state`.

## Git & Workflow Hygiene

- Never force-reset or discard user changes; keep the working tree clean and commit only what the task requires.
- Match the formatting of existing files instead of restyling entire modules.
- Run `uv run ruff format .` before committing to avoid CI failures.
- If you add dependencies, update both `pyproject.toml` and `uv.lock` using `uv sync`.
- Document notable workflow changes in `future_plan.md` or README sections when appropriate.
- Prefer smaller commits with clear messages; CI expects green lint and unit tests before merging.

## Final Checklist Before You Ship

- Run Ruff lint + format, invoke the appropriate pytest subset, and (if applicable) `python visualize_graph.py`.
- Confirm LiteLLM proxy + Streamlit combo still works when your change touches workflow or UI code.
- Ensure observability callbacks still fire (pass `RunnableConfig`, call `flush()` after tests that invoke the workflow).
- Keep this file updated if you introduce new commands, skills, prompt directories, or style conventions.
