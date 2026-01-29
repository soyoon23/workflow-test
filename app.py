"""Streamlit UI for Plan-Act Workflow with multi-turn conversation support."""

import logging
from pathlib import Path

import streamlit as st
import yaml

from src.conversation.history import ConversationHistoryManager
from src.llm.client import LLMClient
from src.prompts.registry import PromptRegistry
from src.skills.registry import SkillRegistry
from src.tools.registry import ToolRegistry
from src.workflow.graph import stream_workflow

logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "llm": {
            "base_url": "http://localhost:4000",
            "model": "openai/qwen:7b",
            "api_key": "sk-1234",
            "temperature": 0.7,
            "max_tokens": 4096,
        },
        "prompts": {
            "templates_dir": "src/prompts/templates",
        },
    }


def initialize_components(config: dict):
    """Initialize all components."""
    llm_config = config.get("llm", {})
    llm_client = LLMClient(
        base_url=llm_config.get("base_url", "http://localhost:4000"),
        model=llm_config.get("model", "openai/qwen:7b"),
        api_key=llm_config.get("api_key", "sk-1234"),
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 4096),
    )

    prompts_config = config.get("prompts", {})
    prompt_registry = PromptRegistry(
        templates_dir=prompts_config.get("templates_dir", "src/prompts/templates")
    )

    tool_registry = ToolRegistry()

    skill_registry = SkillRegistry(templates_dir="src/skills/templates")

    return llm_client, prompt_registry, tool_registry, skill_registry


def _init_session_state(config: dict) -> None:
    """Initialize all session state variables once."""
    if "llm_client" not in st.session_state:
        llm_client, prompt_registry, tool_registry, skill_registry = initialize_components(config)
        st.session_state.llm_client = llm_client
        st.session_state.prompt_registry = prompt_registry
        st.session_state.tool_registry = tool_registry
        st.session_state.skill_registry = skill_registry

    if "history_manager" not in st.session_state:
        conv_config = config.get("conversation", {})
        st.session_state.history_manager = ConversationHistoryManager(
            max_turns=conv_config.get("max_history_turns", 20),
            detail_window=conv_config.get("include_plan_detail_turns", 1),
        )
        st.session_state.chat_messages = []


def _render_sidebar(config: dict) -> tuple:
    """Render sidebar settings.

    Returns: (auto_select_skill, selected_skill_idx, available_skills)
    """
    with st.sidebar:
        st.header("Settings")

        # LLM Settings
        st.subheader("LLM Configuration")
        base_url = st.text_input(
            "LiteLLM Base URL",
            value=config["llm"]["base_url"],
            key="base_url",
        )
        model = st.text_input(
            "Model",
            value=config["llm"]["model"],
            key="model",
        )
        api_key = st.text_input(
            "API Key",
            value=config["llm"]["api_key"],
            key="api_key",
            type="password",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=config["llm"]["temperature"],
            step=0.1,
            key="temperature",
        )

        if st.button("Apply LLM Settings"):
            st.session_state.llm_client = LLMClient(
                base_url=base_url,
                model=model,
                api_key=api_key,
                temperature=temperature,
            )
            st.success("LLM settings applied!")

        st.divider()

        # Prompt Settings
        st.subheader("Prompt Versions")
        prompt_registry = st.session_state.prompt_registry

        for role in prompt_registry.get_available_roles():
            versions = prompt_registry.get_available_versions(role)
            current = prompt_registry.get_active_version(role)

            selected = st.selectbox(
                f"{role.capitalize()} Prompt",
                options=versions,
                index=versions.index(current) if current in versions else 0,
                key=f"prompt_{role}",
            )

            if selected != current:
                prompt_registry.set_active_version(role, selected)

        if st.button("Reload Prompts"):
            prompt_registry.reload()
            st.success("Prompts reloaded!")

        st.divider()

        # Skills selection
        st.subheader("Skills")
        skill_registry = st.session_state.skill_registry
        available_skills = skill_registry.get_available()

        auto_select_skill = st.checkbox(
            "Auto-select skill (LLM decides)",
            value=False,
            key="auto_select_skill",
            help="LLM will analyze your request and choose the best skill automatically",
        )

        skill_options = ["None (Default)"] + [f"{s.trigger} - {s.name}" for s in available_skills]
        selected_skill_idx = st.selectbox(
            "Manual Skill Selection",
            options=range(len(skill_options)),
            format_func=lambda x: skill_options[x],
            key="selected_skill",
            disabled=auto_select_skill,
        )

        if auto_select_skill:
            st.info("LLM will automatically select the best skill for your request")
        elif selected_skill_idx > 0:
            selected_skill = available_skills[selected_skill_idx - 1]
            st.info(f"**{selected_skill.name}**\n\n{selected_skill.description}")
            st.caption(f"Tools: {', '.join(selected_skill.tools)}")
        else:
            st.caption("No skill selected - using default workflow")

        st.divider()

        # Conversation management
        st.subheader("Conversation")
        history_mgr = st.session_state.history_manager
        st.caption(f"Turns: {history_mgr.turn_count}")
        if st.button("Clear Conversation"):
            history_mgr.clear()
            st.session_state.chat_messages = []
            st.rerun()

        st.divider()

        # Tools info
        st.subheader("Available Tools")
        tool_registry = st.session_state.tool_registry
        for tool_name in tool_registry.get_available_tools():
            st.write(f"- {tool_name}")

    return auto_select_skill, selected_skill_idx, available_skills


def _resolve_skill(
    user_request: str,
    auto_select_skill: bool,
    selected_skill_idx: int,
    available_skills: list,
) -> tuple:
    """Resolve active skill and actual request from user input.

    Returns: (active_skill, actual_request, use_auto_select)
    """
    skill_registry = st.session_state.skill_registry
    detected_skill, parsed_request = skill_registry.parse_input(user_request)

    use_auto_select = auto_select_skill and not detected_skill

    if detected_skill:
        return detected_skill, parsed_request, False
    elif use_auto_select:
        return None, user_request, True
    elif selected_skill_idx > 0:
        return available_skills[selected_skill_idx - 1], user_request, False
    else:
        return None, user_request, False


def _format_plan_details(plan: list[dict]) -> str:
    """Format plan steps into a readable string for the expander."""
    lines = []
    for step in plan:
        status_icon = {
            "completed": "[OK]",
            "failed": "[FAIL]",
            "pending": "[...]",
            "in_progress": "[>>]",
        }.get(step["status"], "[?]")

        lines.append(f"{status_icon} Step {step['step_number']}: {step['description']}")
        if step.get("result"):
            result_text = step["result"]
            if len(result_text) > 300:
                result_text = result_text[:300] + "..."
            lines.append(f"    Result: {result_text}")
    return "\n".join(lines)


def _run_workflow_streaming(
    actual_request: str,
    active_skill,
    use_auto_select: bool,
) -> dict | None:
    """Run the workflow with streaming and return the final state."""
    history_mgr = st.session_state.history_manager

    status_placeholder = st.empty()
    final_state = None

    try:
        for event in stream_workflow(
            actual_request,
            st.session_state.llm_client,
            st.session_state.prompt_registry,
            st.session_state.tool_registry,
            st.session_state.skill_registry,
            skill=active_skill,
            auto_select_skill=use_auto_select,
            conversation_history=history_mgr.history,
        ):
            for node_name, state in event.items():
                final_state = state

                if node_name == "plan":
                    step_count = len(state.get("plan", []))
                    skill_info = ""
                    if state.get("active_skill_name") and use_auto_select:
                        skill_info = f" | Skill: {state['active_skill_name']}"
                    status_placeholder.markdown(f"*Planning... ({step_count} steps){skill_info}*")

                elif node_name == "act":
                    idx = state.get("current_step_index", 1)
                    total = len(state.get("plan", []))
                    status_placeholder.markdown(f"*Executing step {idx}/{total}...*")

                elif node_name == "review":
                    status_placeholder.empty()

    except Exception as e:
        logger.exception("Workflow execution failed")
        st.error(f"Workflow execution failed: {e}")
        return None

    return final_state


def _display_assistant_response(final_state: dict | None) -> None:
    """Display the assistant response and save to history."""
    if not final_state:
        return

    history_mgr = st.session_state.history_manager

    # Build answer text
    if final_state.get("final_answer"):
        answer_text = final_state["final_answer"]
    elif final_state.get("error"):
        answer_text = f"Error: {final_state['error']}"
    else:
        answer_text = "Workflow completed without a final answer."

    # Build plan details
    plan_detail_text = ""
    if final_state.get("plan"):
        plan_detail_text = _format_plan_details(final_state["plan"])

    # Render
    st.markdown(answer_text)
    if plan_detail_text:
        with st.expander("Plan Details"):
            st.code(plan_detail_text, language=None)

    # Save to chat messages for redisplay
    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "plan_details": plan_detail_text,
        }
    )

    # Extract turn and add to conversation history
    turn = history_mgr.add_turn(final_state)
    logger.info(
        "Saved turn %d: key_facts=%s",
        turn["turn_number"],
        turn["key_facts"],
    )


def main():
    st.set_page_config(
        page_title="Plan-Act Workflow",
        page_icon="🤖",
        layout="wide",
    )

    st.title("Plan-Act Workflow")
    st.markdown("LangGraph 기반 Plan and Act 워크플로우")

    config = load_config()
    _init_session_state(config)

    auto_select_skill, selected_skill_idx, available_skills = _render_sidebar(config)

    # Display conversation history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("plan_details"):
                with st.expander("Plan Details"):
                    st.code(msg["plan_details"], language=None)

    # Chat input
    skill_registry = st.session_state.skill_registry
    triggers = [s.trigger for s in skill_registry.get_available()]
    placeholder = "메시지를 입력하세요..."
    if triggers:
        placeholder += f" (triggers: {', '.join(triggers)})"

    if user_request := st.chat_input(placeholder=placeholder):
        # Display user message
        st.session_state.chat_messages.append({"role": "user", "content": user_request})
        with st.chat_message("user"):
            st.markdown(user_request)

        # Resolve skill
        active_skill, actual_request, use_auto_select = _resolve_skill(
            user_request, auto_select_skill, selected_skill_idx, available_skills
        )

        if active_skill:
            st.info(f"Skill: **{active_skill.name}** ({active_skill.trigger})")

        # Run workflow and display response
        with st.chat_message("assistant"):
            final_state = _run_workflow_streaming(actual_request, active_skill, use_auto_select)
            _display_assistant_response(final_state)

    # Footer
    st.divider()
    st.caption(
        "Tip: Use skill triggers like `/research`, `/calc`, `/analyze` or select from sidebar"
    )


if __name__ == "__main__":
    main()
