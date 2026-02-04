"""Streamlit UI for Plan-Act Workflow with multi-turn conversation support."""

import json
import logging
from pathlib import Path

import streamlit as st
import yaml

from src.conversation.history import ConversationHistoryManager
from src.llm.client import LLMClient
from src.observability import create_callback
from src.prompts.registry import PromptRegistry
from src.skills.registry import SkillRegistry
from src.tools.registry import ToolRegistry
from src.workflow.graph import stream_workflow
from src.workflow.streaming import StreamCallback, StreamEventType

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

    # Store config for observability callback creation per workflow execution
    if "config" not in st.session_state:
        st.session_state.config = config


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


def _run_workflow_streaming(
    actual_request: str,
    active_skill,
    use_auto_select: bool,
) -> dict | None:
    """Run the workflow with streaming and return the final state."""
    history_mgr = st.session_state.history_manager

    # Create observability callback for this workflow execution
    from datetime import datetime

    obs_callback = None
    if st.session_state.get("config"):
        obs_callback, _obs_ctx = create_callback(
            st.session_state.config,
            trace_name=f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            metadata={
                "source": "streamlit",
                "user_request": actual_request[:100],
                "skill": active_skill.name if active_skill else None,
                "auto_select": use_auto_select,
            },
        )

    callback = StreamCallback()

    # Mutable UI state shared with the event handler
    ui = {
        "plan_container": None,
        "act_containers": {},
        "review_container": None,
        "token_placeholder": None,
        "token_buf": "",
    }

    def _handle_event(event):
        et = event.event_type

        if et == StreamEventType.PHASE_START and event.node_name == "plan":
            ui["plan_container"] = st.status("계획 수립 중...", expanded=True, state="running")

        elif et == StreamEventType.PLAN_READY:
            container = ui.get("plan_container")
            if container:
                with container:
                    plan = event.data.get("plan", [])
                    skill = event.data.get("selected_skill")
                    if skill:
                        st.caption(f"Skill: {skill}")
                    for step in plan:
                        st.markdown(f"**{step['step_number']}.** {step['description']}")
                container.update(
                    label=f"계획 완료 ({len(plan)}단계)",
                    state="complete",
                    expanded=False,
                )

        elif et == StreamEventType.STEP_START:
            step_num = event.data["step_number"]
            desc = event.data["description"]
            container = st.status(
                f"Step {step_num} 실행 중: {desc}",
                expanded=True,
                state="running",
            )
            ui["act_containers"][step_num] = container
            with container:
                ui["token_placeholder"] = st.empty()
                ui["token_buf"] = ""

        elif et == StreamEventType.TOKEN:
            placeholder = ui.get("token_placeholder")
            if placeholder:
                ui["token_buf"] += event.data["text"]
                placeholder.markdown(ui["token_buf"])

        elif et == StreamEventType.TOOL_CALL:
            step_containers = ui.get("act_containers", {})
            if step_containers:
                container = list(step_containers.values())[-1]
                with container:
                    st.info(f"도구 호출: **{event.data['tool_name']}**")

        elif et == StreamEventType.TOOL_RESULT:
            step_containers = ui.get("act_containers", {})
            if step_containers:
                container = list(step_containers.values())[-1]
                with container:
                    result = event.data["result"]
                    result_str = json.dumps(result, ensure_ascii=False, indent=2)
                    if len(result_str) > 500:
                        result_str = result_str[:500] + "..."
                    st.code(result_str, language="json")
                    # Reset token buffer for post-tool response streaming
                    ui["token_placeholder"] = st.empty()
                    ui["token_buf"] = ""

        elif et == StreamEventType.STEP_RESULT:
            step_num = event.data["step_number"]
            container = ui["act_containers"].get(step_num)
            if container:
                container.update(
                    label=f"Step {step_num} 완료",
                    state="complete",
                    expanded=False,
                )
                ui["token_placeholder"] = None
                ui["token_buf"] = ""

        elif et == StreamEventType.PHASE_START and event.node_name == "review":
            ui["review_container"] = st.status("결과 검토 중...", expanded=False, state="running")

        elif et == StreamEventType.REVIEW_READY:
            container = ui.get("review_container")
            if container:
                is_complete = event.data.get("is_complete", False)
                if is_complete:
                    container.update(label="검토 완료", state="complete", expanded=False)
                else:
                    container.update(label="재계획 필요", state="error", expanded=False)

        elif et == StreamEventType.ERROR:
            st.error(f"오류: {event.data.get('message', 'Unknown error')}")

    callback.set_handler(_handle_event)

    # Prepare callbacks
    callbacks = [obs_callback] if obs_callback else None

    final_state: dict = {}
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
            stream_callback=callback,
            callbacks=callbacks,
        ):
            for _, state in event.items():
                final_state.update(state)

        # Flush observability traces after workflow completion
        if obs_callback and hasattr(obs_callback, "flush"):
            obs_callback.flush()

    except Exception as e:
        logger.exception("Workflow execution failed")
        st.error(f"워크플로우 실행 실패: {e}")
        return None

    return final_state or None


def _render_workflow_details(plan: list[dict]) -> None:
    """Render plan/act/review results using st.status containers.

    Used both after live streaming and when replaying chat history,
    so intermediate results persist across Streamlit reruns.
    """
    if not plan:
        return

    # Plan summary
    with st.status(f"계획 완료 ({len(plan)}단계)", state="complete", expanded=False):
        for step in plan:
            st.markdown(f"**{step['step_number']}.** {step['description']}")

    # Each step result
    for step in plan:
        status_label = {
            "completed": f"Step {step['step_number']} 완료",
            "failed": f"Step {step['step_number']} 실패",
            "pending": f"Step {step['step_number']} 대기",
            "in_progress": f"Step {step['step_number']} 진행 중",
        }.get(step["status"], f"Step {step['step_number']}")

        state = "complete" if step["status"] == "completed" else "error"

        with st.status(status_label, state=state, expanded=False):
            if step.get("result"):
                st.markdown(step["result"])


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

    # Render final answer
    st.markdown(answer_text)

    # Save to chat messages for redisplay (include structured plan data)
    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "plan": final_state.get("plan", []),
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
            if msg["role"] == "assistant" and msg.get("plan"):
                _render_workflow_details(msg["plan"])
            st.markdown(msg["content"])

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
