"""Workflow nodes for Plan-Act workflow."""

import json
import logging
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..conversation.context import ConversationContextBuilder
from ..llm.client import LLMClient
from ..prompts.registry import PromptRegistry
from ..skills.registry import Skill, SkillRegistry
from ..tools.registry import ToolRegistry
from .state import AgentState, PlanStep
from .streaming import StreamCallback

logger = logging.getLogger(__name__)


class WorkflowNodes:
    """Nodes for the Plan-Act workflow."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_registry: PromptRegistry,
        tool_registry: ToolRegistry,
        skill_registry: SkillRegistry,
        initial_skill: Optional[Skill] = None,
        stream_callback: Optional[StreamCallback] = None,
    ):
        self.llm = llm_client
        self.prompts = prompt_registry
        self.tools = tool_registry
        self.skill_registry = skill_registry
        self.initial_skill = initial_skill
        self._context_builder = ConversationContextBuilder()
        self._callback = stream_callback or StreamCallback()

    def _get_skill(self, state: AgentState) -> Optional[Skill]:
        """Get active skill from state or initial skill."""
        skill_name = state.get("active_skill_name")
        if skill_name:
            return self.skill_registry.get(skill_name)
        return self.initial_skill

    def _get_prompt(self, role: str, state: AgentState) -> str:
        """Get prompt for a role, considering active skill and conversation context."""
        skill = self._get_skill(state)

        # Always use default prompt from registry
        prompt = self.prompts.get(role)

        # For planner role, handle skill_selection_block placeholder
        if role == "planner":
            skill_selection_block = self._build_skill_selection_block(state)
            prompt = prompt.replace("{skill_selection_block}", skill_selection_block)

        # Append skill context/instructions if available
        if skill and skill.prompt:
            prompt = f"{prompt}\n\n## Skill Context\n{skill.prompt}"

        # Append conversation history context (role-specific)
        history = state.get("conversation_history", [])
        if history:
            conv_context = self._context_builder.build(history, role)
            if conv_context:
                prompt = f"{prompt}\n\n{conv_context}"
                logger.debug(
                    "Injected conversation context into %s prompt (%d chars)",
                    role,
                    len(conv_context),
                )

        return prompt

    def _build_skill_selection_block(self, state: AgentState) -> str:
        """Build skill selection instruction block for planner prompt.

        Returns instruction text when auto_select_skill is True and no skill
        is currently active. Returns empty string otherwise.
        """
        if not state.get("auto_select_skill"):
            return ""

        if state.get("active_skill_name"):
            return ""

        available = self.skill_registry.get_available()
        if not available:
            return ""

        skills_desc = "\n".join([f"- {s.name} ({s.trigger}): {s.description}" for s in available])

        return (
            "## Skill Selection\n"
            "You also need to select the most appropriate skill for this request.\n"
            "Available skills:\n"
            f"{skills_desc}\n\n"
            'Analyze the request and set "selected_skill" in your output to the best skill name, '
            "or null if no specific skill is needed."
        )

    def _get_tools(self, state: AgentState) -> list[dict]:
        """Get tools, filtered by skill if active."""
        skill = self._get_skill(state)

        # Always include use_skill tool for dynamic switching
        use_skill_tool = self._get_use_skill_tool()

        if skill and skill.tools:
            # Only return tools specified by the skill + use_skill
            skill_tools = [
                self.tools.get_definition(name)
                for name in skill.tools
                if self.tools.get_definition(name)
            ]
            return skill_tools + [use_skill_tool]

        return self.tools.get_all_definitions() + [use_skill_tool]

    def _get_use_skill_tool(self) -> dict:
        """Get the use_skill tool definition."""
        available = self.skill_registry.get_available()
        skill_names = [s.name for s in available]

        return {
            "type": "function",
            "function": {
                "name": "use_skill",
                "description": (
                    "Switch to a specialized skill for better task handling. "
                    f"Available skills: {', '.join(skill_names)}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": (
                                f"Name of the skill to activate. Options: {', '.join(skill_names)}"
                            ),
                            "enum": skill_names,
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief reason for switching to this skill",
                        },
                    },
                    "required": ["skill_name"],
                },
            },
        }

    def plan_node(self, state: AgentState) -> dict[str, Any]:
        """Create or update the execution plan."""
        self._callback.phase_start("plan")

        system_prompt = self._get_prompt("planner", state)
        user_request = state["user_request"]

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Create a plan for: {user_request}"),
        ]

        # If we have previous results, include them for replanning
        if state.get("plan") and any(
            step["status"] in ["completed", "failed"] for step in state["plan"]
        ):
            results_summary = self._summarize_results(state["plan"])
            messages.append(
                HumanMessage(
                    content=f"Previous execution results:\n{results_summary}\n\n"
                    "Please update the plan if needed."
                )
            )

        logger.debug("plan_node: sending %d messages to LLM", len(messages))
        response = self.llm.chat(messages)

        # Parse the plan from response
        plan, selected_skill = self._parse_plan(response.content)
        logger.info("plan_node: generated %d steps", len(plan))

        self._callback.plan_ready(
            plan=[{"step_number": s["step_number"], "description": s["description"]} for s in plan],
            skill=selected_skill,
        )
        self._callback.phase_end("plan", step_count=len(plan))

        result = {
            "messages": [response],
            "plan": plan,
            "current_step_index": 0,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

        # If plan selected a skill and auto_select is active, update state
        if selected_skill and state.get("auto_select_skill") and not state.get("active_skill_name"):
            result["active_skill_name"] = selected_skill
            logger.info("plan_node: auto-selected skill=%s", selected_skill)

        return result

    def act_node(self, state: AgentState) -> dict[str, Any]:
        """Execute the current step in the plan."""
        system_prompt = self._get_prompt("actor", state)
        plan = state["plan"]
        current_index = state["current_step_index"]

        if current_index >= len(plan):
            return {"error": "No more steps to execute"}

        current_step = plan[current_index]

        self._callback.phase_start("act")
        self._callback.step_start(current_step["step_number"], current_step["description"])

        # Mark step as in progress
        plan[current_index]["status"] = "in_progress"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Execute this step:\n"
                f"Step {current_step['step_number']}: {current_step['description']}"
            ),
        ]

        # Get available tools (filtered by skill if active)
        tools = self._get_tools(state)

        # Call LLM with tools (use streaming if callback available)
        logger.debug(
            "act_node: executing step %d/%d: %s",
            current_step["step_number"],
            len(plan),
            current_step["description"][:80],
        )
        if self._callback.is_active:
            response = self.llm.stream_with_callback(
                messages,
                on_token=lambda t: self._callback.token("act", t),
                tools=tools,
            )
        else:
            response = self.llm.chat(messages, tools=tools)

        result_messages = [response]
        step_result = response.content

        # Handle tool calls
        new_skill_name = None
        if response.tool_calls:
            for tool_call in response.tool_calls:
                self._callback.tool_call(tool_call["name"], tool_call.get("args", {}))

                # Check if this is use_skill tool
                if tool_call["name"] == "use_skill":
                    new_skill_name = tool_call["args"].get("skill_name")
                    reason = tool_call["args"].get("reason", "")
                    tool_result = {
                        "success": True,
                        "message": f"Switched to skill: {new_skill_name}",
                        "reason": reason,
                    }
                else:
                    tool_result = self.tools.execute_tool_call(tool_call)

                self._callback.tool_result(tool_call["name"], tool_result)

                tool_message = ToolMessage(
                    content=json.dumps(tool_result),
                    tool_call_id=tool_call["id"],
                )
                result_messages.append(tool_message)
                step_result += f"\nTool result: {json.dumps(tool_result)}"

            # Get final response after tool execution
            messages.extend(result_messages)
            if self._callback.is_active:
                final_response = self.llm.stream_with_callback(
                    messages,
                    on_token=lambda t: self._callback.token("act", t),
                )
            else:
                final_response = self.llm.chat(messages)
            result_messages.append(final_response)
            step_result = final_response.content

        # Update step with result
        plan[current_index]["status"] = "completed"
        plan[current_index]["result"] = step_result
        logger.info("act_node: completed step %d", current_step["step_number"])

        self._callback.step_result(current_step["step_number"], step_result)
        self._callback.phase_end("act")

        result = {
            "messages": result_messages,
            "plan": plan,
            "current_step_index": current_index + 1,
        }

        # If skill was switched, update state
        if new_skill_name:
            result["active_skill_name"] = new_skill_name

        return result

    def review_node(self, state: AgentState) -> dict[str, Any]:
        """Review execution results and determine next steps."""
        self._callback.phase_start("review")

        system_prompt = self._get_prompt("reviewer", state)
        user_request = state["user_request"]
        plan = state["plan"]

        # Build review context
        plan_summary = self._format_plan_for_review(plan)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Original request: {user_request}\n\n"
                f"Plan and execution results:\n{plan_summary}\n\n"
                "Please review and determine next steps."
            ),
        ]

        logger.debug("review_node: sending review request to LLM")
        response = self.llm.chat(messages)

        # Parse review result
        review = self._parse_review(response.content)
        logger.info(
            "review_node: is_complete=%s, has_key_facts=%d",
            review.get("is_complete"),
            len(review.get("key_facts", [])),
        )

        self._callback.review_ready(
            is_complete=review.get("is_complete", False),
            final_answer=review.get("final_answer"),
        )
        self._callback.phase_end("review")

        return {
            "messages": [response],
            "is_complete": review.get("is_complete", False),
            "final_answer": review.get("final_answer"),
            "error": review.get("error"),
            "review_key_facts": review.get("key_facts", []),
        }

    def _parse_plan(self, content: str) -> tuple[list[PlanStep], Optional[str]]:
        """Parse plan from LLM response.

        Returns:
            Tuple of (plan_steps, selected_skill_name).
            selected_skill_name is None if no skill was selected.
        """
        selected_skill = None
        plan_data = self._extract_json_payload(content)

        if plan_data:
            skill_name = plan_data.get("selected_skill")
            if skill_name and str(skill_name).lower() not in ("none", "null", ""):
                if self.skill_registry.get(skill_name):
                    selected_skill = skill_name

            steps: list[PlanStep] = []
            for step in plan_data.get("steps", []):
                steps.append(
                    PlanStep(
                        step_number=step.get("step_number", len(steps) + 1),
                        description=step.get("description", ""),
                        status="pending",
                        result=None,
                    )
                )
            if steps:
                return steps, selected_skill

        logger.warning("Failed to parse plan JSON, using fallback")

        # Fallback: create a single step from the content
        return [
            PlanStep(
                step_number=1,
                description=content[:500],
                status="pending",
                result=None,
            )
        ], None

    def _parse_review(self, content: str) -> dict[str, Any]:
        """Parse review from LLM response."""
        review_data = self._extract_json_payload(content)
        if review_data is not None:
            return review_data

        logger.warning("Failed to parse review JSON, using fallback")

        # Fallback: assume complete if we can't parse
        return {
            "is_complete": True,
            "final_answer": content,
        }

    def _extract_json_payload(self, content: str) -> Optional[dict[str, Any]]:
        """Extract a JSON object from LLM output.

        Supports fenced code blocks (```json ... ```), otherwise falls back to
        searching for the first balanced brace section.
        """

        # Prefer fenced code block to avoid grabbing unrelated braces
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})```", content, re.DOTALL)
        raw_payload: Optional[str] = None
        if fence_match:
            raw_payload = fence_match.group(1)
        else:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                raw_payload = content[json_start:json_end]

        if not raw_payload:
            return None

        try:
            return json.loads(raw_payload)
        except json.JSONDecodeError:
            logger.debug("JSON decoding failed for payload: %s", raw_payload[:200])
            return None

    def _summarize_results(self, plan: list[PlanStep]) -> str:
        """Summarize execution results for replanning."""
        lines = []
        for step in plan:
            status_emoji = {
                "completed": "[OK]",
                "failed": "[FAIL]",
                "pending": "[...]",
                "in_progress": "[>>]",
            }.get(step["status"], "[?]")

            lines.append(f"{status_emoji} Step {step['step_number']}: {step['description']}")
            if step.get("result"):
                lines.append(f"   Result: {step['result'][:200]}")

        return "\n".join(lines)

    def _format_plan_for_review(self, plan: list[PlanStep]) -> str:
        """Format plan for review."""
        lines = []
        for step in plan:
            lines.append(f"Step {step['step_number']}: {step['description']}")
            lines.append(f"  Status: {step['status']}")
            if step.get("result"):
                lines.append(f"  Result: {step['result']}")
            lines.append("")

        return "\n".join(lines)
