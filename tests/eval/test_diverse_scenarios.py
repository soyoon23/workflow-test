"""E2E evaluation tests for diverse scenario domains.

Tests the workflow across 9 different domains covering:
- Travel planning
- Data analysis
- Market research
- Document analysis
- Scheduling
- Content creation
- Policy comparison
- Educational content
- Financial analysis

Each domain includes multiple test scenarios with mocked tools.
"""

import json
from pathlib import Path

import pytest
from deepeval.test_case import LLMTestCase

from ..conftest import load_goldens, run_workflow_with_mocks
from .litellm_eval import get_eval_model
from .metrics.step_execution import StepRelevancyGEval
from .observability_helpers import assert_metrics_with_tracing


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "tests" / "datasets"


@pytest.mark.eval
class TestDiverseScenarios:
    """Comprehensive E2E evaluation across diverse domains."""

    @pytest.fixture(autouse=True)
    def _setup(self, eval_config, obs_test_trace, mock_tool_registry):
        """Setup fixtures for diverse scenario tests."""
        self.config = eval_config
        self.mock_registry = mock_tool_registry
        obs_callback, tracing_ctx = obs_test_trace
        self.tracing_ctx = tracing_ctx
        self.obs_callback = obs_callback
        self.eval_model = get_eval_model()

    @pytest.mark.parametrize("domain", [
        "travel_planning",
        "data_analysis",
        "market_research",
        "document_analysis",
        "scheduling",
        "content_creation",
        "policy_comparison",
        "educational_content",
        "financial_analysis",
    ])
    def test_domain_scenarios(self, domain):
        """Test each domain with its golden scenarios.

        For each domain:
        1. Load golden dataset
        2. Register mock tools based on metadata
        3. Execute workflow with mocked tools
        4. Evaluate output quality

        Args:
            domain: Domain name (e.g., "travel_planning")
        """
        # Load golden dataset for the domain
        try:
            goldens = load_goldens(f"{domain}_goldens")
        except FileNotFoundError:
            pytest.skip(f"No golden dataset for domain: {domain}")

        if not goldens:
            pytest.skip(f"Empty golden dataset for domain: {domain}")

        # Test each scenario in the domain
        for idx, golden in enumerate(goldens):
            with self.subTest(domain=domain, scenario_idx=idx):
                self._test_scenario(golden, domain, idx)

    def _test_scenario(self, golden: dict, domain: str, idx: int) -> None:
        """Test a single scenario.

        Args:
            golden: Golden test case dict
            domain: Domain name for logging
            idx: Scenario index for logging
        """
        user_request = golden.get("input", "")
        expected_output = golden.get("expected_output", "")
        metadata = golden.get("metadata", {})

        if not user_request:
            pytest.skip("No user request in golden data")

        # Register mock tools from metadata
        self._register_mock_tools(metadata)

        # Execute workflow with mocked tools
        try:
            state = run_workflow_with_mocks(
                user_request=user_request,
                config=self.config,
                mock_tool_registry=self.mock_registry,
                obs_callback=self.obs_callback,
            )
        except Exception as e:
            pytest.fail(f"Workflow execution failed: {str(e)}")

        actual_output = state.get("final_answer", "")

        # Create test case for evaluation
        test_case = LLMTestCase(
            input=user_request,
            actual_output=actual_output,
            expected_output=expected_output,
            additional_metadata={
                "domain": domain,
                "scenario_idx": idx,
                "complexity": metadata.get("complexity", "medium"),
            }
        )

        # Evaluate using E2E metrics
        metrics = [StepRelevancyGEval]

        # Run tracing and assertions
        assert_metrics_with_tracing(
            test_case,
            metrics,
            self.tracing_ctx,
            span_name=f"diverse_scenario_{domain}_{idx}",
            metadata={
                "domain": domain,
                "scenario_idx": idx,
                "complexity": metadata.get("complexity"),
            }
        )

    def _register_mock_tools(self, metadata: dict) -> None:
        """Register mock tools based on scenario metadata.

        Args:
            metadata: Scenario metadata containing mock_tools
        """
        mock_tools = metadata.get("mock_tools", {})

        for tool_name, tool_config in mock_tools.items():
            if isinstance(tool_config, dict):
                # Single mock response
                if "response" in tool_config:
                    self.mock_registry.register_mock(
                        tool_name,
                        tool_config["response"]
                    )
                # Multiple calls with different responses
                elif "calls" in tool_config:
                    responses = [call.get("response") for call in tool_config["calls"]]
                    self.mock_registry.register_mock(tool_name, responses)

    @pytest.mark.parametrize("domain", [
        "travel_planning",
        "market_research",
    ])
    def test_mock_tool_usage(self, domain):
        """Verify that mock tools are actually being used.

        Tests that mock tools are properly registered and called during
        workflow execution.

        Args:
            domain: Domain name to test
        """
        try:
            goldens = load_goldens(f"{domain}_goldens")
        except FileNotFoundError:
            pytest.skip(f"No golden dataset for domain: {domain}")

        if not goldens:
            pytest.skip(f"Empty golden dataset for domain: {domain}")

        golden = goldens[0]
        user_request = golden.get("input", "")
        metadata = golden.get("metadata", {})

        # Reset mock call counts
        self.mock_registry.reset_call_counts()

        # Register mock tools
        self._register_mock_tools(metadata)

        # Execute workflow
        state = run_workflow_with_mocks(
            user_request=user_request,
            config=self.config,
            mock_tool_registry=self.mock_registry,
            obs_callback=self.obs_callback,
        )

        # Verify that mocked tools were called
        mock_tools = metadata.get("mock_tools", {})
        for tool_name in mock_tools.keys():
            call_count = self.mock_registry.get_call_count(tool_name)
            assert call_count > 0, (
                f"Mock tool '{tool_name}' was registered but never called"
            )

    def test_mock_tool_response_format(self):
        """Verify that mock tool responses have correct format.

        Mock responses should include metadata to distinguish them from
        real tool executions.
        """
        # Register a simple mock
        mock_response = {"success": True, "data": "test"}
        self.mock_registry.register_mock("calculator", mock_response)

        # Execute the tool
        result = self.mock_registry.execute("calculator", expression="1+1")

        # Verify format
        assert result is not None
        assert result.get("__mock__") is True
        assert result.get("success") is True
        assert result.get("data") == "test"

    def test_clearing_mocks_reverts_to_real_tools(self):
        """Verify that clearing mocks reverts to real tool execution.

        After unregistering mocks, tools should fall back to real execution.
        """
        # Register a mock
        self.mock_registry.register_mock("calculator", {"result": 999})

        # Execute with mock
        result_with_mock = self.mock_registry.execute("calculator", expression="1+1")
        assert result_with_mock.get("__mock__") is True

        # Unregister mock
        self.mock_registry.unregister_mock("calculator")

        # Execute without mock (should use real calculator)
        result_without_mock = self.mock_registry.execute("calculator", expression="1+1")
        assert result_without_mock.get("__mock__") is not True
        # Real calculator should return actual result
        assert result_without_mock.get("success") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
