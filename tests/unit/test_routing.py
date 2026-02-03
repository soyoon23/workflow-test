"""Unit tests for workflow routing logic (should_continue, after_review)."""

from src.workflow.graph import after_review, should_continue


class TestShouldContinue:
    """Tests for should_continue() conditional edge function."""

    def test_returns_end_on_error(self, empty_state):
        state = {**empty_state, "error": "Something went wrong"}
        assert should_continue(state) == "end"

    def test_returns_end_on_complete(self, empty_state):
        state = {**empty_state, "is_complete": True}
        assert should_continue(state) == "end"

    def test_returns_end_on_iteration_limit(self, empty_state):
        state = {**empty_state, "iteration_count": 10}
        assert should_continue(state) == "end"

    def test_returns_plan_when_no_plan(self, empty_state):
        state = {**empty_state, "plan": []}
        assert should_continue(state) == "plan"

    def test_returns_review_when_all_steps_done(self, empty_state):
        plan = [
            {"step_number": 1, "description": "Step 1", "status": "completed", "result": "ok"},
        ]
        state = {**empty_state, "plan": plan, "current_step_index": 1}
        assert should_continue(state) == "review"

    def test_returns_act_when_steps_remain(self, empty_state):
        plan = [
            {"step_number": 1, "description": "Step 1", "status": "completed", "result": "ok"},
            {"step_number": 2, "description": "Step 2", "status": "pending", "result": None},
        ]
        state = {**empty_state, "plan": plan, "current_step_index": 1}
        assert should_continue(state) == "act"

    def test_returns_act_for_first_step(self, empty_state):
        plan = [
            {"step_number": 1, "description": "Step 1", "status": "pending", "result": None},
        ]
        state = {**empty_state, "plan": plan, "current_step_index": 0}
        assert should_continue(state) == "act"

    def test_error_takes_priority_over_plan(self, empty_state):
        plan = [
            {"step_number": 1, "description": "Step 1", "status": "pending", "result": None},
        ]
        state = {**empty_state, "plan": plan, "error": "fail"}
        assert should_continue(state) == "end"

    def test_iteration_limit_boundary(self, empty_state):
        plan = [
            {"step_number": 1, "description": "Step 1", "status": "pending", "result": None},
        ]
        state = {**empty_state, "plan": plan, "iteration_count": 9}
        assert should_continue(state) == "act"

        state["iteration_count"] = 10
        assert should_continue(state) == "end"


class TestAfterReview:
    """Tests for after_review() conditional edge function."""

    def test_returns_end_when_complete(self, empty_state):
        state = {**empty_state, "is_complete": True}
        assert after_review(state) == "end"

    def test_returns_plan_when_not_complete(self, empty_state):
        state = {**empty_state, "is_complete": False}
        assert after_review(state) == "plan"

    def test_returns_plan_when_is_complete_missing(self, empty_state):
        state = {**empty_state}
        assert after_review(state) == "plan"
