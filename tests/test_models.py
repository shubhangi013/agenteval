from agenteval.models import TaskSpec, deterministic_task_id


def test_task_id_is_deterministic_and_input_order_independent() -> None:
    left = deterministic_task_id("run", "task", {"b": 2, "a": 1})
    right = deterministic_task_id("run", "task", {"a": 1, "b": 2})
    assert left == right
    assert len(left) == 24


def test_with_id_preserves_existing_id() -> None:
    task = TaskSpec(
        id="external-id",
        run_id="run",
        name="task",
        description="test",
        tool="calculator",
        tool_input={"expression": "1+1"},
        acceptance_criteria="equals two",
    )
    assert task.with_id().id == "external-id"
