import json
import time
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from demo_models import DemoModelSuite
from plan_execute import AgentRuntime
from react_agent import ReActState, build_react_graph, run_react
from schemas import CommitActionInput, MaterialInput, SearchKnowledgeInput, StatusInput
from tools import ToolRegistry
from trajectory_logger import TrajectoryLogger


@pytest.mark.parametrize(
    "model,kwargs",
    [
        (MaterialInput, {"material_code": "", "site": "PLANT-A"}),
        (MaterialInput, {"material_code": "BRG-6205", "site": "UNKNOWN"}),
        (StatusInput, {"material_code": "BRG-6205", "site": "PLANT-A", "horizon_days": 1}),
        (SearchKnowledgeInput, {"query": "x", "top_k": 3}),
        (CommitActionInput, {"material_code": "BRG-6205", "site": "PLANT-A", "action": "sell", "quantity": 1, "reason": "short", "idempotency_key": "abcdefgh"}),
    ],
)
def test_five_schema_failures(model, kwargs) -> None:
    with pytest.raises(ValidationError):
        model(**kwargs)


def test_tools_return_standard_envelope(tmp_path) -> None:
    registry = ToolRegistry(tmp_path / "chroma", tmp_path / "actions.db")
    try:
        response = json.loads(registry.invoke("get_stock_balance", {"material_code": "BRG-6205", "site": "PLANT-A"}))
        assert response["status"] == "ok" and "data" in response
    finally:
        registry.close()


def test_react_cycle_and_fallback(tmp_path) -> None:
    result = run_react("Оціни UNKNOWN-1 на PLANT-A", tmp_path)
    assert result["tools_called"] == ["calculate_stock_status", "search_knowledge"]
    assert result["completed"] is True


def test_plan_execute_cycle(tmp_path) -> None:
    query = "Підготуй план BRG-6205 на PLANT-A без проведення"
    runtime = AgentRuntime(tmp_path, DemoModelSuite(query))
    try:
        result = runtime.run(query, "plan")
        assert result["completed"] is True and result["current_step"] == 3
    finally:
        runtime.close()


class RepeatingModel:
    def invoke(self, messages):
        return AIMessage(
            content="", tool_calls=[{"name": "get_stock_balance", "args": {"material_code": "BRG-6205", "site": "PLANT-A"}, "id": uuid.uuid4().hex, "type": "tool_call"}]
        )


def test_repeat_guard(tmp_path) -> None:
    registry = ToolRegistry(tmp_path / "chroma", tmp_path / "actions.db")
    try:
        state: ReActState = {
            "messages": [SystemMessage(content="test"), HumanMessage(content="test")],
            "step_count": 0,
            "max_steps": 10,
            "deadline": time.monotonic() + 120,
            "signatures": [],
            "tools_called": [],
            "completed": False,
        }
        result = build_react_graph(RepeatingModel(), registry, TrajectoryLogger(tmp_path / "trajectory.json")).invoke(state)
        assert result["stop_reason"] == "loop_detected"
    finally:
        registry.close()
