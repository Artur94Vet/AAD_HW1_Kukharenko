import time
import uuid

from langchain_core.messages import AIMessage

from demo_model import DemoModel
from react_agent import build_graph, initial_state
from trajectory_logger import TrajectoryLogger


class LoopModel(DemoModel):
    def next_message(self, messages):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculate_stock_status",
                    "args": {"material_code": "BRG-6205", "site": "PLANT-A"},
                    "id": uuid.uuid4().hex,
                    "type": "tool_call",
                }
            ],
        )


def test_react_cycle_and_structured_output(tmp_path) -> None:
    result = build_graph(DemoModel(), TrajectoryLogger(tmp_path / "trajectory.json")).invoke(
        initial_state("Оціни BRG-6205 на PLANT-A")
    )
    assert result["final_response"]["status"] == "surplus"
    assert "calculate_stock_status" in result["final_response"]["sources"]


def test_repeat_detector_stops_third_identical_call(tmp_path) -> None:
    result = build_graph(LoopModel(), TrajectoryLogger(tmp_path / "loop.json")).invoke(
        initial_state("loop", max_steps=8)
    )
    assert result["final_response"]["stop_reason"] == "loop_detected"


def test_max_steps_is_visible(tmp_path) -> None:
    result = build_graph(LoopModel(), TrajectoryLogger(tmp_path / "max.json")).invoke(
        initial_state("limit", max_steps=1)
    )
    assert result["final_response"]["stop_reason"] == "max_steps"


def test_timeout_is_visible(tmp_path) -> None:
    state = initial_state("timeout")
    state["deadline"] = time.monotonic() - 1
    result = build_graph(DemoModel(), TrajectoryLogger(tmp_path / "timeout.json")).invoke(state)
    assert result["final_response"]["stop_reason"] == "timeout"
