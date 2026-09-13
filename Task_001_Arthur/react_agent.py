"""ReAct-підграф практичної роботи із захисними механізмами ДЗ1."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from tools import ToolRegistry
from trajectory_logger import TrajectoryLogger

SYSTEM = """Ти — ReAct-аналітик ремонтних запасів. Відповідай українською.
Для фактів викликай tools, не вигадуй ERP-дані. При помилці tool зроби одну
контрольовану fallback-спробу через search_knowledge або поясни, яких даних бракує.
Фактичну дію commit_reduction_action не викликай без окремого HITL-графа.
"""


class ReActModel(Protocol):
    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage: ...


class LiveReActModel:
    def __init__(self, registry: ToolRegistry, model_name: str = "gpt-4.1") -> None:
        self.model = ChatOpenAI(model=model_name, temperature=0, timeout=35, max_retries=1).bind_tools(registry.tools)

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        return self.model.invoke(list(messages))


class DemoReActModel:
    """Мінімальний сценарний double для тестування orchestration."""

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        query = next((str(item.content) for item in messages if isinstance(item, HumanMessage)), "")
        tool_messages = [item for item in messages if isinstance(item, ToolMessage)]
        code = (re.search(r"[A-Z]{2,}(?:-[A-Z0-9]+)+", query.upper()) or ["BRG-6205"])[0]
        site = (re.search(r"PLANT-[ABC]", query.upper()) or ["PLANT-A"])[0]
        if not tool_messages:
            return AIMessage(
                content="",
                tool_calls=[{"name": "calculate_stock_status", "args": {"material_code": code, "site": site, "horizon_days": 90}, "id": uuid.uuid4().hex, "type": "tool_call"}],
            )
        if '"status": "error"' in str(tool_messages[-1].content) and len(tool_messages) == 1:
            return AIMessage(
                content="",
                tool_calls=[{"name": "search_knowledge", "args": {"query": "правила якості даних і невизначеності", "top_k": 2}, "id": uuid.uuid4().hex, "type": "tool_call"}],
            )
        return AIMessage(content="Оцінку завершено: факти відокремлено від припущень.")


class ReActState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int
    max_steps: int
    deadline: float
    signatures: list[str]
    tools_called: list[str]
    completed: bool
    stop_reason: str | None
    answer: str


def build_react_graph(model: ReActModel, registry: ToolRegistry, logger: TrajectoryLogger):
    def agent(state: ReActState) -> dict[str, Any]:
        step = state.get("step_count", 0)
        if time.monotonic() >= state["deadline"]:
            return {"stop_reason": "timeout", "completed": True, "answer": "Зупинено за timeout."}
        if step >= state["max_steps"]:
            return {"stop_reason": "max_steps", "completed": True, "answer": "Зупинено за max_steps."}
        response = model.invoke(state["messages"])
        calls = list(response.tool_calls or [])
        signatures = [*state.get("signatures", [])]
        signatures.extend(f"{call['name']}:{json.dumps(call['args'], sort_keys=True)}" for call in calls)
        if len(signatures) >= 3 and len(set(signatures[-3:])) == 1:
            logger.append("agent", "repeat detector", "loop_detected", tool_calls=calls)
            return {"signatures": signatures, "step_count": step + 1, "stop_reason": "loop_detected", "completed": True, "answer": "Повторний виклик безпечно зупинено."}
        logger.append("agent", {"step": step + 1}, response.content, tool_calls=calls)
        return {"messages": [response], "signatures": signatures, "step_count": step + 1}

    def tools(state: ReActState) -> dict[str, Any]:
        call = state["messages"][-1].tool_calls[0]
        observation = registry.invoke(call["name"], call["args"])
        logger.append("tools", call, observation, tool_calls=[call])
        return {"messages": [ToolMessage(content=observation, tool_call_id=call["id"])], "tools_called": [*state.get("tools_called", []), call["name"]]}

    def finish(state: ReActState) -> dict[str, Any]:
        last = state["messages"][-1]
        return {"completed": True, "answer": state.get("answer") or str(last.content)}

    def route(state: ReActState) -> Literal["tools", "finish"]:
        if state.get("completed") or state.get("stop_reason"):
            return "finish"
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "finish"

    graph = StateGraph(ReActState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_node("finish", finish)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route)
    graph.add_edge("tools", "agent")
    graph.add_edge("finish", END)
    return graph.compile()


def run_react(query: str, root: str | Path, *, live: bool = False) -> dict[str, Any]:
    root = Path(root)
    registry = ToolRegistry(root / "chroma_db", root / "react_actions.db")
    try:
        model: ReActModel = LiveReActModel(registry) if live else DemoReActModel()
        app = build_react_graph(model, registry, TrajectoryLogger(root / "trajectory.json"))
        state: ReActState = {
            "messages": [SystemMessage(content=SYSTEM), HumanMessage(content=query)],
            "step_count": 0,
            "max_steps": 10,
            "deadline": time.monotonic() + 120,
            "signatures": [],
            "tools_called": [],
            "completed": False,
            "stop_reason": None,
        }
        return app.invoke(state)
    finally:
        registry.close()
