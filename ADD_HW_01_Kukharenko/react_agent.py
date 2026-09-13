"""ReAct у LangGraph: agent → tools → agent, а потім structured formatter."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from schemas import AgentResponse
from tools import TOOLS
from trajectory_logger import TrajectoryLogger

SYSTEM_PROMPT = """Ти — уважний ReAct-агент з аналізу ремонтних запасів групи підприємств.
Працюй українською. Факти й розрахунки отримуй тільки через tools. Спочатку перевір
обґрунтованість залишку, потім — можливість переміщення. Не називай запас надлишком,
якщо даних недостатньо. Не розкривай приховані міркування; покажи факти, припущення
та практичну дію. Фактичні списання, продажі й переміщення не виконуй.
"""


class ModelPort(Protocol):
    def next_message(self, messages: Sequence[BaseMessage]) -> AIMessage: ...
    def structure(self, messages: Sequence[BaseMessage], sources: list[str]) -> AgentResponse: ...


class OpenAIModel:
    """Тонкий адаптер: production модель легко замінити у тестах."""

    def __init__(self, model_name: str = "gpt-4.1", timeout: float = 30) -> None:
        model = ChatOpenAI(model=model_name, temperature=0, timeout=timeout, max_retries=1)
        self._tool_model = model.bind_tools(TOOLS)
        self._structured_model = model.with_structured_output(AgentResponse)

    def next_message(self, messages: Sequence[BaseMessage]) -> AIMessage:
        return self._tool_model.invoke(list(messages))

    def structure(self, messages: Sequence[BaseMessage], sources: list[str]) -> AgentResponse:
        prompt = HumanMessage(
            content=(
                "Сформуй фінальну AgentResponse лише з наявних observations. "
                f"Поле sources має дорівнювати {json.dumps(sources, ensure_ascii=False)}."
            )
        )
        result = self._structured_model.invoke([*messages, prompt])
        return result.model_copy(update={"sources": sources, "stop_reason": "completed"})


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int
    max_steps: int
    deadline: float
    call_history: list[str]
    tools_used: list[str]
    stop_reason: str | None
    final_response: dict[str, Any]


def _signature(call: dict[str, Any]) -> str:
    return f"{call['name']}:{json.dumps(call.get('args', {}), sort_keys=True, ensure_ascii=False)}"


def _stop_response(reason: str, sources: list[str]) -> AgentResponse:
    labels = {
        "max_steps": "Досягнуто ліміт кроків; повернуто частковий результат.",
        "timeout": "Час виконання вичерпано; незавершені дії не запускалися.",
        "loop_detected": "Повторний виклик з тими самими аргументами зупинено.",
    }
    return AgentResponse(
        summary=labels[reason],
        status="stopped",
        recommended_actions=["Перевірити вхідні дані та повторити запит."],
        confidence=0.2,
        sources=sources,
        stop_reason=reason,
    )


def build_graph(model: ModelPort, logger: TrajectoryLogger):
    tool_node = ToolNode(TOOLS)

    def agent(state: AgentState) -> dict[str, Any]:
        step = state.get("step_count", 0)
        if time.monotonic() >= state["deadline"]:
            return {"stop_reason": "timeout"}
        if step >= state["max_steps"]:
            return {"stop_reason": "max_steps"}
        answer = model.next_message(state["messages"])
        calls = list(answer.tool_calls or [])
        history = list(state.get("call_history", []))
        history.extend(_signature(call) for call in calls)
        repeated = len(history) >= 3 and len(set(history[-3:])) == 1
        reason = "loop_detected" if repeated else None
        logger.append("agent", {"step": step + 1}, answer.content, tool_calls=calls)
        return {
            "messages": [answer],
            "step_count": step + 1,
            "call_history": history,
            "stop_reason": reason,
        }

    def tools(state: AgentState) -> dict[str, Any]:
        if time.monotonic() >= state["deadline"]:
            return {"stop_reason": "timeout"}
        answer = state["messages"][-1]
        calls = list(answer.tool_calls) if isinstance(answer, AIMessage) else []
        result = tool_node.invoke({"messages": state["messages"]})
        used = [*state.get("tools_used", [])]
        used.extend(call["name"] for call in calls if call["name"] not in used)
        logger.append(
            "tools",
            calls,
            [message.content for message in result["messages"]],
            tool_calls=calls,
        )
        return {"messages": result["messages"], "tools_used": used}

    def formatter(state: AgentState) -> dict[str, Any]:
        reason = state.get("stop_reason")
        response = (
            _stop_response(reason, state.get("tools_used", []))
            if reason
            else model.structure(state["messages"], state.get("tools_used", []))
        )
        logger.append("formatter", {"reason": reason}, response.model_dump())
        return {"final_response": response.model_dump()}

    def after_agent(state: AgentState) -> Literal["tools", "formatter"]:
        if state.get("stop_reason"):
            return "formatter"
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "formatter"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_node("formatter", formatter)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, {"tools": "tools", "formatter": "formatter"})
    graph.add_edge("tools", "agent")
    graph.add_edge("formatter", END)
    return graph.compile()


def initial_state(query: str, *, max_steps: int = 8, timeout_seconds: float = 45) -> AgentState:
    return {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)],
        "step_count": 0,
        "max_steps": max_steps,
        "deadline": time.monotonic() + timeout_seconds,
        "call_history": [],
        "tools_used": [],
        "stop_reason": None,
    }
