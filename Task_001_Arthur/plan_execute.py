"""LangGraph planner → executor → replanner з SQLite persistence та HITL."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from demo_models import ModelSuite
from tools import RISKY_TOOL, ToolRegistry


class PlanExecuteState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    plan: list[str]
    current_step: int
    results: list[dict[str, Any]]
    completed: bool
    pending_action: dict[str, Any] | None
    tools_called: list[str]
    final_answer: str


def initial_state(query: str) -> PlanExecuteState:
    return {
        "messages": [],
        "user_query": query,
        "plan": [],
        "current_step": 0,
        "results": [],
        "completed": False,
        "pending_action": None,
        "tools_called": [],
        "final_answer": "",
    }


def _requests_commit(query: str) -> bool:
    """Відрізнити пораду від прямої команди на side effect."""

    lowered = query.casefold()
    negatives = ("без проведення", "не проводь", "не фіксуй", "лише порад")
    positives = ("зафіксуй", "проведи", "виконай", "спиши", "зупини закупівлю")
    return not any(item in lowered for item in negatives) and any(item in lowered for item in positives)


def build_workflow(models: ModelSuite, registry: ToolRegistry) -> StateGraph:
    def planner(state: PlanExecuteState) -> dict[str, Any]:
        plan = models.create_plan(
            [
                SystemMessage(
                    content="Склади повний, атомарний план аналізу запасу. Для фактичної дії останнім кроком постав commit_reduction_action; підтвердження виконає граф."
                ),
                HumanMessage(content=state["user_query"]),
            ]
        )
        steps = list(plan.steps)
        if not _requests_commit(state["user_query"]):
            steps = [step for step in steps if RISKY_TOOL not in step]
        elif not any(RISKY_TOOL in step for step in steps):
            steps.append(f"Записати погоджену дію через {RISKY_TOOL}")
        return {
            "messages": [
                HumanMessage(content=state["user_query"]),
                AIMessage(content=f"Мета: {plan.goal}\nПлан: {plan.steps}"),
            ],
            "plan": steps,
            "current_step": 0,
            "results": [],
            "completed": False,
        }

    def executor(state: PlanExecuteState) -> dict[str, Any]:
        index = state["current_step"]
        step = state["plan"][index]
        response = models.choose_action(
            [
                SystemMessage(content="Виконай рівно один поточний крок через відповідний tool. Не вигадуй observation."),
                HumanMessage(content=f"Запит: {state['user_query']}\nКрок: {step}\nПопередні результати: {json.dumps(state['results'], ensure_ascii=False)}"),
            ]
        )
        if not response.tool_calls:
            result = {"step": index + 1, "tool": None, "observation": response.content}
            return {"messages": [response], "current_step": index + 1, "results": [*state["results"], result]}
        call = response.tool_calls[0]
        if call["name"] == RISKY_TOOL:
            return {"messages": [response], "pending_action": call}
        observation = registry.invoke(call["name"], call["args"])
        result = {"step": index + 1, "tool": call["name"], "observation": json.loads(observation)}
        return {
            "messages": [response, ToolMessage(content=observation, tool_call_id=call["id"])],
            "current_step": index + 1,
            "results": [*state["results"], result],
            "tools_called": [*state.get("tools_called", []), call["name"]],
        }

    def approval(state: PlanExecuteState) -> dict[str, Any]:
        call = dict(state["pending_action"] or {})
        decision = interrupt(
            {
                "message": "Підтвердити ризикову дію",
                "tool": call.get("name"),
                "args": call.get("args"),
                "allowed_actions": ["approve", "reject", "edit"],
            }
        )
        if decision.get("action") == "reject":
            result = {
                "step": state["current_step"] + 1,
                "tool": call["name"],
                "observation": {"status": "rejected", "reason": decision.get("reason", "Без пояснення")},
            }
            return {
                "pending_action": None,
                "current_step": state["current_step"] + 1,
                "results": [*state["results"], result],
                "tools_called": [*state.get("tools_called", []), f"{call['name']}:rejected"],
            }
        args = dict(call["args"])
        if decision.get("action") == "edit":
            args.update(decision.get("args", {}))
        observation = registry.invoke(call["name"], args)
        result = {
            "step": state["current_step"] + 1,
            "tool": call["name"],
            "observation": json.loads(observation),
            "approval": decision.get("action"),
        }
        return {
            "pending_action": None,
            "current_step": state["current_step"] + 1,
            "results": [*state["results"], result],
            "tools_called": [*state.get("tools_called", []), call["name"]],
        }

    def replanner(state: PlanExecuteState) -> dict[str, Any]:
        decision = models.replan(
            [
                HumanMessage(
                    content=json.dumps(
                        {
                            "plan": state["plan"],
                            "current_step": state["current_step"],
                            "results": state["results"],
                        },
                        ensure_ascii=False,
                    )
                )
            ]
        )
        if decision.action == "finish" or state["current_step"] >= len(state["plan"]):
            risky_called = RISKY_TOOL in state.get("tools_called", [])
            suffix = "Ризикову дію виконано лише після рішення людини." if risky_called else "Ризикові дії не запускалися."
            answer = f"План завершено. Фактичні дані, розрахунок і політика зібрані. {suffix}"
            return {"completed": True, "final_answer": answer}
        if decision.action == "replan":
            prefix = state["plan"][: state["current_step"]]
            return {"plan": [*prefix, *(decision.updated_steps or [])]}
        return {}

    def after_executor(state: PlanExecuteState) -> Literal["approval", "replanner"]:
        return "approval" if state.get("pending_action") else "replanner"

    def after_replanner(state: PlanExecuteState) -> Literal["executor", "__end__"]:
        return "__end__" if state.get("completed") else "executor"

    graph = StateGraph(PlanExecuteState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("approval", approval)
    graph.add_node("replanner", replanner)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", after_executor)
    graph.add_edge("approval", "replanner")
    graph.add_conditional_edges("replanner", after_replanner, {"executor": "executor", "__end__": END})
    return graph


class AgentRuntime:
    def __init__(self, root: str | Path, models: ModelSuite, *, pause_after_executor: bool = False) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = ToolRegistry(self.root / "chroma_db", self.root / "actions.db")
        self.models = models
        self.connection = sqlite3.connect(self.root / "agent_state.db", check_same_thread=False)
        self.saver = SqliteSaver(self.connection)
        self.app = build_workflow(models, self.registry).compile(checkpointer=self.saver, interrupt_after=["executor"] if pause_after_executor else None)

    def run(self, query: str, thread_id: str) -> dict[str, Any]:
        return self.app.invoke(initial_state(query), config={"configurable": {"thread_id": thread_id}})

    def resume(self, thread_id: str, decision: dict[str, Any] | None = None) -> dict[str, Any]:
        resume_input: Command | None = None if decision is None else Command(resume=decision)
        return self.app.invoke(resume_input, config={"configurable": {"thread_id": thread_id}})

    def snapshot(self, thread_id: str):
        return self.app.get_state({"configurable": {"thread_id": thread_id}})

    def close(self) -> None:
        self.registry.close()
        self.connection.close()
