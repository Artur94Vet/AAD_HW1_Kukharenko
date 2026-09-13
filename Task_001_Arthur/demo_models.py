"""Live OpenAI suite та відтворюваний навчальний double."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from schemas import Plan, ReplanDecision
from tools import ToolRegistry


class ModelSuite(Protocol):
    def create_plan(self, messages: Sequence[BaseMessage]) -> Plan: ...
    def choose_action(self, messages: Sequence[BaseMessage]) -> AIMessage: ...
    def replan(self, messages: Sequence[BaseMessage]) -> ReplanDecision: ...


class OpenAIModelSuite:
    def __init__(self, registry: ToolRegistry, model_name: str = "gpt-4.1") -> None:
        model = ChatOpenAI(model=model_name, temperature=0, timeout=35, max_retries=1)
        self.planner = model.with_structured_output(Plan)
        self.executor = model.bind_tools(registry.tools)
        self.replanner = model.with_structured_output(ReplanDecision)

    def create_plan(self, messages: Sequence[BaseMessage]) -> Plan:
        return self.planner.invoke(list(messages))

    def choose_action(self, messages: Sequence[BaseMessage]) -> AIMessage:
        return self.executor.invoke(list(messages))

    def replan(self, messages: Sequence[BaseMessage]) -> ReplanDecision:
        return self.replanner.invoke(list(messages))


class DemoModelSuite:
    """Залишає orchestration тестованим, не вдаючи з себе мовну модель."""

    def __init__(self, query: str = "") -> None:
        self.query = query

    def create_plan(self, messages: Sequence[BaseMessage]) -> Plan:
        query = self.query or next((str(m.content) for m in messages if isinstance(m, HumanMessage)), "")
        steps = [
            "Отримати фактичний залишок через get_stock_balance",
            "Розрахувати статус через calculate_stock_status",
            "Перевірити політику через search_knowledge",
        ]
        if any(word in query.casefold() for word in ("зафіксуй", "виконай", "проведи", "спиши", "зупини закупівлю")):
            steps.append("Записати погоджену дію через commit_reduction_action")
        return Plan(goal="Сформувати доказовий план скорочення запасу", steps=steps)

    def choose_action(self, messages: Sequence[BaseMessage]) -> AIMessage:
        text = str(messages[-1].content)
        query = " ".join(str(m.content) for m in messages)
        step_match = re.search(r"Крок:\s*(.+?)(?:\n|$)", text)
        current_step = step_match.group(1) if step_match else text
        code = (re.search(r"[A-Z]{2,}(?:-[A-Z0-9]+)+", query.upper()) or ["BRG-6205"])[0]
        site = (re.search(r"PLANT-[ABC]", query.upper()) or ["PLANT-A"])[0]
        if "get_stock_balance" in current_step:
            name, args = "get_stock_balance", {"material_code": code, "site": site}
        elif "calculate_stock_status" in current_step:
            name, args = "calculate_stock_status", {"material_code": code, "site": site, "horizon_days": 90}
        elif "search_knowledge" in current_step:
            name, args = (
                "search_knowledge",
                {"query": "політика внутрішнього переміщення та зупинки закупівлі", "top_k": 3},
            )
        else:
            name, args = (
                "commit_reduction_action",
                {
                    "material_code": code,
                    "site": site,
                    "action": "stop_purchase",
                    "quantity": 20,
                    "reason": "Надлишок підтверджено розрахунком і політикою",
                    "idempotency_key": f"demo-{code.lower()}-{site.lower()}",
                },
            )
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": uuid.uuid4().hex, "type": "tool_call"}])

    def replan(self, messages: Sequence[BaseMessage]) -> ReplanDecision:
        state = json.loads(str(messages[-1].content))
        action = "finish" if state["current_step"] >= len(state["plan"]) else "continue"
        return ReplanDecision(action=action, reasoning="Поточний крок завершено; перевіряємо залишок плану.")
