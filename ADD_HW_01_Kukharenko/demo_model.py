"""Детермінована модель для відтворюваних тестів без витрат API."""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from schemas import AgentResponse


class DemoModel:
    def next_message(self, messages: Sequence[BaseMessage]) -> AIMessage:
        query = next((str(m.content) for m in messages if isinstance(m, HumanMessage)), "")
        code = (re.search(r"[A-Z]{2,}(?:-[A-Z0-9]+)+", query.upper()) or ["BRG-6205"])[0]
        site = (re.search(r"PLANT-[ABC]", query.upper()) or ["PLANT-A"])[0]
        observations = [m for m in messages if isinstance(m, ToolMessage)]
        if not observations:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate_stock_status",
                        "args": {
                            "material_code": code,
                            "site": site,
                            "horizon_days": 90,
                            "safety_factor": 1.25,
                        },
                        "id": uuid.uuid4().hex,
                        "type": "tool_call",
                    }
                ],
            )
        if len(observations) == 1 and "перем" in query.casefold():
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "find_transfer_options",
                        "args": {
                            "material_code": code,
                            "source_site": site,
                            "min_transfer_qty": 1,
                        },
                        "id": uuid.uuid4().hex,
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="Факти отримано. Формую обережну рекомендацію.")

    def structure(self, messages: Sequence[BaseMessage], sources: list[str]) -> AgentResponse:
        text = " ".join(str(m.content) for m in messages if isinstance(m, ToolMessage))
        if '"category": "surplus"' in text:
            status = "surplus"
            action = "Перевірити відкриті закупівлі та погодити внутрішнє переміщення."
        elif '"status": "error"' in text:
            status = "not_found"
            action = "Уточнити код матеріалу або підприємство."
        else:
            status = "justified"
            action = "Зберегти резерв і переглянути оцінку після нового ремонтного плану."
        return AgentResponse(
            summary="Оцінку виконано за залишком, попитом і відкритою закупівлею.",
            status=status,
            recommended_actions=[action],
            confidence=0.86,
            sources=sources,
            stop_reason="completed",
        )
