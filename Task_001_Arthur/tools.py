"""Tool registry: read-only аналіз, RAG і одна ризикова side-effect дія."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from knowledge import KnowledgeBase
from schemas import CommitActionInput, MaterialInput, SearchKnowledgeInput, StatusInput
from stock_data import one_row


def envelope(data: object | None = None, error: str | None = None) -> str:
    return json.dumps(
        {"status": "error" if error else "ok", "error" if error else "data": error if error else data},
        ensure_ascii=False,
        indent=2,
    )


class ToolRegistry:
    def __init__(self, chroma_path: str | Path, actions_db: str | Path) -> None:
        self.knowledge = KnowledgeBase(chroma_path)
        self.connection = sqlite3.connect(actions_db, check_same_thread=False)
        self.connection.execute("CREATE TABLE IF NOT EXISTS actions (idempotency_key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self.connection.commit()
        self.tools = self._build()
        self.by_name = {item.name: item for item in self.tools}

    def _build(self) -> list[BaseTool]:
        knowledge = self.knowledge
        connection = self.connection

        @tool(args_schema=MaterialInput)
        def get_stock_balance(material_code: str, site: str) -> str:
            """Отримати залишок, ціну, споживання, плановий попит та відкриту закупівлю."""
            row = one_row(material_code, site)
            return envelope(row) if row else envelope(error=f"{material_code} не знайдено на {site}.")

        @tool(args_schema=StatusInput)
        def calculate_stock_status(material_code: str, site: str, horizon_days: int = 90) -> str:
            """Детерміновано оцінити justified/surplus/uncertain і заморожену вартість."""
            row = one_row(material_code, site)
            if not row:
                return envelope(error=f"{material_code} не знайдено на {site}.")
            demand = max(int(row["planned_90d"]), int(row["annual_usage"]) * horizon_days / 365)
            reserve = {"A": 2, "B": 1, "C": 0}[str(row["criticality"])]
            justified = math.ceil(demand * 1.25 + reserve)
            future = int(row["qty"]) + int(row["open_purchase"])
            surplus = max(0, future - justified)
            category = "uncertain" if int(row["annual_usage"]) == 0 and str(row["criticality"]) == "A" else ("surplus" if surplus > max(2, justified // 2) else "justified")
            return envelope(
                {
                    "material_code": material_code,
                    "site": site,
                    "category": category,
                    "justified_qty": justified,
                    "future_qty": future,
                    "surplus_qty": surplus,
                    "frozen_value_uah": round(surplus * float(row["unit_price_uah"]), 2),
                }
            )

        @tool(args_schema=SearchKnowledgeInput)
        def search_knowledge(query: str, top_k: int = 3) -> str:
            """Знайти у ChromaDB правила резерву, переміщення, закупівлі та вибуття."""
            return envelope({"query": query, "results": knowledge.search(query, top_k)})

        @tool(args_schema=CommitActionInput)
        def commit_reduction_action(material_code: str, site: str, action: str, quantity: int, reason: str, idempotency_key: str) -> str:
            """Записати погоджену дію до локального реєстру; виклик дозволений лише після HITL."""
            row = one_row(material_code, site)
            if not row:
                return envelope(error=f"{material_code} не знайдено на {site}.")
            if quantity > int(row["qty"]) and action != "stop_purchase":
                return envelope(error="Кількість перевищує наявний залишок.")
            payload = {
                "material_code": material_code,
                "site": site,
                "action": action,
                "quantity": quantity,
                "reason": reason,
                "idempotency_key": idempotency_key,
            }
            before = connection.total_changes
            connection.execute(
                "INSERT OR IGNORE INTO actions VALUES (?, ?)",
                (idempotency_key, json.dumps(payload, ensure_ascii=False)),
            )
            connection.commit()
            return envelope({**payload, "created": connection.total_changes > before})

        return [get_stock_balance, calculate_stock_status, search_knowledge, commit_reduction_action]

    def invoke(self, name: str, args: dict[str, Any]) -> str:
        return str(self.by_name[name].invoke(args))

    def action_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM actions").fetchone()[0])

    def close(self) -> None:
        self.connection.close()


RISKY_TOOL = "commit_reduction_action"
