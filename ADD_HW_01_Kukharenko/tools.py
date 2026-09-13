"""Чотири доменні tools. Усі відповіді мають однаковий JSON envelope."""

from __future__ import annotations

import json
import math

from langchain_core.tools import tool

from schemas import (
    MaterialAtSiteInput,
    ReductionProposalInput,
    StockStatusInput,
    TransferSearchInput,
)
from stock_data import one_row, rows_for


def _reply(*, data: object | None = None, error: str | None = None) -> str:
    payload = {"status": "error" if error else "ok"}
    payload["error" if error else "data"] = error if error else data
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _reserve(row: dict[str, object], horizon_days: int, safety_factor: float) -> int:
    annual_usage = int(row["annual_usage"])
    planned = int(row["planned_90d"])
    horizon_usage = annual_usage * horizon_days / 365
    emergency = {"A": 2, "B": 1, "C": 0}[str(row["criticality"])]
    return math.ceil(max(planned, horizon_usage) * safety_factor + emergency)


@tool(args_schema=MaterialAtSiteInput)
def get_stock_balance(material_code: str, site: str) -> str:
    """Отримати залишок, ціну, споживання, ремонтний попит і відкриту закупівлю.

    Використовуйте як джерело фактів про один матеріал на одному підприємстві.
    """

    row = one_row(material_code, site)
    return _reply(data=row) if row else _reply(error=f"{material_code} не знайдено на {site}.")


@tool(args_schema=StockStatusInput)
def calculate_stock_status(
    material_code: str,
    site: str,
    horizon_days: int = 90,
    safety_factor: float = 1.25,
) -> str:
    """Класифікувати запас як обґрунтований, надлишковий або невизначений.

    Розрахунок детермінований: LLM не доручається арифметика, бо в неї й без того
    достатньо творчої роботи.
    """

    row = one_row(material_code, site)
    if not row:
        return _reply(error=f"{material_code} не знайдено на {site}.")
    justified = _reserve(row, horizon_days, safety_factor)
    available = int(row["qty"]) + int(row["open_purchase"])
    surplus = max(0, available - justified)
    months_without_usage = 999 if int(row["annual_usage"]) == 0 else round(12 / int(row["annual_usage"]), 1)
    category = "surplus" if surplus > max(2, justified // 2) else "justified"
    if int(row["annual_usage"]) == 0 and str(row["criticality"]) == "A":
        category = "uncertain"
    return _reply(
        data={
            "material_code": material_code,
            "site": site,
            "category": category,
            "on_hand_qty": row["qty"],
            "open_purchase_qty": row["open_purchase"],
            "justified_qty": justified,
            "surplus_qty": surplus,
            "frozen_value_uah": round(surplus * float(row["unit_price_uah"]), 2),
            "months_per_issue": months_without_usage,
            "assumption": "planned demand and historical usage are both considered",
        }
    )


@tool(args_schema=TransferSearchInput)
def find_transfer_options(material_code: str, source_site: str, min_transfer_qty: int = 1) -> str:
    """Знайти підприємства з дефіцитом того самого матеріалу перед продажем або списанням."""

    source = one_row(material_code, source_site)
    if not source:
        return _reply(error=f"{material_code} не знайдено на {source_site}.")
    source_surplus = max(0, int(source["qty"]) - _reserve(source, 90, 1.25))
    options = []
    for row in rows_for(material_code):
        if row["site"] == source_site:
            continue
        deficit = max(0, _reserve(row, 90, 1.25) - int(row["qty"]) - int(row["open_purchase"]))
        qty = min(source_surplus, deficit)
        if qty >= min_transfer_qty:
            options.append(
                {
                    "target_site": row["site"],
                    "transfer_qty": qty,
                    "avoided_purchase_uah": round(qty * float(row["unit_price_uah"]), 2),
                }
            )
    return _reply(
        data={
            "source_site": source_site,
            "source_surplus_qty": source_surplus,
            "options": options,
        }
    )


@tool(args_schema=ReductionProposalInput)
def build_reduction_proposal(
    material_code: str,
    site: str,
    action: str,
    quantity: int,
    evidence: str,
) -> str:
    """Підготувати контрольовану пропозицію скорочення без фактичного side effect.

    Tool формує лише чернетку для відповідальної особи. Продати десять двигунів
    одним натисканням тут неможливо — і це, погодьмося, непогана властивість.
    """

    row = one_row(material_code, site)
    if not row:
        return _reply(error=f"{material_code} не знайдено на {site}.")
    if quantity > int(row["qty"]):
        return _reply(error="Пропозиція перевищує фактичний залишок.")
    return _reply(
        data={
            "proposal_id": f"DRAFT-{material_code}-{site}",
            "material_code": material_code,
            "site": site,
            "action": action,
            "quantity": quantity,
            "estimated_value_uah": round(quantity * float(row["unit_price_uah"]), 2),
            "evidence": evidence,
            "executed": False,
            "approval_required": True,
        }
    )


TOOLS = [
    get_stock_balance,
    calculate_stock_status,
    find_transfer_options,
    build_reduction_proposal,
]
TOOL_BY_NAME = {item.name: item for item in TOOLS}
