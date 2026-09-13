"""Невеликий, але правдоподібний набір складських даних для демонстрації."""

from __future__ import annotations

STOCK: list[dict[str, object]] = [
    {
        "code": "BRG-6205",
        "name": "Підшипник 6205-2RS",
        "site": "PLANT-A",
        "qty": 84,
        "unit_price_uah": 315.0,
        "criticality": "B",
        "annual_usage": 36,
        "planned_90d": 8,
        "open_purchase": 20,
    },
    {
        "code": "BRG-6205",
        "name": "Підшипник 6205-2RS",
        "site": "PLANT-B",
        "qty": 4,
        "unit_price_uah": 315.0,
        "criticality": "B",
        "annual_usage": 28,
        "planned_90d": 12,
        "open_purchase": 0,
    },
    {
        "code": "PMP-SEAL-17",
        "name": "Торцеве ущільнення насоса",
        "site": "PLANT-A",
        "qty": 11,
        "unit_price_uah": 8400.0,
        "criticality": "A",
        "annual_usage": 2,
        "planned_90d": 1,
        "open_purchase": 0,
    },
    {
        "code": "PMP-SEAL-17",
        "name": "Торцеве ущільнення насоса",
        "site": "PLANT-C",
        "qty": 0,
        "unit_price_uah": 8400.0,
        "criticality": "A",
        "annual_usage": 3,
        "planned_90d": 2,
        "open_purchase": 2,
    },
    {
        "code": "VLV-DN50",
        "name": "Клапан DN50 PN16",
        "site": "PLANT-B",
        "qty": 18,
        "unit_price_uah": 2750.0,
        "criticality": "C",
        "annual_usage": 0,
        "planned_90d": 0,
        "open_purchase": 6,
    },
    {
        "code": "MTR-7K5",
        "name": "Електродвигун 7.5 кВт",
        "site": "PLANT-C",
        "qty": 3,
        "unit_price_uah": 28600.0,
        "criticality": "A",
        "annual_usage": 1,
        "planned_90d": 0,
        "open_purchase": 0,
    },
]


def rows_for(code: str) -> list[dict[str, object]]:
    """Повернути копії рядків матеріалу, щоб tool випадково не змінив fixture."""

    normalized = code.strip().upper()
    return [dict(row) for row in STOCK if row["code"] == normalized]


def one_row(code: str, site: str) -> dict[str, object] | None:
    """Знайти матеріал на конкретному підприємстві."""

    normalized_site = site.strip().upper()
    return next((row for row in rows_for(code) if row["site"] == normalized_site), None)
