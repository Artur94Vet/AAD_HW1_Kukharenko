"""Обов'язкові structured outputs і Pydantic-контракти tools."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SITES = {"PLANT-A", "PLANT-B", "PLANT-C"}


class Plan(BaseModel):
    """Повний план, який planner формує до початку виконання."""

    goal: str = Field(description="Головна ціль запиту користувача.")
    steps: list[str] = Field(min_length=1, max_length=8, description="Від одного до восьми атомарних кроків.")


class ReplanDecision(BaseModel):
    """Рішення replanner після одного виконаного кроку."""

    action: Literal["continue", "replan", "finish"] = Field(description="Продовжити, замінити решту плану або завершити.")
    updated_steps: list[str] | None = Field(default=None, description="Нові невиконані кроки лише для replan.")
    reasoning: str = Field(description="Коротке обґрунтування рішення.")

    @model_validator(mode="after")
    def replan_has_steps(self) -> ReplanDecision:
        if self.action == "replan" and not self.updated_steps:
            raise ValueError("Для action='replan' потрібні updated_steps.")
        return self


class MaterialInput(BaseModel):
    material_code: str = Field(description="Код матеріалу у складському довіднику.")
    site: str = Field(description="Підприємство PLANT-A, PLANT-B або PLANT-C.")

    @field_validator("material_code")
    @classmethod
    def valid_code(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9_-]{0,31}", cleaned):
            raise ValueError("Некоректний код матеріалу.")
        return cleaned

    @field_validator("site")
    @classmethod
    def valid_site(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if cleaned not in SITES:
            raise ValueError(f"Невідоме підприємство: {cleaned}.")
        return cleaned


class StatusInput(MaterialInput):
    horizon_days: int = Field(default=90, description="Горизонт оцінки 30–365 днів.")

    @field_validator("horizon_days")
    @classmethod
    def valid_horizon(cls, value: int) -> int:
        if not 30 <= value <= 365:
            raise ValueError("Горизонт має бути від 30 до 365 днів.")
        return value


class SearchKnowledgeInput(BaseModel):
    query: str = Field(description="Запит про політики резерву, переміщення, списання або закупівель.")
    top_k: int = Field(default=3, description="Кількість документів від 1 до 5.")

    @field_validator("query")
    @classmethod
    def query_is_useful(cls, value: str) -> str:
        cleaned = value.strip()
        if not 3 <= len(cleaned) <= 500:
            raise ValueError("Запит має містити від 3 до 500 символів.")
        return cleaned

    @field_validator("top_k")
    @classmethod
    def top_k_is_small(cls, value: int) -> int:
        if not 1 <= value <= 5:
            raise ValueError("top_k має бути від 1 до 5.")
        return value


class CommitActionInput(MaterialInput):
    action: Literal["transfer", "stop_purchase", "sell", "write_off"] = Field(description="Дія, яку буде записано до реєстру після HITL.")
    quantity: int = Field(description="Кількість одиниць.")
    reason: str = Field(description="Перевірене обґрунтування для аудиту.")
    idempotency_key: str = Field(description="Стабільний ключ 8–64 символи проти дублювання.")

    @field_validator("quantity")
    @classmethod
    def valid_quantity(cls, value: int) -> int:
        if not 1 <= value <= 100_000:
            raise ValueError("Кількість має бути від 1 до 100 000.")
        return value

    @field_validator("reason")
    @classmethod
    def valid_reason(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 15:
            raise ValueError("Причина має містити щонайменше 15 символів.")
        return cleaned

    @field_validator("idempotency_key")
    @classmethod
    def valid_key(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not re.fullmatch(r"[a-z0-9_-]{8,64}", cleaned):
            raise ValueError("idempotency_key: 8–64 lowercase символи [a-z0-9_-].")
        return cleaned
