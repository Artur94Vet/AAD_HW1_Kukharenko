"""Pydantic v2 контракти інструментів складського агента."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

SITES = {"PLANT-A", "PLANT-B", "PLANT-C"}


class MaterialAtSiteInput(BaseModel):
    """Параметри читання залишку одного матеріалу."""

    material_code: str = Field(description="Код матеріалу, наприклад BRG-6205.")
    site: str = Field(description="Підприємство: PLANT-A, PLANT-B або PLANT-C.")

    @field_validator("material_code")
    @classmethod
    def code_is_safe(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned or len(cleaned) > 32 or not all(ch.isalnum() or ch in "-_" for ch in cleaned):
            raise ValueError("Код матеріалу має містити 1–32 літери, цифри, '-' або '_'.")
        return cleaned

    @field_validator("site")
    @classmethod
    def site_exists(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if cleaned not in SITES:
            raise ValueError(f"Невідоме підприємство. Доступні: {', '.join(sorted(SITES))}.")
        return cleaned


class StockStatusInput(MaterialAtSiteInput):
    """Параметри консервативної класифікації запасу."""

    horizon_days: int = Field(default=90, description="Горизонт оцінки від 30 до 365 днів.")
    safety_factor: float = Field(default=1.25, description="Коефіцієнт аварійного резерву 1.0–3.0.")

    @field_validator("horizon_days")
    @classmethod
    def horizon_is_reasonable(cls, value: int) -> int:
        if not 30 <= value <= 365:
            raise ValueError("Горизонт має бути від 30 до 365 днів.")
        return value

    @field_validator("safety_factor")
    @classmethod
    def factor_is_reasonable(cls, value: float) -> float:
        if not 1.0 <= value <= 3.0:
            raise ValueError("Коефіцієнт резерву має бути від 1.0 до 3.0.")
        return value


class TransferSearchInput(BaseModel):
    """Параметри пошуку міжзаводського переміщення."""

    material_code: str = Field(description="Код матеріалу для балансування.")
    source_site: str = Field(description="Підприємство, де перевіряємо надлишок.")
    min_transfer_qty: int = Field(default=1, description="Мінімальна корисна кількість переміщення.")

    @field_validator("material_code")
    @classmethod
    def normalize_code(cls, value: str) -> str:
        return MaterialAtSiteInput(material_code=value, site="PLANT-A").material_code

    @field_validator("source_site")
    @classmethod
    def normalize_site(cls, value: str) -> str:
        return MaterialAtSiteInput(material_code="X", site=value).site

    @field_validator("min_transfer_qty")
    @classmethod
    def transfer_is_positive(cls, value: int) -> int:
        if not 1 <= value <= 10_000:
            raise ValueError("Кількість переміщення має бути від 1 до 10 000.")
        return value


class ReductionProposalInput(MaterialAtSiteInput):
    """Параметри безпечної, лише рекомендаційної пропозиції."""

    action: Literal["transfer", "stop_purchase", "sell", "write_off"] = Field(
        description="Запропонована дія; цей tool нічого не проводить в обліковій системі."
    )
    quantity: int = Field(description="Кількість одиниць у пропозиції.")
    evidence: str = Field(description="Коротке обґрунтування на основі розрахунків.")

    @field_validator("quantity")
    @classmethod
    def quantity_is_positive(cls, value: int) -> int:
        if not 1 <= value <= 100_000:
            raise ValueError("Кількість має бути від 1 до 100 000.")
        return value

    @field_validator("evidence")
    @classmethod
    def evidence_is_present(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 12:
            raise ValueError("Обґрунтування надто коротке: потрібно щонайменше 12 символів.")
        return cleaned


class AgentResponse(BaseModel):
    """Структурована фінальна відповідь ReAct-агента."""

    summary: str = Field(description="Стислий висновок українською.")
    status: Literal["justified", "surplus", "uncertain", "not_found", "stopped"] = Field(
        description="Підсумкова категорія стану запасу."
    )
    recommended_actions: list[str] = Field(description="Конкретні наступні дії.")
    confidence: float = Field(ge=0, le=1, description="Впевненість від 0 до 1.")
    sources: list[str] = Field(description="Назви tools, на яких базується відповідь.")
    stop_reason: str = Field(description="completed або причина захисної зупинки.")
