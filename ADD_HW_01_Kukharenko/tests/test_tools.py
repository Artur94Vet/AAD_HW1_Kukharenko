import json

import pytest
from pydantic import ValidationError

from schemas import ReductionProposalInput, StockStatusInput
from tools import calculate_stock_status, find_transfer_options


def payload(result: str) -> dict:
    return json.loads(result)


def test_surplus_has_frozen_value() -> None:
    data = payload(calculate_stock_status.invoke({"material_code": "BRG-6205", "site": "PLANT-A"}))["data"]
    assert data["category"] == "surplus"
    assert data["frozen_value_uah"] > 0


def test_transfer_prefers_internal_need() -> None:
    data = payload(find_transfer_options.invoke({"material_code": "BRG-6205", "source_site": "PLANT-A"}))[
        "data"
    ]
    assert data["options"][0]["target_site"] == "PLANT-B"


@pytest.mark.parametrize("field,value", [("horizon_days", 7), ("horizon_days", 500), ("safety_factor", 0.5)])
def test_status_schema_rejects_bad_values(field: str, value: object) -> None:
    kwargs = {"material_code": "BRG-6205", "site": "PLANT-A", field: value}
    with pytest.raises(ValidationError):
        StockStatusInput(**kwargs)


def test_proposal_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="надто коротке"):
        ReductionProposalInput(
            material_code="BRG-6205",
            site="PLANT-A",
            action="transfer",
            quantity=2,
            evidence="бо треба",
        )
