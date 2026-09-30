from datetime import date

import pandas as pd
import pytest

from market_pipeline.schemas import ContractViolationError, split_valid
from market_pipeline.sources import SyntheticSource


@pytest.fixture
def bars() -> pd.DataFrame:
    return SyntheticSource().fetch(["AAA", "BBB"], date(2024, 1, 1), date(2024, 2, 1))


def test_clean_data_passes(bars: pd.DataFrame) -> None:
    valid, rejected = split_valid(bars)
    assert len(valid) == len(bars)
    assert rejected.empty


def test_bad_rows_are_quarantined_with_reasons(bars: pd.DataFrame) -> None:
    bars.loc[1, "high"] = bars.loc[1, "low"] * 0.5  # inverted bar
    bars.loc[2, "close"] = -3.0  # negative price
    bars.loc[3, "close"] = None  # in-progress session from the vendor
    valid, rejected = split_valid(bars)

    assert len(valid) == len(bars) - 3
    reasons = dict(zip(rejected.index, rejected["_reject_reason"], strict=True))
    assert "high must be >=" in reasons[1]
    assert "close:greater_than(0)" in reasons[2]
    assert "close:not_nullable" in reasons[3]


def test_duplicate_keys_are_rejected(bars: pd.DataFrame) -> None:
    dup = pd.concat([bars, bars.iloc[[0]]], ignore_index=True)
    _, rejected = split_valid(dup)
    assert len(rejected) == 2  # both copies flagged; ingest dedupes before validating


def test_structural_break_raises(bars: pd.DataFrame) -> None:
    with pytest.raises(ContractViolationError, match="column_in_dataframe"):
        split_valid(bars.drop(columns=["volume"]))
