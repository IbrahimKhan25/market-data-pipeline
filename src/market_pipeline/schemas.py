"""Data contracts for ingested price bars.

Two kinds of failure are handled differently:

* **Row-level** violations (a negative price, high < low, a null close) are expected from
  real vendors. Those rows are split off to quarantine with a reason, and the run goes on.
* **Schema-level** violations (a missing column, the wrong dtype) mean the upstream
  contract changed. Those raise, because continuing would corrupt every downstream table.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

PRICE_TOLERANCE = 1e-6

price_bar_schema = pa.DataFrameSchema(
    {
        "date": pa.Column("datetime64[ns]", nullable=False),
        "ticker": pa.Column(str, pa.Check.str_matches(r"^[A-Z0-9.\-^=]{1,12}$"), nullable=False),
        "open": pa.Column(float, pa.Check.gt(0), nullable=False),
        "high": pa.Column(float, pa.Check.gt(0), nullable=False),
        "low": pa.Column(float, pa.Check.gt(0), nullable=False),
        "close": pa.Column(float, pa.Check.gt(0), nullable=False),
        "adj_close": pa.Column(float, pa.Check.gt(0), nullable=False),
        "volume": pa.Column(float, pa.Check.ge(0), nullable=False),
    },
    checks=[
        pa.Check(
            lambda df: (
                df["high"] * (1 + PRICE_TOLERANCE) >= df[["open", "close", "low"]].max(axis=1)
            ),
            name="high_is_max",
            error="high must be >= open, close and low",
        ),
        pa.Check(
            lambda df: df["low"] * (1 - PRICE_TOLERANCE) <= df[["open", "close"]].min(axis=1),
            name="low_is_min",
            error="low must be <= open and close",
        ),
        pa.Check(
            lambda df: df["date"] <= pd.Timestamp.now().normalize() + pd.Timedelta(days=1),
            name="not_in_future",
        ),
    ],
    unique=["date", "ticker"],
    strict="filter",
    coerce=True,
)


class ContractViolationError(RuntimeError):
    """The incoming frame does not match the expected shape at all."""


def split_valid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate ``df``. Returns ``(valid_rows, rejected_rows_with_reason)``.

    Raises ``ContractViolationError`` for failures that can't be pinned to a row.
    """
    df = df.reset_index(drop=True)
    try:
        return price_bar_schema.validate(df, lazy=True), df.iloc[0:0].assign(_reject_reason="")
    except SchemaErrors as exc:
        failures = exc.failure_cases

    structural = failures[failures["index"].isna()]
    if not structural.empty:
        summary = structural[["schema_context", "column", "check"]].drop_duplicates()
        raise ContractViolationError(f"price data broke its contract:\n{summary.to_string()}")

    failures = failures.assign(index=failures["index"].astype(int))
    # Row-wide checks are reported once per column; label them by check name only.
    is_column_check = failures["schema_context"].eq("Column")
    label = (
        failures["check"]
        .astype(str)
        .where(
            ~is_column_check, failures["column"].astype(str) + ":" + failures["check"].astype(str)
        )
    )
    reasons = label.groupby(failures["index"]).agg(lambda s: "; ".join(sorted(set(s))))

    rejected = df.loc[reasons.index].assign(_reject_reason=reasons.values)
    valid = price_bar_schema.validate(df.drop(index=reasons.index), lazy=True)
    return valid, rejected
