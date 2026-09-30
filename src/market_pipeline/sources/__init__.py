from market_pipeline.sources.base import PRICE_COLUMNS, MarketDataSource
from market_pipeline.sources.synthetic import SyntheticSource
from market_pipeline.sources.yahoo import YahooSource


def get_source(name: str) -> MarketDataSource:
    sources: dict[str, type[YahooSource] | type[SyntheticSource]] = {
        "yahoo": YahooSource,
        "synthetic": SyntheticSource,
    }
    try:
        return sources[name]()
    except KeyError:
        raise ValueError(f"unknown source {name!r}; choose from {sorted(sources)}") from None


__all__ = ["PRICE_COLUMNS", "MarketDataSource", "SyntheticSource", "YahooSource", "get_source"]
