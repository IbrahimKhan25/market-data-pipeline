-- Daily return facts per ticker. Uses adjusted close so dividends and splits don't show up as returns.
with prices as (
    select * from {{ ref('stg_prices') }}
),

lagged as (
    select
        *,
        lag(adj_close) over w as prev_adj_close,
        lag(close) over w     as prev_close
    from prices
    window w as (partition by ticker order by date)
)

select
    ticker,
    date,
    open,
    high,
    low,
    close,
    adj_close,
    volume,
    ln(adj_close / prev_adj_close)  as log_return,
    adj_close / prev_adj_close - 1  as simple_return,
    close - open                    as price_change,
    ln(high / low)                  as log_range,
    ln(open / prev_close)           as overnight_gap
from lagged
