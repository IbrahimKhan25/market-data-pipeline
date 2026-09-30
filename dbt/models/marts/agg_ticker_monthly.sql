-- Monthly performance summary per ticker, for BI dashboards.
select
    ticker,
    date_trunc('month', date)::date                         as month,
    count(*)                                                as trading_days,
    exp(sum(log_return)) - 1                                as monthly_return,
    sqrt({{ var('trading_days') }}) * stddev_samp(log_return) as realised_vol,
    min(low)                                                as month_low,
    max(high)                                               as month_high,
    avg(volume)::bigint                                     as avg_daily_volume,
    sum(case when simple_return > 0 then 1 else 0 end)::double / count(*) as pct_up_days
from {{ ref('fct_daily_returns') }}
where log_return is not null
group by all
