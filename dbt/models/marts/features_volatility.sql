{% set h = var('forecast_horizon') %}
{% set td = var('trading_days') %}
-- Model-ready feature table for volatility forecasting.
--
-- Every feature on row t uses only data up to and including t. The target is the realised
-- volatility over the NEXT `forecast_horizon` sessions (t+1 .. t+h). It stays NULL for the
-- most recent rows, where the future isn't known yet; those rows are what we score in production.
with returns as (
    select * from {{ ref('fct_daily_returns') }}
    where log_return is not null
),

per_ticker as (
    select
        ticker,
        date,
        log_return,
        abs(log_return)                                                                    as abs_return_1d,
        overnight_gap,
        {{ annualised_vol('log_return * log_return', 'rows between 4 preceding and current row') }}  as rv_5d,
        {{ annualised_vol('log_return * log_return', 'rows between 20 preceding and current row') }} as rv_21d,
        {{ annualised_vol('log_return * log_return', 'rows between 62 preceding and current row') }} as rv_63d,
        sqrt({{ td }} / (4 * ln(2)) * avg(log_range * log_range) over (
            partition by ticker order by date rows between 4 preceding and current row))   as parkinson_5d,
        sqrt({{ td }} / (4 * ln(2)) * avg(log_range * log_range) over (
            partition by ticker order by date rows between 20 preceding and current row))  as parkinson_21d,
        sum(log_return) over (
            partition by ticker order by date rows between 20 preceding and current row)   as momentum_21d,
        (ln(volume + 1) - avg(ln(volume + 1)) over w21)
            / nullif(stddev_samp(ln(volume + 1)) over w21, 0)                              as volume_z_21d,
        row_number() over (partition by ticker order by date)                              as history_len,

        -- Target: forward-looking window. Everything above this line is backward-looking.
        {{ annualised_vol('log_return * log_return', 'rows between 1 following and ' ~ h ~ ' following') }} as rv_fwd,
        count(*) over (
            partition by ticker order by date rows between 1 following and {{ h }} following) as fwd_obs
    from returns
    window w21 as (partition by ticker order by date rows between 20 preceding and current row)
)

select
    ticker,
    date,
    log_return,
    abs_return_1d,
    overnight_gap,
    rv_5d,
    rv_21d,
    rv_63d,
    parkinson_5d,
    parkinson_21d,
    momentum_21d,
    coalesce(volume_z_21d, 0)                          as volume_z_21d,
    avg(rv_21d) over (partition by date)               as market_rv_21d,
    rv_21d / nullif(avg(rv_21d) over (partition by date), 0) as relative_rv_21d,
    isodow(date)                                       as day_of_week,
    case when fwd_obs = {{ h }} then rv_fwd end        as target_rv_fwd,
    fwd_obs = {{ h }}                                  as is_labeled
from per_ticker
where history_len >= {{ var('min_history') }}
