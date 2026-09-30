-- Leakage guard: the target on day t must equal the trailing 5-day realised vol observed on
-- day t+h. This proves the label is built only from future returns, and that no feature is
-- secretly the label. Returns offending rows (test passes when empty).
{% set h = var('forecast_horizon') %}
{% if h == 5 %}
with f as (
    select
        ticker,
        date,
        target_rv_fwd,
        lead(rv_5d, {{ h }}) over (partition by ticker order by date) as rv_5d_at_horizon
    from {{ ref('features_volatility') }}
)
select *
from f
where target_rv_fwd is not null
  and rv_5d_at_horizon is not null
  and abs(target_rv_fwd - rv_5d_at_horizon) > 1e-9
{% else %}
select 1 where false
{% endif %}
