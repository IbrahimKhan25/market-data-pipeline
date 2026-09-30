{# Annualised volatility from the mean of squared daily log returns over a window frame. #}
{% macro annualised_vol(squared_expr, frame) -%}
    sqrt({{ var('trading_days') }} * avg({{ squared_expr }}) over (partition by ticker order by date {{ frame }}))
{%- endmacro %}
