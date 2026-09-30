-- Bars that survived contract validation must still be internally consistent after deduplication.
select *
from {{ ref('stg_prices') }}
where high < greatest(open, close) * (1 - 1e-6)
   or low  > least(open, close) * (1 + 1e-6)
