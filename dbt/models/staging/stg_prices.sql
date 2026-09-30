-- Deduplicates the append-only bronze log: the most recent ingest wins for each (ticker, date).
-- This is what makes re-running ingestion idempotent.
with source as (
    select * from {{ source('bronze', 'prices') }}
)

select
    ticker::varchar                 as ticker,
    date::date                      as date,
    open::double                    as open,
    high::double                    as high,
    low::double                     as low,
    close::double                   as close,
    adj_close::double               as adj_close,
    volume::bigint                  as volume,
    ingested_at::timestamp          as ingested_at,
    source::varchar                 as source,
    run_id::varchar                 as run_id
from source
qualify row_number() over (partition by ticker, date order by ingested_at desc, run_id desc) = 1
