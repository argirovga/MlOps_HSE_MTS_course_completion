CREATE TABLE IF NOT EXISTS hw3.max_txn_by_state_agg
(
    us_state           String,
    max_amount_state   AggregateFunction(max, Float64),
    max_cat_state      AggregateFunction(argMax, Int32, Float64)
)
ENGINE = AggregatingMergeTree
ORDER BY us_state;

CREATE MATERIALIZED VIEW IF NOT EXISTS hw3.mv_max_txn_by_state
TO hw3.max_txn_by_state_agg
AS
SELECT
    us_state,
    maxState(amount)              AS max_amount_state,
    argMaxState(cat_id, amount)   AS max_cat_state
FROM hw3.transactions_mt
GROUP BY us_state;