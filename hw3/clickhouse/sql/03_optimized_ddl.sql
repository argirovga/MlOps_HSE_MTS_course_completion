-- На всякий случай дропаем старые MV и таблицу
DROP TABLE IF EXISTS hw3.mv_max_txn_by_state;
DROP TABLE IF EXISTS hw3.max_txn_by_state_agg;

-- Агрегирующая таблица: cat_id теперь String
CREATE TABLE IF NOT EXISTS hw3.max_txn_by_state_agg
(
    us_state         String,
    max_amount_state AggregateFunction(max, Float64),
    max_cat_state    AggregateFunction(argMax, String, Float64)
)
ENGINE = AggregatingMergeTree
ORDER BY us_state;

-- MV, слушающий transactions_mt и наполняющий агрегаты
CREATE MATERIALIZED VIEW IF NOT EXISTS hw3.mv_max_txn_by_state
TO hw3.max_txn_by_state_agg
AS
SELECT
    us_state,
    maxState(amount)            AS max_amount_state,
    argMaxState(cat_id, amount) AS max_cat_state
FROM hw3.transactions_mt
GROUP BY us_state;