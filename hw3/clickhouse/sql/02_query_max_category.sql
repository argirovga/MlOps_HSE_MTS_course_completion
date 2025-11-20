SELECT
    us_state,
    argMax(cat_id, amount) AS max_category,
    max(amount) AS max_amount
FROM hw3.transactions_mt
GROUP BY us_state
ORDER BY us_state;