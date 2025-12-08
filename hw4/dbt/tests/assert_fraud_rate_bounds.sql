select *
from {{ ref('mart_fraud_by_category') }}
where fraud_rate > 1 or fraud_rate < 0
