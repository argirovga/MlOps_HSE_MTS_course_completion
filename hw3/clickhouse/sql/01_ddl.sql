-- Полностью пересоздаём базу для чистоты
DROP DATABASE IF EXISTS hw3;

CREATE DATABASE IF NOT EXISTS hw3;

-- 1) Kafka-таблица, читает из топика transactions_raw
CREATE TABLE hw3.transactions_kafka
(
    transaction_id String,
    us_state       String,
    cat_id         String,
    amount         Float64
)
ENGINE = Kafka
SETTINGS
    kafka_broker_list          = 'kafka:9092',
    kafka_topic_list           = 'transactions_raw',
    kafka_group_name           = 'ch_consumer_hw3',
    kafka_format               = 'JSONEachRow',
    kafka_num_consumers        = 1,
    kafka_skip_broken_messages = 1;

-- 2) Основная таблица для хранения транзакций
CREATE TABLE hw3.transactions_mt
(
    transaction_id String,
    us_state       String,
    cat_id         String,
    amount         Float64,
    event_date     Date DEFAULT today()
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_date)
ORDER BY (us_state, transaction_id)
SETTINGS index_granularity = 8192;

-- 3) MV, который САМ читает из Kafka и пишет в MergeTree
CREATE MATERIALIZED VIEW hw3.mv_kafka_to_mt
TO hw3.transactions_mt
AS
SELECT
    transaction_id,
    us_state,
    cat_id,
    amount,
    today() AS event_date
FROM hw3.transactions_kafka;