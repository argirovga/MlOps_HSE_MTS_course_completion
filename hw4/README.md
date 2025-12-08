# HW4 — dbt project on ClickHouse fraud dataset

Домашка по dbt (ClickHouse adapter). В папке `dbt/` лежит полноценный проект с источником `transactions_db.transactions`, витринами и тестами на надеюсь))) 10 баллов.

## Структура

```bash
dbt/
├── dbt_project.yml
├── packages.yml
├── macros/
│   └── amount_bucket.sql
├── models/
│   ├── sources/sources.yml
│   ├── staging/
│   │   ├── stg_transactions.sql
│   │   └── stg_transactions.yml
│   └── marts/
│       ├── mart_daily_state_metrics.sql
│       ├── mart_fraud_by_category.sql
│       ├── mart_fraud_by_state.sql
│       ├── mart_customer_risk_profile.sql
│       ├── mart_hourly_fraud_pattern.sql
│       ├── mart_merchant_analytics.sql
│       └── schema.yml
├── seeds/states.csv
├── tests/
│   ├── assert_no_negative_amounts.sql
│   ├── assert_fraud_rate_bounds.sql
│   └── unit/stg_transactions.yml
├── .sqlfluff
├── .pre-commit-config.yaml
└── Makefile
```

## Настройка

1. Установите зависимости (понадобится `dbt-clickhouse` и dbt-core 1.8+):
   ```bash
   pip install "dbt-core==1.8.4" dbt-clickhouse sqlfluff pre-commit
   ```
2. Добавьте профиль `hw4_dbt` в `~/.dbt/profiles.yml` или рядом с проектом:
   ```yaml
   hw4_dbt:
     target: dev
     outputs:
       dev:
         type: clickhouse
         host: localhost
         port: 8123
         schema: analytics
         user: default
         password: ""
         secure: false
         threads: 4
   ```

## Как запускать

```bash
cd dbt
dbt deps
dbt seed
dbt run
dbt test
dbt docs generate
```

SQL линтинг и авто-фикс:
```bash
cd dbt
sqlfluff lint .
sqlfluff fix .
```

Docs/DAG: `dbt docs serve` и открыть http://localhost:8080.

Источник данных: `data/raw_data/test.csv` уже положен как сид `dbt/seeds/transactions.csv` (по умолчанию база `analytics_transactions_db`). Запуск `dbt seed` создаст таблицу-источник для всего пайплайна.

### Быстрый старт через Docker

```bash
docker compose up -d clickhouse
# ждём healthcheck или проверяем вручную
curl -u hw4:hw4_pass http://localhost:8123/ping

cd dbt
dbt deps
dbt seed
dbt run
dbt test
```

Контейнеры/тома: `docker-compose.yml` создаёт сервис `hw4_clickhouse` с томом `clickhouse-data` (HTTP 8123, Native 9000). Дефолтные креды для dbt: `user=hw4`, `password=hw4_pass`, `host=localhost`, `port=8123`.

## Что реализовано

- Слои: sources в staging в marts.
- 6 витрин с метриками по датам, категориям, штатам, клиентам, часам и мерчантам.
- Кастомный макрос `amount_bucket`.
- Пакеты: dbt_utils, dbt_expectations (metaplane fork).
- Тесты: schema, singular, unit.
- Инструменты качества: sqlfluff, pre-commit, Makefile.
