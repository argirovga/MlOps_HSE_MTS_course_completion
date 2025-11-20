# HW's by Argirov Georgy

В данном репозитории можно найти все дз выполненые мною на курсе MLops

---

### Table of Contents

- [HW 1](#hw-1)
- [HW 2](#hw-2)
- [HW 3](#hw-3)

# HW 1

## 📂 Структура дз

```bash
hw1/
│
├── data/
│   ├── logs/                   # логи работы
│   ├── input/                  # входные CSV-файлы для скоринга
│   ├── output/                 # результаты предсказаний
│   └── model_weights/          # веса модели
│       └──best_hist_gb_model.pkl
├── model_training/
│   └── simple_training.ipynb   # быстрые экспериментами  
├── src/
│   └── model.py                # обертка над моделью
├── app.py                      # основной сервис
├── Dockerfile                  # docker-образ
└── requirements.txt            # реки
```

---

## Запуск

### 1. Подготовка окружения

Перейдите в директорию дз1:

```bash
cd hw1
```

Установите зависимости:

```bash
pip install -r requirements.txt
```

---

### 2. Запуск через Docker

Соберите docker-образ:

```bash
docker build -t hw1-ml-service .
```

Запустите контейнер:

```bash
docker run -d \
  -v $(pwd)/data/input:/data/input \
  -v $(pwd)/data/output:/data/output \
  -v $(pwd)/data/logs:/data/logs \
  -v $(pwd)/data/model_weights:/data/model_weights \
  --name hw1_service \
  hw1-ml-service
```

Сервис начнет мониторить папку `data/input` на появление новых файлов.

---

### 3. Использование

Чтобы запустить скоринг, просто положите CSV-файл в директорию:

```
hw1/data/input/
```

Сервис автоматически:
- обнаружит новый файл,
- прогонит его через модель,
- сохранит результат в `hw1/data/output/`  
  под именем  
  `predictions_<timestamp>_<filename>.csv`.

---

### 4. Логи

Все логи работы сервиса сохраняются в:

```
hw1/data/logs/service.log
```

Посмотреть в реальном времени:

```bash
tail -f hw1/data/logs/service.log
```

---

### 5. Формат входного CSV

Ожидаемые колонки:

```
amount, lat, lon, population_city, cat_id, us_state
```

---

### 6. Остановка контейнера

```bash
docker stop hw1_service
docker rm hw1_service
```

---

### 7. Альтернатива — запуск без Docker

```bash
cd hw1
python app.py
```

После запуска можно также просто добавлять CSV-файлы в `data/input`, и сервис будет их автоматически обрабатывать.

---

# HW 2

## CТЕК
- Redpanda (Kafka API, без Zookeeper как в примере из семерской репы)
- PostgreSQL 16
- Python-сервисы: producer -> scorer -> sink
- Streamlit UI

## СТРУКТУРА РЕПОЗИТОРИЯ

```bash
├─ data/
│  ├─ logs/                       # (опционально) локальные логи сервисов
│  ├─ model_weights/
│  │  └─ best_rf_model.pkl        # обученная модель из HW1
│  └─ test.csv                    # тестовый датасет для стрима
├─ requirements/
│  ├─ requirements_producer.txt
│  ├─ requirements_scorer.txt
│  ├─ requirements_sink.txt
│  └─ requirements_ui.txt
├─ services/
│  ├─ producer/
│  │  └─ producer.py              # читает test.csv -> topic `transactions`
│  ├─ scorer/
│  │  └─ main.py                  # читает `transactions` -> скорит -> пишет `scores`
│  └─ sink/
│     └─ main.py                  # читает `scores` -> пишет в Postgres
├─ src/
│  ├─ __init__.py
│  ├─ models.py                   # ModelWrapper + ColumnSelector
│  └─ model.py                    # shim для совместимости с пиклом HW1
├─ ui/
│  └─ app.py                      # Streamlit (кнопка «Посмотреть результаты»)
├─ Dockerfile.producer
├─ Dockerfile.scorer
├─ Dockerfile.sink
├─ Dockerfile.ui
├─ docker-compose.yml
└─ README.md
```

## БЫСТРЫЙ СТАРТ (С НУЛЯ)
1) Поднять стек
  ```bash
  docker compose down -v  # если до этого уже запускали что-то
  docker compose up --build -d
  docker compose ps   # убедиться, что все Up (продьюсер может быть exited)
  ```

2) Отправить данные в Kafka
  ```bash
  docker compose run --rm producer
  ```
  Producer читает ./data/test.csv и отправляет строки в topic transactions.

3) Проверить, что записи попали в БД
  ```bash
  docker exec -it db psql -U mlops -d mlops -c "
  SELECT transaction_id, score, fraud_flag, scored_at
  FROM scores ORDER BY scored_at DESC LIMIT 10;"
  ```

4) Открыть UI
   Браузер: http://localhost:8501 -> кнопка «Посмотреть результаты»
   - слева: последние 10 транзакций с fraud_flag = 1
   - справа: гистограмма скоров последних 100 транзакций
   
   **важная пометка: в тестовой выборке из кагла, нет никаких транзакшен айди, поэтому они будут нан если вы их не добавите при проверке моего дз**

## ПОВТОРНЫЙ ПРОГОН
- Изменил/добавил строки в data/test.csv -> переотправить:
  docker compose run --rm producer
- Обнови страницу UI.

## НАСТРОЙКИ (env в docker-compose.yml)
- Порог фрода:
  scorer -> THRESHOLD=0.5
- Топики:
  вход: transactions
  выход: scores
- Модель:
  scorer -> MODEL_PATH=/data/model_weights/best_rf_model.pkl

## КАК ЭТО РАБОТАЕТ (КОРОТКО)
1. producer читает data/test.csv, добавляет transaction_id если его нет (по номеру строки), публикует JSON в Kafka topic transactions.
2. scorer читает сообщения, делает препроцессинг как в HW1 (через pipeline), считает score (вероятность), ставит fraud_flag по порогу и публикует результат в topic scores.
3. sink читает scores и пишет в Postgres таблицу scores (создаёт её при необходимости).
4. ui показывает результаты из БД.


# HW 3

## CТЕК
- Redpanda (Kafka API, без Zookeeper)
- ClickHouse 23.8
- Python 3.11 (отдельный producer: CSV → Kafka)
- Docker Compose

## СТРУКТУРА РЕПОЗИТОРИЯ

```bash
.
├─ data/
│  └─ raw_data/
│     └─ train.csv
├─ results/
│  └─ max_category_per_state.csv # сюда кладем результат п.3
├─ services/
│  └─ producer.py # п.1
├─ clickhouse/
│  └─ sql/
│     ├─ 01_ddl.sql # п.2 (DDL Kafka->MergeTree+MV)
│     ├─ 02_query_max_category.sql # п.3 (запрос на CSV)
│     └─ 03_optimized_ddl.sql # п.4 (оптимизация)
├─ Dockerfile.producer_hw3
└─ docker-compose.yml
```

## БЫСТРЫЙ СТАРТ (С НУЛЯ БЕЗ оптимизации)

1.  Чистый старт (сотрёт данные ClickHouse)

```bash
docker compose down -v
```

2. Запустить брокер и ClickHouse (ждём healthcheck)

```bash
docker compose up -d kafka clickhouse
```

Проверяем доступность ClickHouse:
```bash
curl http://localhost:8123/ping
```

3. Применить DDL (Kafka Engine → MergeTree + MV)

```bash
docker compose exec -T clickhouse \
  clickhouse-client --multiquery < clickhouse/sql/01_ddl.sql
```

Проверка:
```bash
docker compose exec clickhouse \
  clickhouse-client -q "SHOW TABLES FROM hw3"
```

Должны появится три таблицы

4. Отправить данные из CSV в Kafka topic `transactions_raw`

```bash
docker compose run --rm producer
```

Проверка:
```bash
docker compose exec clickhouse \
  clickhouse-client -q "SELECT count() FROM hw3.transactions_mt"
```

Должно быть больше 0

5. Выгрузка результатов в CSV
```bash
mkdir -p results

docker compose exec -T clickhouse \
  clickhouse-client \
  --format CSVWithNames \
  --query="$(cat clickhouse/sql/02_query_max_category.sql)" \
  > results/max_category_per_state.csv
```

## БЫСТРЫЙ СТАРТ (С НУЛЯ С оптимизацией)

1.  Чистый старт (сотрёт данные ClickHouse)

```bash
docker compose down -v
```

2. Запустить брокер и ClickHouse (ждём healthcheck)

```bash
docker compose up -d kafka clickhouse
```

Проверяем доступность ClickHouse:
```bash
curl http://localhost:8123/ping
```

3. Применить DDL (Kafka Engine → MergeTree + MV)

```bash
docker compose exec -T clickhouse \
  clickhouse-client --multiquery < clickhouse/sql/01_ddl.sql
```

Проверка:
```bash
docker compose exec clickhouse \
  clickhouse-client -q "SHOW TABLES FROM hw3"
```

Должны появится три таблицы

4. Отправить данные из CSV в Kafka topic `transactions_raw`

```bash
docker compose run --rm producer
```

Проверка:
```bash
docker compose exec clickhouse \
  clickhouse-client -q "SELECT count() FROM hw3.transactions_mt"
```

Должно быть больше 0

5. ВКЛЮЧАЕМ ОПТИМИЗАЦИЮ

```bash
docker compose exec -T clickhouse \
  clickhouse-client --multiquery < clickhouse/sql/03_optimized_ddl.sql
```

Проверка:
```bash
docker compose exec clickhouse \
  clickhouse-client -q "SHOW TABLES FROM hw3"
```
Должны были появится новые таблицы

6. Подгружаем старые данные (единожды):

```bash
docker compose exec clickhouse \
  clickhouse-client -q "
    INSERT INTO hw3.max_txn_by_state_agg
    SELECT
        us_state,
        maxState(amount),
        argMaxState(cat_id, amount)
    FROM hw3.transactions_mt
    GROUP BY us_state
  "
```

7. Выгрузка результатов в CSV
```bash
mkdir -p results

docker compose exec -T clickhouse \
  clickhouse-client \
  --format CSVWithNames \
  -q "
    SELECT
        us_state,
        argMaxMerge(max_cat_state) AS max_category,
        maxMerge(max_amount_state) AS max_amount
    FROM hw3.max_txn_by_state_agg
    GROUP BY us_state
    ORDER BY us_state
  " \
  > results/max_category_per_state.csv
```

## НАСТРОЙКИ (env в docker-compose.yml)

- Producer:
	- `KAFKA_BOOTSTRAP_SERVERS=kafka:9092`
	- `KAFKA_TOPIC=transactions_raw`
	- `CSV_PATH=/data/raw_data/test.csv`
	- `STARTUP_DELAY_SEC=3`
- ClickHouse:
	- Порты: `HTTP 8123`, `Native 9000`
	- SQL: `clickhouse/sql/*.sql` монтируются в `/docker-entrypoint-initdb.d`
- Redpanda (Kafka API):
	- `--advertise-kafka-addr=PLAINTEXT://kafka:9092`
	- `redpanda.auto_create_topics_enabled=true` (включено)

## КАК ЭТО РАБОТАЕТ (КОРОТКО)

1. `services/producer.py` читает `data/raw_data/train.csv`, формирует JSON-строки, добавляет transaction_id (если нет — по номеру строки), публикует в Kafka topic transactions_raw.
2. В ClickHouse создаётся `hw3.transactions_kafka` (движок Kafka, формат JSONEachRow), материализованная вьюха mv_to_mt переливает поток
