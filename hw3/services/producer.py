import os
import time
import json
import math
from typing import Any, Dict

import pandas as pd
from kafka import KafkaProducer


BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "transactions_raw")
CSV_PATH = os.getenv("CSV_PATH", "/data/raw_data/train.csv")
STARTUP_DELAY_SEC = int(os.getenv("STARTUP_DELAY_SEC", "5"))

# какие поля можно использовать как ключ сообщения
KEY_ID_FIELDS = ["transaction_id", "id"]


def _is_empty(v: Any) -> bool:
    """Проверка 'пустоты' значения (None, NaN, пустая строка)."""
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return str(v).strip() == ""


def _row_transaction_id(row: Dict[str, Any], i: int) -> str:
    """Выбрать transaction_id из строки или сгенерировать по номеру."""
    for k in KEY_ID_FIELDS:
        v = row.get(k)
        if not _is_empty(v):
            return str(v)
    return str(i)


def main() -> None:
    print(f"[producer] Waiting {STARTUP_DELAY_SEC} seconds before start...")
    time.sleep(STARTUP_DELAY_SEC)

    print(f"[producer] Reading CSV from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)

    # заменяем NaN на None, чтобы корректно сериализовать в JSON
    df = df.where(pd.notnull(df), None)

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda v: v.encode("utf-8"),
        linger_ms=20,
        retries=10,
    )

    records = df.to_dict(orient="records")
    for i, rec in enumerate(records, start=1):
        tid = _row_transaction_id(rec, i)
        # гарантируем наличие поля transaction_id
        if "transaction_id" not in rec or _is_empty(rec["transaction_id"]):
            rec["transaction_id"] = tid

        producer.send(TOPIC, key=tid, value=rec)

    producer.flush()
    print(f"[producer] Produced {len(records)} messages to topic '{TOPIC}'")


if __name__ == "__main__":
    main()