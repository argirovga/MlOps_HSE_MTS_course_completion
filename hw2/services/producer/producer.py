import os, time, json
import pandas as pd
from kafka import KafkaProducer

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC = os.getenv("KAFKA_OUT_TOPIC", "transactions")
CSV_PATH = os.getenv("TEST_CSV", "/data/test.csv")

KEY_ID_FIELDS = ["transaction_id", "id"]

def _row_transaction_id(row: dict, i: int) -> str:
    for k in KEY_ID_FIELDS:
        if k in row:
            return str(row[k])
    return str(i)

def main():
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda v: v.encode("utf-8"),
        retries=10,
        linger_ms=20,
    )

    df = pd.read_csv(CSV_PATH)
    records = df.to_dict(orient="records")

    for i, rec in enumerate(records):
        tid = _row_transaction_id(rec, i)
        producer.send(TOPIC, key=tid, value=rec)
    producer.flush()
    print(f"Produced {len(records)} messages to topic '{TOPIC}'.")

if __name__ == "__main__":
    # таймаут чтобы кафка запустилась внутри докера
    time.sleep(int(os.getenv("STARTUP_DELAY_SEC", "8")))
    main()