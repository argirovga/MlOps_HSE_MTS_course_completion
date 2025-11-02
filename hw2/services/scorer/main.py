import os, json, time, logging
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from src.model import ModelWrapper, ColumnSelector

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("scorer")

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
IN_TOPIC = os.getenv("KAFKA_IN_TOPIC", "transactions")
OUT_TOPIC = os.getenv("KAFKA_OUT_TOPIC", "scores")
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "scorer")
MODEL_PATH = os.getenv("MODEL_PATH", "/data/model_weights/best_rf_model.pkl")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

ID_CANDIDATES = ["transaction_id", "id"]

def get_tid(rec: dict) -> str:
    for k in ID_CANDIDATES:
        if k in rec:
            return str(rec[k])
    return rec.get("uid") or rec.get("UUID") or "NA" # добавил NA так как в файле из соревы нет никакого транзакшен ид

def main():
    time.sleep(int(os.getenv("STARTUP_DELAY_SEC", "8")))

    consumer = KafkaConsumer(
        IN_TOPIC,
        bootstrap_servers=BOOTSTRAP,
        group_id=GROUP_ID,
        enable_auto_commit=True,
        auto_offset_reset="earliest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda v: v.encode("utf-8"),
        retries=10,
    )

    mw = ModelWrapper(MODEL_PATH, threshold=THRESHOLD)
    log.info("Scorer started. Listening to '%s'...", IN_TOPIC)

    for msg in consumer:
        rec = msg.value
        tid = get_tid(rec)
        df = pd.DataFrame([rec])

        try:
            mw.update_scoring_file(df)
            mw.preprocess()
            score = float(mw.predict_proba()[0])
            flag = int(score >= THRESHOLD)

            out = {"transaction_id": tid, "score": score, "fraud_flag": flag}
            producer.send(OUT_TOPIC, key=tid, value=out)
            log.info("Scored tid=%s score=%.5f flag=%d", tid, score, flag)
        except Exception as e:
            log.exception("Failed scoring tid=%s: %s", tid, e)

if __name__ == "__main__":
    main()