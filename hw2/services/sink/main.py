# services/sink/main.py
import os, time, json, logging
from kafka import KafkaConsumer
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sink")

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC = os.getenv("KAFKA_SCORES_TOPIC", "scores")
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "scores_sink")

PG_USER = os.getenv("POSTGRES_USER", "mlops")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "mlops")
PG_DB   = os.getenv("POSTGRES_DB", "mlops")
PG_HOST = os.getenv("POSTGRES_HOST", "db")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")

TABLE = os.getenv("PG_TABLE", "scores")

DDL_TMPL = """
CREATE TABLE IF NOT EXISTS {table} (
  id SERIAL PRIMARY KEY,
  transaction_id TEXT NOT NULL,
  score DOUBLE PRECISION NOT NULL,
  fraud_flag BOOLEAN NOT NULL,
  scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_{table}_scored_at ON {table}(scored_at DESC);
CREATE INDEX IF NOT EXISTS idx_{table}_fraud ON {table}(fraud_flag);
"""

INSERT_SQL_TMPL = "INSERT INTO {table} (transaction_id, score, fraud_flag) VALUES (:tid, :score, :flag);"

def ensure_table(engine, table: str):
    ddl = DDL_TMPL.format(table=table)
    with engine.begin() as conn:
        for stmt in ddl.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.exec_driver_sql(s + ";")

def main():
    time.sleep(int(os.getenv("STARTUP_DELAY_SEC", "3")))

    engine = create_engine(f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}")

    ensure_table(engine, TABLE)
    log.info("Ensured table '%s' exists.", TABLE)

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        group_id=GROUP_ID,
        enable_auto_commit=True,
        auto_offset_reset="earliest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    log.info("Sink started. Reading '%s' and writing to %s.%s", TOPIC, PG_DB, TABLE)

    insert_sql = text(INSERT_SQL_TMPL.format(table=TABLE))

    for msg in consumer:
        rec = msg.value
        try:
            with engine.begin() as conn:
                conn.execute(insert_sql, {
                    "tid": str(rec["transaction_id"]),
                    "score": float(rec["score"]),
                    "flag": bool(rec["fraud_flag"]),
                })
            log.info("Inserted tid=%s score=%.5f flag=%s", rec["transaction_id"], rec["score"], rec["fraud_flag"])
        except Exception as e:
            log.exception("DB insert failed for %s: %s", rec, e)

if __name__ == "__main__":
    main()