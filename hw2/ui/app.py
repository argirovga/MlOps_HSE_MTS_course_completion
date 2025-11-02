import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

st.set_page_config(page_title="Fraud Scoring Dashboard", layout="wide")
st.title("Fraud Scoring Dashboard")

PG_USER = os.getenv("POSTGRES_USER", "mlops")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "mlops")
PG_DB   = os.getenv("POSTGRES_DB", "mlops")
PG_HOST = os.getenv("POSTGRES_HOST", "db")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")
TABLE   = os.getenv("PG_TABLE", "scores")

engine = create_engine(f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}")

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
  id SERIAL PRIMARY KEY,
  transaction_id TEXT NOT NULL,
  score DOUBLE PRECISION NOT NULL,
  fraud_flag BOOLEAN NOT NULL,
  scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_{TABLE}_scored_at ON {TABLE}(scored_at DESC);
CREATE INDEX IF NOT EXISTS idx_{TABLE}_fraud ON {TABLE}(fraud_flag);
"""

def ensure_table():
    with engine.begin() as conn:
        for stmt in DDL.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.exec_driver_sql(s + ";")

if st.button("Посмотреть результаты"):
    ensure_table()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Последние 10 fraud_flag = 1")
        q1 = f"""
        SELECT transaction_id, score, fraud_flag, scored_at
        FROM {TABLE}
        WHERE fraud_flag = true
        ORDER BY scored_at DESC
        LIMIT 10;
        """
        df1 = pd.read_sql(q1, engine)
        st.dataframe(df1, use_container_width=True)

    with col2:
        st.subheader("Гистограмма распределения скоров последних 100 транзакций")
        q2 = f"SELECT score FROM {TABLE} ORDER BY scored_at DESC LIMIT 100;"
        df2 = pd.read_sql(q2, engine)
        if not df2.empty:
            import numpy as np
            counts, bins = np.histogram(df2["score"].astype(float), bins=20, range=(0.0, 1.0))
            hist_df = pd.DataFrame({"bin_left": bins[:-1], "count": counts}).set_index("bin_left")
            st.bar_chart(hist_df)
        else:
            st.info("Пока нет данных в базе — отправьте сообщения продюсером и обновите.")