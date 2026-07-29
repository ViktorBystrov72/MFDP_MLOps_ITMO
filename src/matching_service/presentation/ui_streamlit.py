from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Matching Exposition ↔ Deals", layout="wide")
st.title("Матчинг экспозиции и сделок (Neolithic)")

st.markdown(
    """
Пилот: **Тюмень**. Сервис сопоставляет объявления экспозиции со сделками.
Данные — продакшен-БД Neolithic; обучение/инференс — серверы компании с GPU.
"""
)

# В Docker Compose API — сервис `api`; локально без Docker — http://127.0.0.1:8000
_default_api = os.getenv("MATCHING_API_URL", "http://api:8000")
api = st.text_input("API URL", _default_api)

uploaded = st.file_uploader("CSV с признаками пар", type=["csv"])
threshold = st.slider("Порог матча", 0.0, 1.0, 0.5, 0.01)

if uploaded and st.button("Прогнать batch"):
    df = pd.read_csv(uploaded)
    pairs = df.to_dict(orient="records")
    r = requests.post(
        f"{api}/match/batch", json={"pairs": pairs, "threshold": threshold}, timeout=120
    )
    r.raise_for_status()
    pred = pd.DataFrame(r.json())
    st.dataframe(pd.concat([df.reset_index(drop=True), pred], axis=1))
    st.metric("Доля матчей", f"{pred['is_match'].mean():.1%}")
    st.metric("На проверку", int(pred["needs_review"].sum()))
else:
    st.info(
        "Загрузите CSV с колонками: area_diff, floor_diff, same_flat_number, "
        "same_rooms, same_building, price_rel_diff (опционально fuzzy_flat). "
        "Пример: artifacts/datasets/ui_demo_pairs.csv или tyumen_pairs_sample.csv. "
        "В Docker поле API URL должно быть http://api:8000"
    )
