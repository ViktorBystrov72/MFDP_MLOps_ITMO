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

_default_api = os.getenv("MATCHING_API_URL", "http://api:8000")
api = st.sidebar.text_input("API URL", _default_api)
threshold = st.sidebar.slider("Порог матча (precision-first)", 0.0, 1.0, 0.95, 0.01)
margin = st.sidebar.slider("Минимальный отрыв top-1", 0.0, 0.3, 0.05, 0.01)
use_embeddings = st.sidebar.checkbox("Embeddings (BGE)", value=True)
use_reranker = st.sidebar.checkbox("BGE reranker", value=True)
use_llm = st.sidebar.checkbox("LLM voting на abstain", value=True)
llm_voting = st.sidebar.checkbox("LLM voting 2 из 3", value=True, disabled=not use_llm)

db_tab, csv_tab = st.tabs(["Матчинг сделок из БД", "CSV готовых кандидатов"])

with db_tab:
    st.caption(
        "API сам загружает сделку, формирует кандидатов в том же корпусе и временном окне, "
        "рассчитывает PD-aware признаки и применяет grouped cascade."
    )
    deal_ids_text = st.text_area(
        "ID сделок (по одному UUID на строку)",
        placeholder="bd6d6d0d-a6b1-4f3f-b100-afc805fb4575",
    )
    max_candidates = st.slider("Максимум кандидатов на сделку", 5, 100, 30, 5)
    if st.button("Найти объявления и сматчить", type="primary"):
        deal_ids = [value.strip() for value in deal_ids_text.splitlines() if value.strip()]
        if not deal_ids:
            st.warning("Укажите хотя бы один deal_id")
        else:
            payload = {
                "deal_ids": deal_ids,
                "threshold": threshold,
                "margin_threshold": margin,
                "max_candidates_per_deal": max_candidates,
                "use_embeddings": use_embeddings,
                "use_reranker": use_reranker,
                "use_llm": use_llm,
                "llm_voting": llm_voting,
            }
            try:
                response = requests.post(
                    f"{api}/match/deals",
                    json=payload,
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as exc:
                st.error(f"API недоступен: {exc}")
            else:
                first, second, third = st.columns(3)
                first.metric("Матчей", result["matched"])
                second.metric("На проверку", result["review"])
                third.metric("Сделок с кандидатами", result["deals_with_candidates"])
                rows = pd.DataFrame(result["items"])
                if rows.empty:
                    st.info("Для указанных сделок кандидаты в временном окне не найдены")
                else:
                    st.dataframe(
                        rows.sort_values(["deal_id", "candidate_rank"]),
                        use_container_width=True,
                    )
                    st.download_button(
                        "Скачать результат CSV",
                        rows.to_csv(index=False).encode("utf-8"),
                        "matching_result.csv",
                        "text/csv",
                    )

with csv_tab:
    uploaded = st.file_uploader("CSV с готовыми парами и признаками", type=["csv"])
    use_cascade = st.checkbox("Grouped cascade", value=True)
    if uploaded and st.button("Прогнать CSV batch"):
        frame = pd.read_csv(uploaded)
        payload = {
            "pairs": frame.to_dict(orient="records"),
            "threshold": threshold,
            "margin_threshold": margin,
            "use_cascade": use_cascade,
            "use_embeddings": use_embeddings,
            "use_reranker": use_reranker,
            "use_llm": use_llm,
            "llm_voting": llm_voting,
        }
        try:
            response = requests.post(
                f"{api}/match/batch",
                json=payload,
                timeout=300,
            )
            response.raise_for_status()
            prediction = pd.DataFrame(response.json())
        except requests.RequestException as exc:
            st.error(f"API недоступен: {exc}")
        else:
            st.dataframe(
                pd.concat([frame.reset_index(drop=True), prediction], axis=1),
                use_container_width=True,
            )
            first, second = st.columns(2)
            first.metric("Доля матчей", f"{prediction['is_match'].mean():.1%}")
            second.metric("На проверку", int(prediction["needs_review"].sum()))
    elif not uploaded:
        st.info(
            "Пример: artifacts/datasets/tyumen_pd_aware_sample.csv. "
            "Для grouped cascade нужны deal_id и несколько flat_id-кандидатов на сделку."
        )
