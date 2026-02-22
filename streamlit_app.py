import streamlit as st
import pandas as pd
import numpy as np
import os

from app import run_pipeline_for_sku, optimize_S

st.set_page_config(page_title="Akıllı Tedarik Sistemi", layout="wide")

st.title("📦 Akıllı Tedarik ve Stok Optimizasyon Sistemi")
st.markdown("Monte Carlo Tabanlı (s,S) Envanter Optimizasyonu")

uploaded_file = st.file_uploader(
    "Talep Verisi Yükle (CSV veya Excel)",
    type=["csv", "xlsx"]
)

def load_uploaded_file(uploaded_file):
    file_ext = os.path.splitext(uploaded_file.name)[1]

    if file_ext == ".csv":
        df = pd.read_csv(uploaded_file)
    elif file_ext == ".xlsx":
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Desteklenmeyen dosya formatı")
        return None

    df.columns = df.columns.str.lower()

    required_cols = {"date", "sku", "demand"}
    if not required_cols.issubset(set(df.columns)):
        st.error("Dosya şu kolonları içermeli: date, sku, demand")
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sku", "date"])

    return df


if uploaded_file:

    df = load_uploaded_file(uploaded_file)

    if df is not None:

        st.success("Dosya başarıyla yüklendi")

        lead_time = st.slider("Lead Time (Hafta)", 1, 8, 2)
        review_period = st.slider("Review Period (Hafta)", 1, 8, 4)
        service_level = st.slider("Hedef Servis Seviyesi", 0.80, 0.99, 0.95)
        current_stock = st.number_input("Mevcut Stok", min_value=0, value=5)

        if st.button("Optimizasyonu Çalıştır"):

            results = []

            for sku in df["sku"].unique():

                df_sku = df[df["sku"] == sku].copy()
                forecast = run_pipeline_for_sku(df_sku)

                if forecast is None:
                    continue

                result = optimize_S(
                    forecast=forecast,
                    lead_time=lead_time,
                    review_period=review_period,
                    target_service_level=service_level,
                    current_stock=current_stock
                )

                results.append({
                    "SKU": sku,
                    "s (ROP)": result["s"],
                    "Optimize S": result["optimized_S"],
                    "Sipariş": result["order_quantity"],
                    "Gerçekleşen Stok-Out": result["final_stockout"]
                })

            result_df = pd.DataFrame(results)

            st.subheader("📊 Optimizasyon Sonuçları")
            st.dataframe(result_df)

            st.subheader("📈 Sipariş Miktarları")
            st.bar_chart(result_df.set_index("SKU")["Sipariş"])