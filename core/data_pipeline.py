import pandas as pd
import numpy as np


def load_data(file):
    df = pd.read_excel(file)

    # Zorunlu kolon kontrolü
    required_columns = ["Date", "SKU", "Demand"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"{col} sütunu eksik!")

    # Veri tip dönüşümleri
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce")

    # Eksik ve hatalı değer temizleme
    df = df.dropna(subset=["Date", "Demand"])
    df = df[df["Demand"] >= 0]

    # Eğer aynı gün aynı SKU birden fazla varsa topluyoruz
    df = (
        df.groupby(["Date", "SKU"], as_index=False)
        .agg({"Demand": "sum"})
    )

    return df


def preprocess_data(df):
    df = df.sort_values(["SKU", "Date"])

    result = []
    sku_summary_list = []

    for sku in df["SKU"].unique():
        sku_df = df[df["SKU"] == sku].copy()

        # Günlük tam tarih aralığı oluştur
        full_date_range = pd.date_range(
            start=sku_df["Date"].min(),
            end=sku_df["Date"].max(),
            freq="D"
        )

        sku_df = sku_df.set_index("Date")
        sku_df = sku_df.reindex(full_date_range, fill_value=0)
        sku_df["SKU"] = sku
        sku_df = sku_df.rename_axis("Date").reset_index()

        # --- İSTATİSTİK ÖZETLER ---
        total_days = len(sku_df)
        zero_days = (sku_df["Demand"] == 0).sum()
        zero_ratio = zero_days / total_days

        mean_demand = sku_df["Demand"].mean()
        std_demand = sku_df["Demand"].std()
        var_demand = sku_df["Demand"].var()

        sku_summary_list.append({
            "SKU": sku,
            "Total_Days": total_days,
            "Zero_Days": zero_days,
            "Zero_Ratio": zero_ratio,
            "Mean_Demand": mean_demand,
            "Std_Demand": std_demand,
            "Variance_Demand": var_demand
        })

        result.append(sku_df)

    final_df = pd.concat(result, ignore_index=True)
    sku_summary_df = pd.DataFrame(sku_summary_list)

    return final_df, sku_summary_df