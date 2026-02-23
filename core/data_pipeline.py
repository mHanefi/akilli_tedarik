import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    df.columns = df.columns.str.lower()
    df = df.dropna(subset=["date", "demand"])
    df = df[df["demand"] >= 0]
    df["date"] = pd.to_datetime(df["date"])

    # KRİTİK DÜZELTME: mevcut_stok kolonunun kaybolmaması için listeye eklendi
    agg_funcs = {"demand": "sum"}
    for col in ["lead_time", "parca_ailesi", "arac_modeli", "birim_fiyat", "lot_size", "mevcut_stok"]:
        if col in df.columns:
            agg_funcs[col] = "first"
            
    df = df.groupby(["date", "sku"], as_index=False).agg(agg_funcs)

    processed_dfs = []
    for sku in df["sku"].unique():
        sku_df = df[df["sku"] == sku].copy()
        full_date_range = pd.date_range(start=sku_df["date"].min(), end=sku_df["date"].max(), freq="W") 
        
        sku_df = sku_df.set_index("date")
        sku_df = sku_df.reindex(full_date_range)
        sku_df["demand"] = sku_df["demand"].fillna(0)
        sku_df["sku"] = sku
        
        # Tüm statik bilgileri ve STOK bilgisini boş haftalara yay
        for col in ["lead_time", "parca_ailesi", "arac_modeli", "birim_fiyat", "lot_size", "mevcut_stok"]:
            if col in sku_df.columns:
                sku_df[col] = sku_df[col].ffill().bfill()
            
        sku_df = sku_df.rename_axis("date").reset_index()
        processed_dfs.append(sku_df)

    final_df = pd.concat(processed_dfs, ignore_index=True)
    return final_df