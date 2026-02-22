import pandas as pd
import os
import numpy as np
from scipy.stats import norm

from core.adida import ADIDA
from core.sba import SBA
from core.tsb import TSB
from core.catboost_model import CatBoostModel


# -----------------------------
# DOSYA OKUMA
# -----------------------------
def load_data(file_path):
    ext = os.path.splitext(file_path)[1]

    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Desteklenmeyen dosya formatı")

    df.columns = df.columns.str.lower()

    required_cols = {"date", "sku", "demand"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError("Dosya şu kolonları içermeli: date, sku, demand")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sku", "date"])

    return df


# -----------------------------
# GELECEK TAHMİN
# -----------------------------
def forecast_future(model, last_row, steps=8):
    future_predictions = []
    current_row = last_row.copy()

    for _ in range(steps):
        pred = model.predict(current_row)[0]
        future_predictions.append(float(pred))

        current_row["lag3"] = current_row["lag2"]
        current_row["lag2"] = current_row["lag1"]
        current_row["lag1"] = pred

    return future_predictions


# -----------------------------
# STOK-OUT SİMÜLASYONU
# -----------------------------
def simulate_stock_out(forecast, stock_level, lead_time, simulations=4000):

    forecast_array = np.array(forecast)
    mean = np.mean(forecast_array)
    std = np.std(forecast_array)

    stock_out_count = 0

    for _ in range(simulations):
        simulated_demand = np.random.normal(mean, std, lead_time)
        total_demand = np.sum(simulated_demand)

        if total_demand > stock_level:
            stock_out_count += 1

    return stock_out_count / simulations


# -----------------------------
# S OPTİMİZASYONU
# -----------------------------
def optimize_S(
    forecast,
    lead_time,
    review_period,
    target_service_level,
    current_stock
):

    mean_demand = np.mean(forecast)
    std_demand = np.std(forecast)

    z_value = norm.ppf(target_service_level)

    safety_stock = z_value * std_demand * np.sqrt(lead_time)

    s = (mean_demand * lead_time) + safety_stock

    # Başlangıç S
    S = (mean_demand * (lead_time + review_period)) + safety_stock

    step = mean_demand  # ayarlama adımı

    for _ in range(50):

        order_qty = max(S - current_stock, 0)
        effective_stock = current_stock + order_qty

        stock_out_prob = simulate_stock_out(
            forecast,
            effective_stock,
            lead_time
        )

        target_stockout = 1 - target_service_level

        if abs(stock_out_prob - target_stockout) < 0.01:
            break

        if stock_out_prob > target_stockout:
            S += step
        else:
            S -= step

        if S < s:
            S = s

    final_order = max(S - current_stock, 0)

    return {
        "s": round(s, 2),
        "optimized_S": round(S, 2),
        "order_quantity": round(final_order, 2),
        "final_stockout": round(stock_out_prob, 3)
    }


# -----------------------------
# SKU PIPELINE
# -----------------------------
def run_pipeline_for_sku(df_sku):

    series = df_sku["demand"].reset_index(drop=True)

    adida_model = ADIDA(aggregation_window=4)
    adida_output = adida_model.aggregate(series)

    sba_model = SBA(alpha=0.1)
    sba_output = sba_model.fit(adida_output)

    tsb_model = TSB(alpha=0.1, beta=0.1)
    tsb_output = tsb_model.fit(adida_output)

    df_features = pd.DataFrame({
        "demand": series,
        "lag1": series.shift(1),
        "lag2": series.shift(2),
        "lag3": series.shift(3),
        "adida": adida_output,
        "sba": sba_output,
        "tsb": tsb_output
    })

    df_features = df_features.dropna()

    if df_features.empty:
        return None

    X = df_features.drop("demand", axis=1)
    y = df_features["demand"]

    model = CatBoostModel()
    model.train(X, y)

    last_row = X.iloc[-1:]

    future = forecast_future(model, last_row, steps=8)

    return future


# -----------------------------
# ANA PROGRAM
# -----------------------------
if __name__ == "__main__":

    file_path = "data/demand.csv"
    df = load_data(file_path)

    sku_list = df["sku"].unique()

    print("\nMONTE CARLO TABANLI (s,S) OPTİMİZASYON AKTİF:\n")

    for sku in sku_list:

        df_sku = df[df["sku"] == sku].copy()
        forecast = run_pipeline_for_sku(df_sku)

        if forecast is None:
            print(f"{sku} için yeterli veri yok.")
            continue

        result = optimize_S(
            forecast=forecast,
            lead_time=2,
            review_period=4,
            target_service_level=0.95,
            current_stock=5
        )

        print(f"SKU: {sku}")
        print("8 Haftalık Tahmin:", forecast)
        print("s (Reorder Point):", result["s"])
        print("Optimize Edilmiş S:", result["optimized_S"])
        print("Önerilen Sipariş:", result["order_quantity"])
        print("Gerçekleşen Stok-Out:", result["final_stockout"])
        print("-" * 60)