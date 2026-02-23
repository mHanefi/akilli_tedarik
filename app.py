import pandas as pd
import numpy as np
from scipy.stats import norm
import math
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from core.data_pipeline import load_and_preprocess_data
from core.adida import ADIDA
from core.sba import SBA
from core.tsb import TSB
from core.catboost_model import CatBoostModel

def build_global_dataset(df):
    all_features = []
    for sku in df["sku"].unique():
        sku_df = df[df["sku"] == sku].copy().reset_index(drop=True)
        demand_series = sku_df["demand"]
        
        adida_engine = ADIDA(aggregation_window="auto")
        adida_out = adida_engine.aggregate(demand_series)
        sba_engine = SBA(alpha=0.1)
        sba_out = sba_engine.fit(adida_out)
        tsb_engine = TSB(alpha=0.1, beta=0.1)
        tsb_out = tsb_engine.fit(adida_out)
        
        sku_df["lag1"] = demand_series.shift(1)
        sku_df["lag2"] = demand_series.shift(2)
        sku_df["lag3"] = demand_series.shift(3)
        sku_df["adida"] = adida_out
        sku_df["sba"] = sba_out
        sku_df["tsb_forecast"] = tsb_out["tsb_forecast"]
        sku_df["tsb_probability"] = tsb_out["tsb_probability"]
        sku_df["tsb_magnitude"] = tsb_out["tsb_magnitude"]
            
        all_features.append(sku_df)
        
    return pd.concat(all_features, ignore_index=True).dropna().reset_index(drop=True)

def train_and_validate_model(X, y, cat_features):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    eval_model = CatBoostModel(cat_features=cat_features)
    eval_model.train(X_train, y_train)
    preds = np.maximum(eval_model.predict(X_test), 0)
    
    # Sadece dürüst sapma metrikleri bırakıldı.
    metrics = {
        "RMSE": round(math.sqrt(mean_squared_error(y_test, preds)), 2),
        "MAE": round(mean_absolute_error(y_test, preds), 2)
    }
    
    final_model = CatBoostModel(cat_features=cat_features)
    final_model.train(X, y)
    
    return final_model, metrics

def forecast_future_for_sku(model, last_row, future_plan_df, sku_arac_modeli, steps=8):
    future_predictions = []
    current_row = last_row.copy()
    plan_filtered = future_plan_df[future_plan_df["arac_modeli"] == sku_arac_modeli].sort_values("date")
    planlanmis_uretimler = plan_filtered["planlanan_uretim"].values
    
    baseline_production = np.mean(planlanmis_uretimler) if len(planlanmis_uretimler) > 0 else 1.0
    if baseline_production == 0: baseline_production = 1.0

    for step in range(steps):
        base_pred = max(0.0, float(model.predict(current_row)[0]))
        if step < len(planlanmis_uretimler):
            katsayi = planlanmis_uretimler[step] / baseline_production
            final_pred = base_pred * katsayi
        else:
            final_pred = base_pred

        future_predictions.append(final_pred)
        current_row["lag3"], current_row["lag2"], current_row["lag1"] = current_row["lag2"], current_row["lag1"], final_pred

    return future_predictions

def simulate_stock_out(forecast, stock_level, lead_time, simulations=4000):
    """
    OPTİMİZASYON: Python 'for' döngüsü iptal edildi. 
    NumPy vektörizasyonu ile 4000 simülasyon tek matris işlemiyle milisaniyede hesaplanıyor.
    """
    forecast_array = np.array(forecast)
    mean_demand = np.mean(forecast_array)

    if mean_demand <= 0.01:
        return 0.0

    lt_int = int(lead_time) if lead_time > 0 else 1
    
    # 4000 simülasyon x Lead Time boyutunda dev bir Poisson matrisi tek seferde oluşturuluyor
    simulated_demands = np.random.poisson(lam=mean_demand, size=(simulations, lt_int))
    
    # Her simülasyon için satır toplamları alınıyor
    total_demands = np.sum(simulated_demands, axis=1)
    
    # Stok seviyesini aşan simülasyonların oranı bulunuyor
    stock_out_prob = np.sum(total_demands > stock_level) / simulations
    return float(stock_out_prob)

def optimize_inventory(forecast, lead_time, review_period, target_service_level, current_stock, lot_size, birim_fiyat):
    mean_demand = np.mean(forecast)
    var_demand = np.var(forecast) if np.var(forecast) > 0 else 0.1
    var_lead_time = (lead_time * 0.2) ** 2 
    sigma_L = math.sqrt((lead_time * var_demand) + ((mean_demand**2) * var_lead_time))
    
    z_val = norm.ppf(target_service_level)
    safety_stock = z_val * sigma_L
    
    s = (mean_demand * lead_time) + safety_stock
    S = (mean_demand * (lead_time + review_period)) + safety_stock
    
    raw_qty = max(S - current_stock, 0)
    final_order = math.ceil(raw_qty / lot_size) * lot_size if raw_qty > 0 else 0
    
    risk = simulate_stock_out(forecast, current_stock + final_order, lead_time)
    weeks_of_supply = current_stock / mean_demand if mean_demand > 0 else 99.0
    
    return {
        "s_reorder_point": round(s, 1),
        "S_order_up_to": round(S, 1),
        "raw_suggestion": round(raw_qty, 1),
        "final_order_qty": int(final_order),
        "lot_size": int(lot_size),
        "toplam_maliyet_euro": round(final_order * birim_fiyat, 2),
        "final_stockout_risk": round(risk, 3),
        "wos": round(weeks_of_supply, 1)
    }