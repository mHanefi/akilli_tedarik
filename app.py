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

def calculate_mase(y_true, y_pred, y_train):
    y_train_arr = np.array(y_train)
    naive_mae = np.mean(np.abs(y_train_arr[1:] - y_train_arr[:-1]))
    if naive_mae == 0: return 0.0
    return float(mean_absolute_error(y_true, y_pred) / naive_mae)

def build_global_dataset(df):
    all_features = []
    for sku in df["sku"].unique():
        sku_df = df[df["sku"] == sku].copy().reset_index(drop=True)
        demand_series = sku_df["demand"]
        
        adida_engine = ADIDA(aggregation_window="auto")
        adida_out = adida_engine.aggregate(demand_series)
        sba_engine = SBA(alpha="auto")
        sba_out = sba_engine.fit(demand_series)
        tsb_engine = TSB(alpha="auto", beta="auto")
        tsb_out = tsb_engine.fit(demand_series)
        
        sku_df["lag1"] = demand_series.shift(1)
        sku_df["lag2"] = demand_series.shift(2)
        sku_df["lag3"] = demand_series.shift(3)
        
        sku_df["adida"] = adida_out
        sku_df["sba"] = sba_out
        sku_df["tsb_forecast"] = tsb_out["tsb_forecast"]
        sku_df["tsb_probability"] = tsb_out["tsb_probability"]
        sku_df["tsb_magnitude"] = tsb_out["tsb_magnitude"]
            
        all_features.append(sku_df)
        
    global_df = pd.concat(all_features, ignore_index=True)
    global_df = global_df.sort_values("date").dropna().reset_index(drop=True)
    return global_df

def train_and_validate_model(X, y, cat_features):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    eval_model = CatBoostModel(cat_features=cat_features)
    eval_model.train(X_train, y_train, eval_set=(X_test, y_test))
    preds = np.maximum(eval_model.predict(X_test), 0)
    
    mase_val = calculate_mase(y_test, preds, y_train)
    rmse_val = math.sqrt(mean_squared_error(y_test, preds))
    mae_val = mean_absolute_error(y_test, preds)
    
    metrics = {
        "RMSE": round(rmse_val, 2),
        "MAE": round(mae_val, 2),
        "MASE": round(mase_val, 3)
    }
    
    best_iteration = eval_model.get_best_iteration()
    if best_iteration is None: best_iteration = 500
        
    final_model = CatBoostModel(cat_features=cat_features)
    final_model.model.set_params(iterations=best_iteration) 
    final_model.train(X, y)
    
    # Modelin içine bakıyoruz (Açıklanabilir YZ için)
    feat_imp = final_model.get_feature_importance()
    
    return final_model, metrics, feat_imp

def forecast_future_for_sku(model, last_row, future_plan_df, forecast_steps=8):
    future_predictions = []
    current_row = last_row.copy()
    plan_sorted = future_plan_df.sort_values("date")
    
    uretim_cols = [c for c in plan_sorted.columns if c.startswith("uretim_")]
    idx = current_row.index[0] 
    
    for step in range(forecast_steps):
        if step < len(plan_sorted):
            for u_col in uretim_cols:
                if u_col in plan_sorted.columns:
                    current_row.loc[idx, u_col] = plan_sorted.iloc[step][u_col]
            
        pred_features = current_row.drop(["date", "demand", "mevcut_stok", "lot_size", "birim_fiyat"], axis=1, errors='ignore')
        final_pred = max(0.0, float(model.predict(pred_features)[0]))
        future_predictions.append(final_pred)
        
        current_row.loc[idx, "lag3"] = current_row.loc[idx, "lag2"]
        current_row.loc[idx, "lag2"] = current_row.loc[idx, "lag1"]
        current_row.loc[idx, "lag1"] = final_pred

    return future_predictions

def simulate_stock_out(forecast, stock_level, lead_time, simulations=4000):
    forecast_array = np.array(forecast)
    mean_demand = np.mean(forecast_array)
    if mean_demand <= 0.01: return 0.0

    lt_std = lead_time * 0.2
    simulated_lts = np.maximum(0.5, np.random.normal(lead_time, lt_std, simulations))
    simulated_total_demands = np.random.poisson(lam=mean_demand * simulated_lts)
    return float(np.sum(simulated_total_demands > stock_level) / simulations)

def optimize_inventory(forecast, hist_demand, lead_time, review_period, target_service_level, current_stock, lot_size, birim_fiyat):
    mean_demand = np.mean(forecast)
    var_demand = np.var(hist_demand) if np.var(hist_demand) > 0 else 0.1
    var_lead_time = (lead_time * 0.2) ** 2 
    sigma_L = math.sqrt((lead_time * var_demand) + ((mean_demand**2) * var_lead_time))
    
    z_val = norm.ppf(target_service_level)
    safety_stock = z_val * sigma_L
    
    s = (mean_demand * lead_time) + safety_stock
    S = (mean_demand * (lead_time + review_period)) + safety_stock
    
    if current_stock <= s:
        raw_qty = max(S - current_stock, 0)
    else:
        raw_qty = 0
        
    final_order = math.ceil(raw_qty / lot_size) * lot_size if raw_qty > 0 else 0
    
    risk = simulate_stock_out(forecast, current_stock, lead_time)
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