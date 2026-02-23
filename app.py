import pandas as pd
import numpy as np
from scipy.stats import norm
import math
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from core.data_pipeline import load_and_preprocess_data
from core.adida import ADIDA
from core.sba import SBA
from core.tsb import TSB
from core.catboost_model import CatBoostModel

# -----------------------------
# GLOBAL ÖZNİTELİK (FEATURE) ÜRETİMİ
# -----------------------------
def build_global_dataset(df):
    all_features = []
    
    for sku in df["sku"].unique():
        sku_df = df[df["sku"] == sku].copy().reset_index(drop=True)
        demand_series = sku_df["demand"]
        
        adida = ADIDA(aggregation_window=4)
        adida_out = adida.aggregate(demand_series)
        
        sba = SBA(alpha=0.1)
        sba_out = sba.fit(adida_out)
        
        tsb = TSB(alpha=0.1, beta=0.1)
        tsb_out = tsb.fit(adida_out)
        
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
    global_df = global_df.dropna().reset_index(drop=True) 
    
    return global_df

# -----------------------------
# AKADEMİK MODEL DOĞRULAMA VE EĞİTİM
# -----------------------------
def train_and_validate_model(X, y, cat_features):
    # Akademik doğrulama için veriyi %80 Eğitim, %20 Test olarak ayırıyoruz
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # 1. Test Modeli (Sadece metrik hesaplamak için)
    eval_model = CatBoostModel(cat_features=cat_features)
    eval_model.train(X_train, y_train)
    
    preds = eval_model.predict(X_test)
    
    # Sıfırın altındaki tahminleri düzelt (Talep eksi olamaz)
    preds = np.maximum(preds, 0)
    
    # Metrikleri Hesapla
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    metrics = {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 3)
    }
    
    # 2. Nihai Model (Gelecek tahmini için %100 veri ile eğitilmiş gerçek model)
    final_model = CatBoostModel(cat_features=cat_features)
    final_model.train(X, y)
    
    return final_model, metrics

# -----------------------------
# ÜRETİM PLANI ENTEGRELİ DİNAMİK TAHMİN
# -----------------------------
def forecast_future_for_sku(model, last_row, future_plan_df, sku_arac_modeli, steps=8):
    future_predictions = []
    current_row = last_row.copy()
    
    plan_filtered = future_plan_df[future_plan_df["arac_modeli"] == sku_arac_modeli].sort_values("date")
    planlanmis_uretimler = plan_filtered["planlanan_uretim"].values
    
    baseline_production = 15.0 

    for step in range(steps):
        base_pred = model.predict(current_row)[0]
        base_pred = max(0.0, float(base_pred))
        
        if step < len(planlanmis_uretimler):
            hedef_uretim = planlanmis_uretimler[step]
            katsayi = hedef_uretim / baseline_production
            final_pred = base_pred * katsayi
        else:
            final_pred = base_pred

        future_predictions.append(final_pred)

        current_row["lag3"] = current_row["lag2"]
        current_row["lag2"] = current_row["lag1"]
        current_row["lag1"] = final_pred

    return future_predictions

# -----------------------------
# MONTE CARLO STOK SİMÜLASYONU
# -----------------------------
def simulate_stock_out(forecast, stock_level, lead_time, simulations=4000):
    forecast_array = np.array(forecast)
    mean = np.mean(forecast_array)
    std = np.std(forecast_array) if np.std(forecast_array) > 0 else 0.1

    stock_out_count = 0
    lead_time_int = int(lead_time) if lead_time > 0 else 1
    
    for _ in range(simulations):
        simulated_demand = np.random.normal(mean, std, lead_time_int)
        total_demand = np.sum(simulated_demand)

        if total_demand > stock_level:
            stock_out_count += 1

    return stock_out_count / simulations

# -----------------------------
# İŞ MANTIĞI EKLENMİŞ ENVANTER OPTİMİZASYONU
# -----------------------------
def optimize_inventory(forecast, lead_time, review_period, target_service_level, current_stock, lot_size, birim_fiyat):
    mean_demand = np.mean(forecast)
    std_demand = np.std(forecast) if np.std(forecast) > 0 else 0.1

    z_value = norm.ppf(target_service_level)
    safety_stock = z_value * std_demand * np.sqrt(lead_time)

    s = (mean_demand * lead_time) + safety_stock
    S = (mean_demand * (lead_time + review_period)) + safety_stock
    step = mean_demand if mean_demand > 0 else 1.0

    for _ in range(50):
        raw_order_qty = max(S - current_stock, 0)
        actual_order_qty = math.ceil(raw_order_qty / lot_size) * lot_size if raw_order_qty > 0 else 0
        
        effective_stock = current_stock + actual_order_qty
        stock_out_prob = simulate_stock_out(forecast, effective_stock, lead_time)
        target_stockout = 1 - target_service_level

        if abs(stock_out_prob - target_stockout) < 0.01:
            break

        if stock_out_prob > target_stockout:
            S += step  
        else:
            S -= step  

        if S < s:
            S = s

    final_raw_order = max(S - current_stock, 0)
    final_order = math.ceil(final_raw_order / lot_size) * lot_size if final_raw_order > 0 else 0
    toplam_maliyet = final_order * birim_fiyat

    return {
        "lead_time_used": lead_time,
        "s_reorder_point": round(s, 2),
        "S_order_up_to": round(S, 2),
        "raw_suggestion": round(final_raw_order, 2),
        "final_order_qty": int(final_order),
        "lot_size": int(lot_size),
        "toplam_maliyet_euro": round(toplam_maliyet, 2),
        "final_stockout_risk": round(stock_out_prob, 3)
    }
# -----------------------------
# ANA ÇALIŞTIRMA BLOĞU (Burası eksikti!)
# -----------------------------
if __name__ == "__main__":
    file_path = "data/demand.csv"
    plan_path = "data/production_plan.csv"
    
    print("1. Veri yükleniyor ve ön işleme yapılıyor...")
    df = load_and_preprocess_data(file_path)
    future_plan_df = pd.read_csv(plan_path)
    
    print("2. TSB ve SBA İstatistiksel öznitelikleri çıkartılıyor...")
    global_df = build_global_dataset(df)
    
    print("3. Hibrit CatBoost Modeli Eğitiliyor ve Test Ediliyor...")
    X = global_df.drop(["date", "demand"], axis=1)
    y = global_df["demand"]
    cat_features = ["sku", "parca_ailesi", "arac_modeli"]
    
    model, metrics = train_and_validate_model(X, y, cat_features)
    print(f"--> Model Başarısı - R2: {metrics['R2']}, RMSE: {metrics['RMSE']}")
    
    print("\n4. Optimizasyon çalıştırılıyor...\n")
    sku_list = df["sku"].unique()
    
    for sku in sku_list:
        sku_data = global_df[global_df["sku"] == sku]
        if sku_data.empty: continue
            
        last_row = sku_data.drop(["date", "demand"], axis=1).iloc[-1:]
        
        sku_lead_time = last_row["lead_time"].values[0]
        sku_lot_size = last_row["lot_size"].values[0]
        sku_fiyat = last_row["birim_fiyat"].values[0]
        sku_arac_modeli = last_row["arac_modeli"].values[0]
        
        forecast = forecast_future_for_sku(model, last_row, future_plan_df, sku_arac_modeli, steps=8)
        
        result = optimize_inventory(
            forecast=forecast, lead_time=sku_lead_time, review_period=4,
            target_service_level=0.95, current_stock=5,
            lot_size=sku_lot_size, birim_fiyat=sku_fiyat
        )
        
        print(f"--- {sku} --- | Sipariş: {result['final_order_qty']} Adet | Maliyet: €{result['toplam_maliyet_euro']}")