import pandas as pd
import numpy as np
import os

def create_dummy_data():
    os.makedirs("data", exist_ok=True)
    
    # SKU Bazlı Farklı Başlangıç Stokları ve Finansal Parametreler
    skus = [
        {"sku": "SKU_001", "lead_time": 1, "parca_ailesi": "Vida_Somun", "arac_modeli": "Lion's City", "birim_fiyat": 0.15, "lot_size": 500, "mevcut_stok": 1200},
        {"sku": "SKU_002", "lead_time": 6, "parca_ailesi": "Ithal_Elektronik", "arac_modeli": "NEOPLAN Tourliner", "birim_fiyat": 120.0, "lot_size": 10, "mevcut_stok": 8},
        {"sku": "SKU_003", "lead_time": 3, "parca_ailesi": "Motor_Aksami", "arac_modeli": "Lion's Coach", "birim_fiyat": 45.5, "lot_size": 50, "mevcut_stok": 60}
    ]
    
    dates = pd.date_range(start="2024-01-01", periods=104, freq="W")
    demand_records = []
    
    for sku_info in skus:
        for date in dates:
            demand = np.random.randint(10, 150) if np.random.rand() > 0.6 else 0
            demand_records.append({
                "date": date, "sku": sku_info["sku"], "demand": demand,
                "lead_time": sku_info["lead_time"], "parca_ailesi": sku_info["parca_ailesi"],
                "arac_modeli": sku_info["arac_modeli"], "birim_fiyat": sku_info["birim_fiyat"],
                "lot_size": sku_info["lot_size"], "mevcut_stok": sku_info["mevcut_stok"]
            })
            
    pd.DataFrame(demand_records).to_csv("data/demand.csv", index=False)
    
    # Gelecek Üretim Planı (MRP)
    future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=7), periods=8, freq="W")
    plan_records = []
    for date in future_dates:
        for model in ["Lion's City", "NEOPLAN Tourliner", "Lion's Coach"]:
            plan_records.append({"date": date, "arac_modeli": model, "planlanan_uretim": np.random.randint(5, 25)})
    pd.DataFrame(plan_records).to_csv("data/production_plan.csv", index=False)
    print("✅ Finansal ve Stok verili dummy dosyalar hazır!")

if __name__ == "__main__":
    create_dummy_data()