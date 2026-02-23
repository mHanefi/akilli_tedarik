import pandas as pd
import numpy as np
import os

def create_dummy_data():
    os.makedirs("data", exist_ok=True)
    
    # MAN'daki duruma uygun, FİYAT ve PAKET İÇİ MİKTAR (Lot Size) eklenmiş senaryo:
    skus = [
        {"sku": "SKU_001", "lead_time": 1, "parca_ailesi": "Vida_Somun", "arac_modeli": "Lion's City", "birim_fiyat": 0.15, "lot_size": 500},
        {"sku": "SKU_002", "lead_time": 6, "parca_ailesi": "Ithal_Elektronik", "arac_modeli": "NEOPLAN Tourliner", "birim_fiyat": 120.0, "lot_size": 10},
        {"sku": "SKU_003", "lead_time": 3, "parca_ailesi": "Motor_Aksami", "arac_modeli": "Lion's Coach", "birim_fiyat": 45.5, "lot_size": 50}
    ]
    
    dates = pd.date_range(start="2024-01-01", periods=104, freq="W")
    demand_records = []
    
    for sku_info in skus:
        for date in dates:
            if np.random.rand() > 0.6:
                demand = np.random.randint(10, 150)
            else:
                demand = 0
                
            demand_records.append({
                "date": date,
                "sku": sku_info["sku"],
                "demand": demand,
                "lead_time": sku_info["lead_time"],
                "parca_ailesi": sku_info["parca_ailesi"],
                "arac_modeli": sku_info["arac_modeli"],
                "birim_fiyat": sku_info["birim_fiyat"],
                "lot_size": sku_info["lot_size"]
            })
            
    df_demand = pd.DataFrame(demand_records)
    df_demand.to_csv("data/demand.csv", index=False)
    print("✅ data/demand.csv başarıyla oluşturuldu! (Finansal veriler eklendi)")

if __name__ == "__main__":
    create_dummy_data() 