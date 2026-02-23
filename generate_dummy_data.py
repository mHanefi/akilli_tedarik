import pandas as pd
import numpy as np
import os

def create_advanced_dummy_data():
    os.makedirs("data", exist_ok=True)
    
    np.random.seed(42) # Tutarlı veriler için
    
    # DENGELİ VERİ SETİ: Birim fiyatlar ve talep aralıkları grafiklerde şık durması için optimize edildi
    skus = [
        {"sku": "SKU_101_LC", "lead_time": 2, "parca_ailesi": "Bağlantı_Elemanı", "arac_modeli": "Lion's City", "birim_fiyat": 3.5, "lot_size": 100, "zero_prob": 0.3, "demand_range": (150, 300)},
        {"sku": "SKU_102_LC", "lead_time": 4, "parca_ailesi": "Kabin_Plastiği", "arac_modeli": "Lion's City", "birim_fiyat": 18.5, "lot_size": 20, "zero_prob": 0.5, "demand_range": (40, 90)},
        {"sku": "SKU_103_LC", "lead_time": 5, "parca_ailesi": "Elektronik_Sensör", "arac_modeli": "Lion's City", "birim_fiyat": 45.0, "lot_size": 10, "zero_prob": 0.7, "demand_range": (15, 40)},

        {"sku": "SKU_201_NT", "lead_time": 1, "parca_ailesi": "Bağlantı_Elemanı", "arac_modeli": "NEOPLAN Tourliner", "birim_fiyat": 4.0, "lot_size": 100, "zero_prob": 0.4, "demand_range": (120, 250)},
        {"sku": "SKU_202_NT", "lead_time": 6, "parca_ailesi": "Aydınlatma_Grubu", "arac_modeli": "NEOPLAN Tourliner", "birim_fiyat": 65.0, "lot_size": 5, "zero_prob": 0.8, "demand_range": (10, 25)},
        {"sku": "SKU_203_NT", "lead_time": 3, "parca_ailesi": "Sızdırmazlık", "arac_modeli": "NEOPLAN Tourliner", "birim_fiyat": 12.5, "lot_size": 50, "zero_prob": 0.5, "demand_range": (80, 160)},

        {"sku": "SKU_301_LCO", "lead_time": 3, "parca_ailesi": "Filtre_Grubu", "arac_modeli": "Lion's Coach", "birim_fiyat": 22.0, "lot_size": 25, "zero_prob": 0.5, "demand_range": (50, 120)},
        {"sku": "SKU_302_LCO", "lead_time": 7, "parca_ailesi": "Elektronik_Sensör", "arac_modeli": "Lion's Coach", "birim_fiyat": 85.0, "lot_size": 5, "zero_prob": 0.85, "demand_range": (5, 15)},
        {"sku": "SKU_303_LCO", "lead_time": 2, "parca_ailesi": "Sızdırmazlık", "arac_modeli": "Lion's Coach", "birim_fiyat": 14.0, "lot_size": 50, "zero_prob": 0.4, "demand_range": (70, 140)},

        {"sku": "SKU_401_LI", "lead_time": 2, "parca_ailesi": "Kabin_Plastiği", "arac_modeli": "Lion's Intercity", "birim_fiyat": 15.0, "lot_size": 30, "zero_prob": 0.6, "demand_range": (45, 100)},
        {"sku": "SKU_402_LI", "lead_time": 4, "parca_ailesi": "Filtre_Grubu", "arac_modeli": "Lion's Intercity", "birim_fiyat": 35.0, "lot_size": 15, "zero_prob": 0.7, "demand_range": (20, 60)},
        {"sku": "SKU_403_LI", "lead_time": 1, "parca_ailesi": "Bağlantı_Elemanı", "arac_modeli": "Lion's Intercity", "birim_fiyat": 2.8, "lot_size": 200, "zero_prob": 0.2, "demand_range": (200, 450)}
    ]
    
    dates = pd.date_range(start="2024-01-01", periods=104, freq="W")
    demand_records = []
    
    for sku_info in skus:
        # GERÇEKÇİ STOK: Ortalama talebin 1.5 ile 4.0 katı (haftası) kadar rastgele stok ata
        ortalama_talep = (sku_info["demand_range"][0] + sku_info["demand_range"][1]) / 2
        gercekci_stok = int(ortalama_talep * np.random.uniform(1.5, 4.0))
        
        for date in dates:
            if np.random.rand() > sku_info["zero_prob"]:
                demand = np.random.randint(sku_info["demand_range"][0], sku_info["demand_range"][1])
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
                "lot_size": sku_info["lot_size"],
                "mevcut_stok": gercekci_stok 
            })
            
    pd.DataFrame(demand_records).to_csv("data/demand.csv", index=False)

    future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=7), periods=8, freq="W")
    plan_records = []
    models = ["Lion's City", "NEOPLAN Tourliner", "Lion's Coach", "Lion's Intercity"]
    
    for date in future_dates:
        for model in models:
            if model == "Lion's City": uretim = np.random.randint(20, 30)
            elif model == "NEOPLAN Tourliner": uretim = np.random.randint(10, 18)
            elif model == "Lion's Coach": uretim = np.random.randint(15, 25)
            else: uretim = np.random.randint(18, 28)
                
            plan_records.append({"date": date, "arac_modeli": model, "planlanan_uretim": uretim})
            
    pd.DataFrame(plan_records).to_csv("data/production_plan.csv", index=False)
    print("✅ Dummy veriler başarıyla dengelendi ve güncellendi!")

if __name__ == "__main__":
    create_advanced_dummy_data()