import pandas as pd
import numpy as np
import os

def create_advanced_dummy_data():
    os.makedirs("data", exist_ok=True)
    np.random.seed(42)

    # Gerçek MAN ve NEOPLAN Modelleri
    models = [
        "Lions_City_12", "Lions_City_18E", "Lions_Coach", "Lions_Intercity",
        "NEOPLAN_Tourliner", "NEOPLAN_Skyliner", "NEOPLAN_Cityliner",
        "Lions_Chassis", "TGE_Minibus", "eTGE"
    ]

    dates_hist = pd.date_range(start="2024-01-01", periods=104, freq="W")
    dates_future = pd.date_range(start=dates_hist[-1] + pd.Timedelta(days=7), periods=8, freq="W")

    hist_prod = {m: np.random.randint(5, 25, size=104) for m in models}
    future_prod = {m: np.random.randint(5, 25, size=8) for m in models}

    skus = []
    # 40 Adet Ortak Kullanımlı (BOM) Parça Üretiyoruz
    for i in range(1, 41):
        # Hangi araçta kaç tane kullanıldığına dair rastgele Ürün Ağacı (BOM) matrisi
        bom = np.random.choice([0, 1, 2, 4], size=10, p=[0.4, 0.3, 0.2, 0.1])
        if sum(bom) == 0: bom[np.random.randint(0, 10)] = 1 
        
        skus.append({
            "sku": f"SKU_{i:03d}",
            "bom": bom,
            "lead_time": np.random.randint(2, 8),
            "parca_ailesi": np.random.choice(["Bağlantı_Elemanı", "Kabin_Plastiği", "Sensör", "Filtre", "Sızdırmazlık"]),
            "birim_fiyat": round(np.random.uniform(2.5, 150.0), 2),
            "lot_size": np.random.choice([10, 50, 100, 200])
        })

    demand_records = []
    for t, date in enumerate(dates_hist):
        for sku in skus:
            # Gerçek Tüketim = Her aracın üretim adedi * O araçtaki parça kullanım sayısı
            base_demand = sum(sku["bom"][j] * hist_prod[models[j]][t] for j in range(10))
            
            if np.random.rand() < 0.3: # Kesikli talep doğası gereği 0 olduğu haftalar
                demand = 0
            else:
                demand = int(base_demand * np.random.uniform(0.5, 1.5))

            rec = {
                "date": date,
                "sku": sku["sku"],
                "demand": demand,
                "lead_time": sku["lead_time"],
                "parca_ailesi": sku["parca_ailesi"],
                "birim_fiyat": sku["birim_fiyat"],
                "lot_size": sku["lot_size"],
                # MANTIK: Sipariş verdirtmek için başlangıç stoğunu çok tehlikeli/düşük seviyede tutuyoruz!
                "mevcut_stok": np.random.randint(0, max(5, int(base_demand * 0.3))) 
            }
            # 10 farklı aracın üretim planı modele kolon kolon veriliyor
            for j, m in enumerate(models):
                rec[f"uretim_{m}"] = hist_prod[m][t]
            demand_records.append(rec)

    pd.DataFrame(demand_records).to_csv("data/demand.csv", index=False)

    plan_records = []
    for t, date in enumerate(dates_future):
        rec = {"date": date}
        for j, m in enumerate(models):
            rec[f"uretim_{m}"] = future_prod[m][t]
        plan_records.append(rec)

    pd.DataFrame(plan_records).to_csv("data/production_plan.csv", index=False)
    print("✅ 40 Parça ve 10 Araçlık Çoklu BOM Verisi Başarıyla Oluşturuldu!")

if __name__ == "__main__":
    create_advanced_dummy_data()