import numpy as np
import pandas as pd

class TSB:
    def __init__(self, alpha=0.1, beta=0.1):
        self.alpha = alpha # Miktar düzeltme katsayısı
        self.beta = beta   # Olasılık düzeltme katsayısı

    def fit(self, series):
        demand = pd.Series(series).fillna(0).values
        n = len(demand)

        z = np.zeros(n) # Beklenen talep büyüklüğü
        p = np.zeros(n) # Talep gerçekleşme olasılığı
        forecast = np.zeros(n)

        first_demand_idx = np.where(demand > 0)[0]
        if len(first_demand_idx) == 0:
            return pd.Series(np.zeros(n))
            
        first_demand = first_demand_idx[0]
        z[first_demand] = demand[first_demand]
        p[first_demand] = 1

        for t in range(first_demand + 1, n):
            indicator = 1 if demand[t] > 0 else 0
            
            # Olasılık HER ZAMAN güncellenir (Eskimeyi yakalayan yer burası)
            p[t] = p[t-1] + self.beta * (indicator - p[t-1])
            
            # Talep büyüklüğü SADECE talep gerçekleştiğinde güncellenir!
            if indicator == 1:
                z[t] = z[t-1] + self.alpha * (demand[t] - z[t-1])
            else:
                z[t] = z[t-1]
            
            # Nihai tahmin: Olasılık x Beklenen Büyüklük
            forecast[t] = p[t] * z[t]
            
        return pd.DataFrame({
            'tsb_forecast': forecast,
            'tsb_probability': p,  # CatBoost'a "Eskime Sinyali" olarak gidecek!
            'tsb_magnitude': z     # CatBoost'a "Referans Miktar" olarak gidecek!
        })