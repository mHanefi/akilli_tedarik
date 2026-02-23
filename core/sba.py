import numpy as np
import pandas as pd

class SBA:
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, series):
        demand = pd.Series(series).fillna(0).values
        n = len(demand)

        z = np.zeros(n)  # Talep büyüklüğü üstel ortalaması
        p = np.zeros(n)  # Talep aralığı (interval) üstel ortalaması
        forecast = np.zeros(n)

        # İlk pozitif talebi bul
        first_demand_idx = np.where(demand > 0)[0]
        if len(first_demand_idx) == 0:
            return pd.Series(np.zeros(n))
        
        first_demand = first_demand_idx[0]
        z[first_demand] = demand[first_demand]
        p[first_demand] = 1
        
        interval = 1

        for t in range(first_demand + 1, n):
            if demand[t] > 0:
                # Talep geldi: Hem büyüklüğü hem aralığı güncelle
                z[t] = z[t-1] + self.alpha * (demand[t] - z[t-1])
                p[t] = p[t-1] + self.alpha * (interval - p[t-1])
                interval = 1 # Aralığı sıfırla (1'den başlar)
            else:
                # Talep yok: Değerler sabit kalır, aralık sayacı artar
                z[t] = z[t-1]
                p[t] = p[t-1]
                interval += 1
            
            # (1 - a/2) katsayısı ile yansız (unbiased) SBA tahmini
            if p[t] != 0:
                forecast[t] = (1 - (self.alpha / 2)) * (z[t] / p[t])

        return pd.Series(forecast) 