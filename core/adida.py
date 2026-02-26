import numpy as np
import pandas as pd
import math

class ADIDA:
    def __init__(self, aggregation_window="auto", alpha=0.1):
        self.aggregation_window = aggregation_window
        self.alpha = alpha  # Toplulaştırılmış seri için Üstel Düzeltme (SES) katsayısı

    def aggregate(self, series):
        demand = pd.Series(series).fillna(0).values
        n = len(demand)
        
        # 1. ADIM: Dinamik Pencere (k) Belirleme
        if self.aggregation_window == "auto":
            non_zero_count = np.sum(demand > 0)
            k = math.ceil(n / non_zero_count) if non_zero_count > 0 else 1
        else:
            k = self.aggregation_window
        k = max(1, int(k))
        
        # 2. ADIM: AGGREGATION (Toplulaştırma) - Birbirine girmeyen (non-overlapping) bloklar
        num_blocks = math.ceil(n / k)
        agg_demand = np.zeros(num_blocks)
        for i in range(num_blocks):
            start_idx = i * k
            end_idx = min((i + 1) * k, n)
            agg_demand[i] = np.sum(demand[start_idx:end_idx])
            
        # 3. ADIM: FORECAST (Tahmin) - Bloklar arası Üstel Düzeltme (SES)
        agg_forecast = np.zeros(num_blocks)
        agg_forecast[0] = agg_demand[0] # İlk bloğu başlangıç kabul et
        for i in range(1, num_blocks):
            # i. bloğun tahmini, sadece i-1 (geçmiş) blokların verisiyle yapılır! (VERİ SIZINTISI SIFIR)
            agg_forecast[i] = self.alpha * agg_demand[i-1] + (1 - self.alpha) * agg_forecast[i-1]
            
        # 4. ADIM: DISAGGREGATION (Ayrıştırma) - Tahmini haftalara geri dağıt
        disaggregated_forecast = np.zeros(n)
        for i in range(num_blocks):
            start_idx = i * k
            end_idx = min((i + 1) * k, n)
            
            # Bulunan blok tahminini k'ya bölerek o bloğun içindeki haftalara dağıt
            val = agg_forecast[i] / k
            disaggregated_forecast[start_idx:end_idx] = val
            
        return pd.Series(disaggregated_forecast)