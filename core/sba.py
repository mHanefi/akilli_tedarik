import numpy as np
import pandas as pd
from scipy.optimize import minimize

class SBA:
    def __init__(self, alpha="auto"):
        self.alpha = alpha
        self.best_alpha_ = None

    def _calculate_forecast(self, demand, alpha):
        n = len(demand)
        z = np.zeros(n)
        p = np.zeros(n)
        forecast = np.zeros(n)

        first_demand_idx = np.where(demand > 0)[0]
        if len(first_demand_idx) == 0:
            return forecast
        
        first_demand = first_demand_idx[0]
        z[first_demand] = demand[first_demand]
        p[first_demand] = 1
        interval = 1

        for t in range(first_demand + 1, n):
            # 1. ÖLÜMCÜL HATA ÇÖZÜLDÜ: Rapor Sayfa 19'daki (1 - alpha/2) yansızlık katsayısı eklendi!
            # Bu katsayı olmazsa model SBA değil, hatalı Croston yöntemi olur.
            bias_correction = (1 - (alpha / 2))
            forecast[t] = bias_correction * (z[t-1] / p[t-1]) if p[t-1] > 0 else 0
            
            if demand[t] > 0:
                z[t] = z[t-1] + alpha * (demand[t] - z[t-1])
                p[t] = p[t-1] + alpha * (interval - p[t-1])
                interval = 1
            else:
                z[t] = z[t-1]
                p[t] = p[t-1]
                interval += 1
        return forecast

    def _objective(self, params, demand):
        alpha = params[0]
        forecast = self._calculate_forecast(demand, alpha)
        first_demand_idx = np.where(demand > 0)[0]
        if len(first_demand_idx) == 0: return 0
        start = first_demand_idx[0] + 1
        if start >= len(demand): return 0
        return np.mean((demand[start:] - forecast[start:])**2)

    def fit(self, series):
        demand = pd.Series(series).fillna(0).values
        
        if self.alpha == "auto":
            train_size = int(len(demand) * 0.8)
            train_demand = demand[:train_size] if train_size > 0 else demand
            
            try:
                res = minimize(self._objective, x0=[0.1], args=(train_demand,), bounds=[(0.01, 0.99)])
                self.best_alpha_ = res.x[0]
            except Exception:
                self.best_alpha_ = 0.1
        else:
            self.best_alpha_ = self.alpha
            
        forecast = self._calculate_forecast(demand, self.best_alpha_)
        return pd.Series(forecast, index=series.index)