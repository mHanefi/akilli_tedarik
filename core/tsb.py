import numpy as np
import pandas as pd
from scipy.optimize import minimize

class TSB:
    def __init__(self, alpha="auto", beta="auto"):
        self.alpha = alpha
        self.beta = beta
        self.best_alpha_ = None
        self.best_beta_ = None

    def _calculate_forecast(self, demand, alpha, beta):
        n = len(demand)
        z = np.zeros(n) 
        p = np.zeros(n) 
        forecast = np.zeros(n)
        prob_series = np.zeros(n)
        mag_series = np.zeros(n)

        first_demand_idx = np.where(demand > 0)[0]
        if len(first_demand_idx) == 0:
            return forecast, prob_series, mag_series
            
        first_demand = first_demand_idx[0]
        z[first_demand] = demand[first_demand]
        p[first_demand] = 1

        for t in range(first_demand + 1, n):
            forecast[t] = z[t-1] * p[t-1]
            prob_series[t] = p[t-1]
            mag_series[t] = z[t-1]
            
            indicator = 1 if demand[t] > 0 else 0
            p[t] = p[t-1] + beta * (indicator - p[t-1])
            p[t] = np.clip(p[t], 0.0, 1.0)
            
            if indicator == 1:
                z[t] = z[t-1] + alpha * (demand[t] - z[t-1])
            else:
                z[t] = z[t-1]
                
        return forecast, prob_series, mag_series

    def _objective(self, params, demand):
        alpha, beta = params
        forecast, _, _ = self._calculate_forecast(demand, alpha, beta)
        first_demand_idx = np.where(demand > 0)[0]
        if len(first_demand_idx) == 0: return 0
        start = first_demand_idx[0] + 1
        if start >= len(demand): return 0
        return np.mean((demand[start:] - forecast[start:])**2)

    def fit(self, series):
        demand = pd.Series(series).fillna(0).values
        
        if self.alpha == "auto" or self.beta == "auto":
            train_size = int(len(demand) * 0.8)
            train_demand = demand[:train_size] if train_size > 0 else demand
            
            try:
                res = minimize(self._objective, x0=[0.1, 0.1], args=(train_demand,), bounds=[(0.01, 0.99), (0.01, 0.99)])
                self.best_alpha_, self.best_beta_ = res.x
            except Exception:
                self.best_alpha_, self.best_beta_ = 0.1, 0.1
        else:
            self.best_alpha_, self.best_beta_ = self.alpha, self.beta
            
        forecast, prob, mag = self._calculate_forecast(demand, self.best_alpha_, self.best_beta_)
        
        return pd.DataFrame({
            "tsb_forecast": forecast,
            "tsb_probability": prob,
            "tsb_magnitude": mag
        }, index=series.index)