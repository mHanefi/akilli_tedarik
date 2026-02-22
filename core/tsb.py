import numpy as np
import pandas as pd


class TSB:
    def __init__(self, alpha=0.1, beta=0.1):
        self.alpha = alpha
        self.beta = beta

    def fit(self, series):
        series = pd.Series(series).fillna(0)
        demand = series.values
        n = len(demand)

        z = np.zeros(n)
        p = np.zeros(n)
        forecast = np.zeros(n)

        # İlk talep noktasını bul
        first_demand = np.argmax(demand > 0)

        if demand[first_demand] == 0:
            return pd.Series(np.zeros(n))

        z[first_demand] = demand[first_demand]
        p[first_demand] = 1

        for t in range(first_demand + 1, n):
            indicator = 1 if demand[t] > 0 else 0

            z[t] = z[t-1] + self.alpha * (demand[t] - z[t-1])
            p[t] = p[t-1] + self.beta * (indicator - p[t-1])

            forecast[t] = p[t] * z[t]

        return pd.Series(forecast)