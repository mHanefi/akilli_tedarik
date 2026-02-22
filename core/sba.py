import numpy as np
import pandas as pd


class SBA:
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, series):
        series = pd.Series(series).fillna(0)

        demand = series.values
        n = len(demand)

        z = np.zeros(n)  # demand size
        p = np.zeros(n)  # interval
        forecast = np.zeros(n)

        first_demand = np.argmax(demand > 0)

        if demand[first_demand] == 0:
            return pd.Series(np.zeros(n))

        z[first_demand] = demand[first_demand]
        p[first_demand] = 1

        interval = 1

        for t in range(first_demand + 1, n):
            if demand[t] > 0:
                z[t] = z[t-1] + self.alpha * (demand[t] - z[t-1])
                p[t] = p[t-1] + self.alpha * (interval - p[t-1])
                interval = 1
            else:
                z[t] = z[t-1]
                p[t] = p[t-1]
                interval += 1

            forecast[t] = (1 - self.alpha / 2) * (z[t] / p[t]) if p[t] != 0 else 0

        return pd.Series(forecast)