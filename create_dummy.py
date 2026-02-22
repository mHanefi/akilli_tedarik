import pandas as pd
import numpy as np

dates = pd.date_range(start="2023-01-01", periods=52, freq="W")

data = []

for sku in ["A", "B"]:

    for date in dates:
        demand = np.random.poisson(lam=3)  # ortalama 3 talep
        data.append([date, sku, demand])

df = pd.DataFrame(data, columns=["date", "sku", "demand"])

df.to_excel("dummy_demand.xlsx", index=False)

print("dummy_demand.xlsx oluşturuldu.")