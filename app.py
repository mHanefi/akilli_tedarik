from core.adida import ADIDA
from core.sba import SBA

dummy_series = [0, 2, 0, 3, 0, 0, 5, 0, 1, 0, 0, 4]

# ADIDA
adida_model = ADIDA(aggregation_window=4)
adida_output = adida_model.aggregate(dummy_series)

# SBA
sba_model = SBA(alpha=0.1)
sba_output = sba_model.fit(adida_output)

print("ADIDA Output:")
print(adida_output)

print("\nSBA Output:")
print(sba_output)