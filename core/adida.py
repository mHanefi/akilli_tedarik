import numpy as np
import pandas as pd


class ADIDA:
    def __init__(self, aggregation_window=4):
        self.aggregation_window = aggregation_window

    def aggregate(self, series):
        """
        Weekly talep serisini alır,
        belirli pencere uzunluğunda toplar,
        tekrar haftalık ölçeğe indirir.
        """
        series = pd.Series(series).fillna(0)

        # Agregasyon
        aggregated = series.groupby(
            np.arange(len(series)) // self.aggregation_window
        ).sum()

        # Tekrar haftalık ölçeğe indirme
        disaggregated = np.repeat(
            aggregated.values / self.aggregation_window,
            self.aggregation_window
        )

        # Orijinal uzunluğa kırp
        disaggregated = disaggregated[:len(series)]

        return pd.Series(disaggregated)