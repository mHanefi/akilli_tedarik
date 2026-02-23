import numpy as np
import pandas as pd

class ADIDA:
    def __init__(self, aggregation_window=4):
        self.aggregation_window = aggregation_window

    def aggregate(self, series):
        """
        Zaman serisini k uzunluğundaki bloklar halinde toplar (Y_T) 
        ve ortalama tüketim hızını seriye geri dağıtır.
        """
        series = pd.Series(series).fillna(0).reset_index(drop=True)
        n = len(series)
        
        # Belirlenen pencere boyutuna göre veriyi grupla ve topla
        grouped_indices = np.arange(n) // self.aggregation_window
        aggregated = series.groupby(grouped_indices).sum()
        
        # Toplanan veriyi pencere boyutuna bölerek güncel/haftalık ortalamayı bul
        disaggregated = np.repeat(
            aggregated.values / self.aggregation_window, 
            self.aggregation_window
        )
        
        # Orijinal boyuta kırp (Artık veri gürültüden arındırılmış düzeltilmiş seri)
        return pd.Series(disaggregated[:n], index=series.index)