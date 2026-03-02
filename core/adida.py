import numpy as np
import pandas as pd
import math

# ========================================================================================================
# Bu dosya, MAN Türkiye ERP verisindeki 'Kesikli Talep' (Intermittent Demand) problemini çözen 
# matematiksel ön işlemcidir
# ADIDA (Aggregate-Disaggregate Intermittent Demand Approach), 
# yüksek varyanslı C-Sınıfı parçalar için literatürdeki en güncel akademik çözümlerden biridir.
# NEDEN YAZDIK?
# Makine öğrenmesi (CatBoost) haftalarca "0" çeken bir talebi görünce sinyali kaybeder. ADIDA, bu "0"lı 
# zaman aralıklarını üst üste katlayıp (Aggregation) zamanı sıkıştırır, orada oluşan düzleştirilmiş
# talebi Üstel Düzeltme ile öğrenir ve sonra tekrar haftalık bazda (Disaggregation) geri açar.
# ========================================================================================================

class ADIDA:
    def __init__(self, aggregation_window="auto", alpha=0.1):
        # Alpha parametresi (0.1), geçmişin izini taşıyan Üstel Düzeltme (SES) katsayısıdır.
        # Bu düşük değer, modelin anlık 'gürültülere' (spikes) karşı dirençli olmasını sağlar.
        self.aggregation_window = aggregation_window
        self.alpha = alpha  # Toplulaştırılmış seri için Üstel Düzeltme (SES) katsayısı
        

    def aggregate(self, series):
        demand = pd.Series(series).fillna(0).values
        n = len(demand)
        
        # 1. ADIM: Dinamik Pencere (k) Belirleme
        # Sistem "auto" modunda çalışırken, her parçanın kendi tüketim karakteristiğine
        # (intermittency degree) göre özel bir zaman penceresi (k) bulur. Haftada 1 tüketilen parça ile 
        # ayda 1 tüketilen parça aynı pencerede değerlendirilmez. Bu, 'Dinamik Optimizasyon'dur
        if self.aggregation_window == "auto":
            non_zero_count = np.sum(demand > 0)
            k = math.ceil(n / non_zero_count) if non_zero_count > 0 else 1
        else:
            k = self.aggregation_window
        k = max(1, int(k))
        
        # 2. ADIM: AGGREGATION (Toplulaştırma) - Birbirine girmeyen (non-overlapping) bloklar
        # Pandas'ın basit 'rolling' fonksiyonu örtüşen (overlapping) pencereler yaratır 
        # ve bu akademik olarak kesikli talepte otokorelasyon sorunlarına (yanlış sinyal) yol açar.
        # Biz burada döngüyle "Birbirine Girmeyen (Non-overlapping) Zaman Blokları" inşa ediyoruz.
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
            # i. bloğun tahmini, sadece i-1 (geçmiş) blokların verisiyle yapılır (VERİ SIZINTISI SIFIR)
            # Eğer formülde 'agg_demand[i]' kullansaydık, 
            # model geleceği görerek (Data Leakage) bugünü tahmin etmiş olurdu. Biz 'i-1' kullanarak, 
            # sistemin SADECE geçmişi (Causal) referans alarak nedensellik ilkesine uymasını garantiledik.
            agg_forecast[i] = self.alpha * agg_demand[i-1] + (1 - self.alpha) * agg_forecast[i-1]
            
        # 4. ADIM: DISAGGREGATION (Ayrıştırma) - Tahmini haftalara geri dağıt
        # Zamanı sıkıştırıp ana trendi (sinyali) bulduk, şimdi bu bulduğumuz pürüzsüz
        # değeri tekrar standart haftalık periyotlara bölüştürüp asıl makine öğrenmesi motorumuz
        # olan CatBoost'a verilmek üzere yepyeni bir "Öznitelik" (Feature) olarak hazır hale getiriyoruz.
        disaggregated_forecast = np.zeros(n)
        for i in range(num_blocks):
            start_idx = i * k
            end_idx = min((i + 1) * k, n)
            
            # Bulunan blok tahminini k'ya bölerek o bloğun içindeki haftalara dağıt
            val = agg_forecast[i] / k
            disaggregated_forecast[start_idx:end_idx] = val
            
        return pd.Series(disaggregated_forecast)