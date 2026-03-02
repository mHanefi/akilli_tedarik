import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ========================================================================================================
# Bu dosya, ünlü Croston modelinin en büyük teorik hatasını çözen "Syntetos-Boylan Approximation (SBA)" 
# modelinin Endüstri 4.0'a (Makine Öğrenmesi Entegrasyonu) uyarlanmış Python mimarisidir.
#
# NEDEN YAZDIK? (Neden Doğrudan Croston Kullanmadık?)
# Standart Croston modeli, kesikli taleplerin "aralıklarını" (interval) ve "büyüklüklerini" (size)
# ayrı ayrı hesaplar. Ancak Syntetos ve Boylan (2001) ispatlamıştır ki; Croston'ın kullandığı 
# ters çevirme (inversion) işlemi matematiksel olarak taraflıdır (biased) ve her zaman talebi 
# "Olması Gerekenden Yüksek (Over-forecast)" hesaplayarak fabrikada gereksiz stok (ölü sermaye) yaratır.
# Bu algoritma, Croston'ın bu aşırı tahmin (bias) hastalığını tedavi eden mucizevi filtredir.
# ========================================================================================================

class SBA:
    def __init__(self, alpha="auto"):
        # Alpha, modelin geçmişten ne kadar etkileneceğini belirleyen düzeltme katsayısıdır.
        # Bu değeri "auto" bırakarak, C-sınıfı bir cıvata ile, daha az kullanılan etiket gibi parçaların
        # kendi spesifik alpha değerlerini optimizasyonla (kendi kendilerine) bulmalarını sağlıyoruz.
        self.alpha = alpha
        self.best_alpha_ = None

    def _calculate_forecast(self, demand, alpha):
        n = len(demand)
        z = np.zeros(n) # Talebin Büyüklüğü (Size)
        p = np.zeros(n) # Talepler Arası Geçen Süre (Interval)
        forecast = np.zeros(n)

        first_demand_idx = np.where(demand > 0)[0]
        if len(first_demand_idx) == 0:
            return forecast
        
        first_demand = first_demand_idx[0]
        # Sistemin "Ateşlenme" Noktası
        z[first_demand] = demand[first_demand]
        p[first_demand] = 1
        interval = 1

        for t in range(first_demand + 1, n):
            # [AKADEMİK VE FİNANSAL NOKTAS - YANSIZLIK KATSAYISI]:
            # Klasik Croston'da formül sadece (z[t-1] / p[t-1])'dir.
            # Biz buraya literatürdeki ünlü (1 - alpha/2) yansızlık (bias correction) katsayısını ekledik.
            # Bu küçük çarpım, MAN Türkiye'nin yüz binlerce Euro'luk gereksiz stok maliyetini (Overstock)
            # engelleyen ve modeli "Croston" yerine "SBA" yapan o noktadır.
            bias_correction = (1 - (alpha / 2))
            
            # [VERİ SIZINTISI ENGELİ]: Tahmin (forecast[t]), daima BİR ÖNCEKİ (t-1) güncellemelerle yapılır. 
            # Model asla geleceği (t) göremez. Nedensellik  ilkesi korunmuştur.
            forecast[t] = bias_correction * (z[t-1] / p[t-1]) if p[t-1] > 0 else 0
            
            # SADECE talep geldiği hafta (demand[t] > 0) model kendini günceller.
            if demand[t] > 0:
                z[t] = z[t-1] + alpha * (demand[t] - z[t-1]) # Yeni talep büyüklüğü öğrenilir
                p[t] = p[t-1] + alpha * (interval - p[t-1])  # Yeni talep aralığı (sıklık) öğrenilir
                interval = 1 # Sayaç sıfırlanır
            else:
                # Talep yoksa (0 ise), model donar ve bekler. Sadece aralık sayacı (interval) artar.
                z[t] = z[t-1]
                p[t] = p[t-1]
                interval += 1
        return forecast

    def _objective(self, params, demand):
        # [OPTİMİZASYON MOTORU]: Hataların Karesi Ortalamasını (MSE) minimize edecek hedef fonksiyon.
        alpha = params[0]
        forecast = self._calculate_forecast(demand, alpha)
        first_demand_idx = np.where(demand > 0)[0]
        if len(first_demand_idx) == 0: return 0
        start = first_demand_idx[0] + 1
        if start >= len(demand): return 0
        return np.mean((demand[start:] - forecast[start:])**2)

    def fit(self, series):
        demand = pd.Series(series).fillna(0).values
        
        # [DİNAMİK MAKİNE ÖĞRENMESİ VİZYONU]: 
        if self.alpha == "auto":
            # Model, optimum Alpha'yı ararken "geleceği" görmesin diye verinin sadece ilk %80'inde (Train) çalışır.
            train_size = int(len(demand) * 0.8)
            train_demand = demand[:train_size] if train_size > 0 else demand
            
            try:
                # SciPy optimizasyonu, 0.01 ile 0.99 arasında en düşük hatayı veren Alpha'yı arar ve bulur.
                res = minimize(self._objective, x0=[0.1], args=(train_demand,), bounds=[(0.01, 0.99)])
                self.best_alpha_ = res.x[0]
            except Exception:
                # Sistemsel bir kopmada kodu çökertmez, güvenli değere (0.1) döner.
                self.best_alpha_ = 0.1
        else:
            self.best_alpha_ = self.alpha
            
        # Optimize edilmiş parametrelerle son hesaplama yapılır.
        # Bu pürüzsüzleştirilmiş "SBA Serisi", Makine Öğrenmesi (CatBoost) trendi kavraması için ana kolonlardan biri olur.
        forecast = self._calculate_forecast(demand, self.best_alpha_)
        return pd.Series(forecast, index=series.index)