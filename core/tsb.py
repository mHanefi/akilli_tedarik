import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ========================================================================================================
# Bu dosya, klasik zaman serilerinin (ARIMA, Üstel Düzeltme) C-Sınıfı yedek parçalarda Neden Çöktüğünü
# kanıtlayan ve buna çözüm üreten 'Teunter-Syntetos-Babai (TSB)' algoritmasının yeniden yazılmış halidir.
#
# NEDEN TSB KULLANDIK? (Croston Yerine Neden TSB?)
# Ünlü Croston modeli, sadece talep geldiği zaman kendini günceller. Eğer bir parça "Ölü" 
# duruma geçerse ve aylarca talep görmezse, Croston hala o parçanın tüketildiğini sanır ve sipariş verir.
# TSB modeli ise talebi İKİYE BÖLER:
# 1. P(t): O hafta talebin gelme OLASILIĞI (Her hafta güncellenir, gelmezse olasılık düşer)
# 2. Z(t): Talep gelirse KAÇ ADET (Büyüklük) geleceği.
# İşte CatBoost'a bu iki ayrı zeka filtresini vererek, YZ'nin 'ölü' parçaları fark etmesini sağlıyoruz!
# ========================================================================================================

class TSB:
    def __init__(self, alpha="auto", beta="auto"):
        # Alpha (Büyüklük/Adet güncelleme hızı) ve Beta (Olasılık güncelleme hızı).
        # Çoğu akademik tezde bu değerler 0.1 gibi sabit (hardcoded) verilir ve geçilir. 
        # Bizim sistemimizde "auto" özelliği vardır. Her C-Sınıfı parça kendi karakterine göre 
        # kendi optimum Alpha ve Beta değerini matematiksel olarak kendi bulur.
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
        # Sistemin başlangıç noktası 
        z[first_demand] = demand[first_demand]
        p[first_demand] = 1

        for t in range(first_demand + 1, n):
            # [VERİ SIZINTISI ENGELİ - SIFIR GELECEK ETKİSİ]:
            # t. haftanın tahmini (forecast[t]), sadece ama sadece 
            # bir önceki haftanın (t-1) olasılık (p) ve büyüklük (z) değerleri çarpılarak bulunur.
            # Modelin geleceği (demand[t]) görerek kopya çekmesi fiziksel ve matematiksel olarak imkansızdır.
            forecast[t] = z[t-1] * p[t-1]
            prob_series[t] = p[t-1]
            mag_series[t] = z[t-1]
            
            # 1. Aşama: Olasılık (P) Güncellemesi - TSB'nin Kalbi
            indicator = 1 if demand[t] > 0 else 0
            # Eğer o hafta talep GELMEDİYSE (indicator=0), P (Olasılık) değeri Beta katsayısı kadar AŞAĞI çekilir.
            # Bu sayede parça "Ölü (Obsolete)" ise sistem bunu hemen anlar ve makine öğrenmesine sinyal yollar.
            p[t] = p[t-1] + beta * (indicator - p[t-1])
            p[t] = np.clip(p[t], 0.0, 1.0)
            
            # 2. Aşama: Büyüklük (Z) Güncellemesi
            if indicator == 1:
                # SADECE talep geldiğinde adetler (Z) güncellenir.
                z[t] = z[t-1] + alpha * (demand[t] - z[t-1])
            else:
                z[t] = z[t-1]
                
        return forecast, prob_series, mag_series

    def _objective(self, params, demand):
        # [OPTİMİZASYON AMAÇ FONKSİYONU]:
        # SciPy motorunun minimuma indirmeye (Minimize) çalıştığı "Hata Fonksiyonu" (MSE).
        alpha, beta = params
        forecast, _, _ = self._calculate_forecast(demand, alpha, beta)
        first_demand_idx = np.where(demand > 0)[0]
        if len(first_demand_idx) == 0: return 0
        start = first_demand_idx[0] + 1
        if start >= len(demand): return 0
        return np.mean((demand[start:] - forecast[start:])**2)

    def fit(self, series):
        demand = pd.Series(series).fillna(0).values
        
        # [XAI & MAKİNE ÖĞRENMESİ VİZYONU - DİNAMİK OPTİMİZASYON]:
        # Biz modeli sabit parametrelerle körlemesine bırakmıyoruz.
        if self.alpha == "auto" or self.beta == "auto":
            
            # Model, optimum Alpha ve Beta'yı ararken tüm veriyi kullanmaz! (Eğitim/Test Ayrımı)
            # Verinin sadece ilk %80'ini kullanarak kendi içine bir deneme sınavı (Train) yapar.
            train_size = int(len(demand) * 0.8)
            train_demand = demand[:train_size] if train_size > 0 else demand
            
            try:
                # SciPy Minimize motoru ile (0.01 ile 0.99 sınırları arasında) 
                # o parçanın karakterine en uygun Alpha ve Beta değerlerini iteratif olarak bulur!
                res = minimize(self._objective, x0=[0.1, 0.1], args=(train_demand,), bounds=[(0.01, 0.99), (0.01, 0.99)])
                self.best_alpha_, self.best_beta_ = res.x
            except Exception:
                # Eğer nadir bir matematiksel yakınsama sorunu olursa sistemi çökertmez,
                # literatürdeki genel geçer kabul gören (0.1) değerlerini güvenli liman olarak atar.
                self.best_alpha_, self.best_beta_ = 0.1, 0.1
        else:
            self.best_alpha_, self.best_beta_ = self.alpha, self.beta
            
        # Sonuç: CatBoost motoruna sadece 'tahmin' değil, 3 farklı yepyeni matematiksel özellik (Feature) verilir!
        # Böylece CatBoost; parçanın trendini, gerçekleşme olasılığını ve büyüklük karakterini ayrı ayrı öğrenir.
        forecast, prob, mag = self._calculate_forecast(demand, self.best_alpha_, self.best_beta_)
        
        return pd.DataFrame({
            "tsb_forecast": forecast,
            "tsb_probability": prob,
            "tsb_magnitude": mag
        }, index=series.index)