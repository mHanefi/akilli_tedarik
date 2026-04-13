import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def calculate_nbd_parameters(mean, variance):
    """
    Negatif Binom Dağılımı (NBD) için ortalama ve varyans değerlerini 
    numpy'ın beklediği 'n' (başarı sayısı) ve 'p' (olasılık) parametrelerine dönüştürür.
    Not: NBD'de varyans her zaman ortalamadan büyük olmalıdır (Overdispersion).
    """
    if variance <= mean:
        raise ValueError("Aralıklı/Dalgalı talepte varyans ortalamadan büyük olmalıdır (Overdispersion).")
    
    p = mean / variance
    n = (mean ** 2) / (variance - mean)
    return n, p

def generate_zinb_demand(days, p_occurrence, mean_size, var_size):
    """
    Sıfır Şişirilmiş Negatif Binom (ZINB) veri üretim fonksiyonu.
    Aşama 1: Bernoulli süreci ile talebin gerçekleşip gerçekleşmeyeceğini belirle (0 veya 1).
    Aşama 2: Gerçekleşen talepler için NBD ile sipariş miktarını belirle.
    """
    # Aşama 1: Bernoulli Dağılımı (Talep Olasılığı)
    # p_occurrence olasılığı ile 1, (1-p_occurrence) olasılığı ile 0 üretir.
    occurrence = np.random.binomial(n=1, p=p_occurrence, size=days)
    
    # Aşama 2: Negatif Binom Dağılımı (Talep Boyutu)
    n_param, p_param = calculate_nbd_parameters(mean_size, var_size)
    demand_sizes = np.random.negative_binomial(n=n_param, p=p_param, size=days)
    
    # Aşama 1 ve Aşama 2'yi birleştir (Sadece talep gerçekleşen günlerde miktar bas)
    final_demand = occurrence * demand_sizes
    return final_demand

def main():
    print("Akademik sentetik veri üretimi (ZINB tabanlı) başlatılıyor...")
    
    # Veri setinin tarih aralığı (2 Yıllık geçmiş veri)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    num_days = len(date_range)
    
    # Simüle edilecek Kanban parçalarının profilleri (ZINB Parametreleri)
    # p_occurrence: Bernoulli olasılığı (Örn: 0.15 = %15 ihtimalle o gün talep gelir)
    # mean_size: Talep geldiğinde ortalama miktar
    # var_size: Varyans (Gürültü/Dalgalanma seviyesi - Ortalamadan büyük olmalı)
    parts_config = [
        {'Kodu': 'CIVATA-M8-103', 'Ailesi': 'Bağlantı', 'BOM_Katsayisi': 12, 'Lead_Time': 14, 'p_occurrence': 0.15, 'mean': 150, 'var': 600},
        {'Kodu': 'PUL-10MM-201', 'Ailesi': 'Bağlantı', 'BOM_Katsayisi': 24, 'Lead_Time': 7,  'p_occurrence': 0.30, 'mean': 300, 'var': 1500},
        {'Kodu': 'SOMUN-M12-005', 'Ailesi': 'Bağlantı', 'BOM_Katsayisi': 4,  'Lead_Time': 30, 'p_occurrence': 0.05, 'mean': 40,  'var': 200},
        {'Kodu': 'KLIPS-PLST-55', 'Ailesi': 'Kabin',    'BOM_Katsayisi': 2,  'Lead_Time': 45, 'p_occurrence': 0.08, 'mean': 20,  'var': 80},
        {'Kodu': 'HORTUM-KLP-88', 'Ailesi': 'Motor',    'BOM_Katsayisi': 1,  'Lead_Time': 21, 'p_occurrence': 0.10, 'mean': 10,  'var': 30}
    ]
    
    # Tekrarlanabilirlik (Reproducibility) için sabit seed (Akademik testler için şarttır)
    np.random.seed(42)
    
    all_data = []
    
    for part in parts_config:
        # ZINB algoritması ile talebi üret
        demand_series = generate_zinb_demand(
            days=num_days, 
            p_occurrence=part['p_occurrence'], 
            mean_size=part['mean'], 
            var_size=part['var']
        )
        
        # Parçaya ait DataFrame'i oluştur
        df_part = pd.DataFrame({
            'Tarih': date_range,
            'Parca_Kodu': part['Kodu'],
            'Parca_Ailesi': part['Ailesi'],
            'BOM_Katsayisi': part['BOM_Katsayisi'],
            'Lead_Time': part['Lead_Time'],
            'Talep_Miktari': demand_series
        })
        
        all_data.append(df_part)
    
    # Tüm parçaları tek bir veri setinde birleştir
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Data klasörünü kontrol et ve kaydet
    os.makedirs('data', exist_ok=True)
    save_path = os.path.join('data', 'dummy_inventory_data.csv')
    final_df.to_csv(save_path, index=False)
    
    print(f"Veri üretimi başarıyla tamamlandı!")
    print(f"Toplam Satır: {len(final_df)} | Kayıt Yeri: {save_path}")
    print("\nÜretilen veriden bir kesit (Çoklu sıfırları ve patlamaları inceleyebilirsiniz):")
    print(final_df[final_df['Parca_Kodu'] == 'SOMUN-M12-005'].head(15))

if __name__ == "__main__":
    main()
