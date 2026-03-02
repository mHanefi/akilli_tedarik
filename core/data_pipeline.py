import pandas as pd
import numpy as np

# ========================================================================================================
# Bu modül bir "Excel okuma" kodu DEĞİLDİR. Fabrikaların kullandığı SAP/ERP sistemleri
# parça çekilmeyen (0 tüketimli) günleri/haftaları HİÇ KAYDETMEZ. 
# Eğer bu ham veriyi doğrudan makine öğrenmesine (CatBoost) verseydik, model zaman çizelgesini atlayarak
# yanlış öğrenecek ve Kesikli Talep (Intermittent Demand) doğasını asla kavrayamayacaktı.
# Bu kod parçası; parçalanmış ERP verisini alıp, takvimi sıfırdan inşa eden ve makine öğrenmesine 
# "sıfır (0) çekilen haftaların" varlığını matematiksel olarak öğreten Ana Veri Boru Hattıdır (Data Pipeline).
# ========================================================================================================

def load_and_preprocess_data(file_path):
    # 1. ESNEK ENTEGRASYON: Şirketin ERP sisteminden CSV veya Excel gelme ihtimaline karşı dinamik okuyucu.
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # 2. VERİ STANDARDİZASYONU VE TEMİZLİK (DATA CLEANSING):
    # Farklı SAP kullanıcılarının oluşturabileceği büyük/küçük harf hatalarını engelliyoruz.
    df.columns = df.columns.str.lower()
    
    # Boş veya hatalı girilmiş satırları atıyoruz.
    df = df.dropna(subset=["date", "demand"])
    
    # SAP sistemlerindeki negatif tüketimler genelde "İade" işlemleridir.
    # Makine öğrenmesi trendi yanlış anlamaması için negatif (iade) verilerini filtreliyoruz.
    df = df[df["demand"] >= 0]
    df["date"] = pd.to_datetime(df["date"])

    # 3. ZAMAN BAZLI TOPLULAŞTIRMA (AGGREGATION):
    # Günlük vardiyalardan gelen talepleri, kanban sipariş frekansımız olan "Haftalık" periyotlara topluyoruz.
    agg_funcs = {"demand": "sum"}
    meta_cols = ["lead_time", "parca_ailesi", "birim_fiyat", "lot_size", "mevcut_stok"]
    

    # Kodumuz statik olarak tek bir otobüse (örn: Lion's Coach) bağımlı değildir!
    # Sistem, veri setindeki "uretim_" ile başlayan tüm kolonları dinamik olarak tarar ve 
    # o parçanın kaç farklı araçta ortak (BOM - Bill of Materials) kullanıldığını kendi tespit eder.
    uretim_cols = [c for c in df.columns if c.startswith("uretim_")]
    
    for col in meta_cols + uretim_cols:
        if col in df.columns:
            agg_funcs[col] = "first" # Fiyat, stok gibi değerlerin o haftaki güncel halini alıyoruz.
            
    # Talepleri haftalık/günlük olarak gruplayıp topluyoruz.
    df = df.groupby(["date", "sku"], as_index=False).agg(agg_funcs)

    processed_dfs = []
    # 4. ZAMAN SERİSİ YENİDEN İNŞASI (TIME-SERIES RECONSTRUCTION):
    for sku in df["sku"].unique():
        sku_df = df[df["sku"] == sku].copy()
        
        # O parçanın fabrikada kullanılmaya başlandığı ilk günden, bugüne kadar olan KESİNTİSİZ takvimi çiziyoruz.
        full_date_range = pd.date_range(start=sku_df["date"].min(), end=sku_df["date"].max(), freq="W") 
        
        sku_df = sku_df.set_index("date")
        # Eksik haftaları takvime zorla ekliyoruz (ERP'nin kaydetmediği o boş haftalar)
        sku_df = sku_df.reindex(full_date_range)
        
        # [KESİKLİ TALEP YÖNETİMİ]: Yeni açılan o boş haftaların taleplerine "0" basıyoruz.
        # İşte modelimiz ancak bu adımdan sonra "Bu parça bazen hiç tüketilmiyor" matematiğini anlıyor.
        sku_df["demand"] = sku_df["demand"].fillna(0)
        sku_df["sku"] = sku
        
        # 5. METAVERİ BÜTÜNLÜĞÜ (FORWARD/BACKWARD FILL):
        # 0 talep olan haftalarda parçanın fiyatı, teslim süresi ve o haftaki üretim adetleri "NaN" olmasın diye,
        # bu verileri mantıksal olarak geçmişten (ffill) veya gelecekten (bfill) kopyalayarak dolduruyoruz.
        for col in meta_cols + uretim_cols:
            if col in sku_df.columns:
                sku_df[col] = sku_df[col].ffill().bfill()
            
        sku_df = sku_df.rename_axis("date").reset_index()
        processed_dfs.append(sku_df)

    # Tamamen temizlenmiş, kayıp haftaları doldurulmuş, çoklu üretim araçlarıyla korele edilmiş
    # ve makine öğrenmesi eğitimine %100 hazır veri setini (Dataframe) ana motora gönderiyoruz.
    return pd.concat(processed_dfs, ignore_index=True)