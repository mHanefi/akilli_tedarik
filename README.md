# 🚚 MAN Türkiye A.Ş. | Hibrit Makine Öğrenmesi Tabanlı Çoklu BOM Envanter Optimizasyonu

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=for-the-badge&logo=catboost&logoColor=black)
![Optuna](https://img.shields.io/badge/Optuna-20232A?style=for-the-badge&logo=optuna&logoColor=blue)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Advanced-success?style=for-the-badge)

Bu proje, **Gazi Üniversitesi Endüstri Mühendisliği Bölümü** bitirme projesi kapsamında, **MAN Türkiye A.Ş.** fabrikasındaki aralıklı (intermittent) ve gürültülü C-sınıfı Kanban parçalarının sipariş yönetimini optimize etmek amacıyla geliştirilmiş kurumsal bir makine öğrenmesi ve karar destek sistemidir (SaaS).

---

## 🎯 Projenin Amacı ve Çözülen Problem
Ağır ticari araç (otobüs/kamyon) üretiminde, bazı alt bileşenlerin (cıvata, pul, özel bağlantı elemanları) talebi son derece düzensiz ve aralıklıdır. Geleneksel istatistiksel yöntemler (Min-Max, Standart Sapma) bu gürültüyü çözemediği için fabrikalarda ya yüksek **Stok Tutma Maliyeti** ya da **Stoksuzluk (Stock-out)** kaynaklı üretim bandı aksamaları yaşanır.

Bu proje; üretim planını (MRP - Çoklu Araç BOM Yapısı), istatistiksel sinyal işleme algoritmalarını ve Gradient Boosting mimarisini tek bir potada eriterek **stok maliyetlerini minimize eden** hibrit bir siparişleme algoritmasıdır.

## 🧠 Teknik Mimari ve Kullanılan Teknolojiler

Modelimiz, aralıklı talebi modellemek için salt bir tahmin algoritması yerine **Aşamalı Hibrit (İstatistik + Makine Öğrenmesi)** bir yaklaşım kullanır:

1. **Öznitelik Mühendisliği (Sinyal İşleme):**
   * **ADIDA (Aggregate-Allocate Intermittent Demand Approach):** Talebin zaman periyotlarındaki aralığını yumuşatır.
   * **SBA (Syntetos-Boylan Approximation):** Kesikli taleplerdeki bias (yanlılık) sorununu giderir.
   * **TSB (Teunter-Syntetos-Babai):** Talebin gerçekleşme olasılığını (Probability) ve boyutunu (Size) ayrı ayrı hesaplar.
2. **Nihai Karar Motoru (CatBoost):** * İstatistiksel modellerden gelen pürüzsüz sinyalleri, MAN üretim planı, Lead Time (Tedarik Süresi) ve Parça Ailesi gibi verilerle birleştirir.
   * Simetrik Karar Ağaçları (Oblivious Trees) kullanarak ezberlemeyi (Overfitting) bloke eder.
3. **Hiperparametre Optimizasyonu (Optuna):** * Bayesyen optimizasyon yaklaşımı ile Zaman Serisi Çapraz Doğrulaması (Time-Series Split) yaparak en optimum ağaç derinliği ve öğrenme hızını dinamik olarak bulur.

## 🚀 Temel Özellikler
* **Çoklu BOM Entegrasyonu:** Farklı araç modellerinden (MAN/NEOPLAN) gelen üretim planlarını tek bir C-Sınıfı parçaya (SKU) konsolide eder.
* **XAI (Açıklanabilir Yapay Zeka):** Modelin bir "Kara Kutu" (Black Box) olmasını engeller. Sipariş miktarını belirlerken hangi değişkene yüzde kaç güvendiğini şeffafça raporlar (Feature Importance).
* **Dinamik Risk Yönetimi:** Stoksuz kalma riskini RMSE loss fonksiyonu ile üssel olarak cezalandırarak fabrikanın duruş riskini minimize eder.
* **On-Premise / Cloud Uyumluluğu:** İnternetsiz (Yerel sunucu) ortamda çalışabildiği gibi, Streamlit Community Cloud üzerinde canlı (Deployment) olarak da erişilebilir.

---

## ⚙️ Kurulum ve Çalıştırma (Lokal Sunucu İçin)

Projenin yerel bir makinede (veya fabrika intranetinde) çalıştırılması için aşağıdaki adımları izleyin:

```bash
# 1. Repoyu bilgisayarınıza indirin
git clone [https://github.com/mHanefi/man-siparis-ai.git](https://github.com/mHanefi/man-siparis-ai.git)
cd man-siparis-ai

# 2. Gerekli kütüphaneleri (Dependencies) yükleyin
pip install -r requirements.txt

# 3. Streamlit web sunucusunu başlatın
streamlit run streamlit_app.py
