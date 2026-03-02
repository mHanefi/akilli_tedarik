from catboost import CatBoostRegressor

# ========================================================================================================
# Bu dosya, sistemin ANA BEYNİ olan ve "Gradient Boosting" (Eğimi Artırma) mantığıyla çalışan
# CatBoost (Categorical Boosting) makine öğrenmesi motorunun sıfırdan konfigüre edilmiş mimarisidir.
# 
# NEDEN XGBOOST VEYA LİGHTGBM DEĞİL DE CATBOOST?
# 1. Simetrik Ağaçlar (Oblivious Trees): CatBoost, diğer algoritmalar gibi rastgele asimetrik dallanmaz.
#    Ağacın her seviyesinde aynı kuralı uygulayarak yapısal "Ezberlemeyi (Overfitting)" fiziksel olarak engeller.
# 2. Kategorik Veri Gücü: Parça Ailesi, Araç Modeli gibi metin verilerini (Categorical Features) 
#    "One-Hot Encoding" ile yüzlerce kolona bölüp sistemi şişirmek yerine, kendi içinde matematiksel 
#    olarak çözer (Target Encoding). Sistem inanılmaz hızlı ve hafiftir!
# ========================================================================================================

class CatBoostModel:
    def __init__(self, cat_features=None):
        self.cat_features = cat_features if cat_features is not None else []
        
        # Sistemin körlemesine değil, mühendislik sınırları içinde çalışmasını sağlayan ayarlar:
        self.model = CatBoostRegressor(
            iterations=1000,       # Maksimum 1000 karar ağacı kurmasına izin veriyoruz.
            learning_rate=0.05,    # Öğrenme Hızı: Düşük tutarak (0.05) acele edip hataya düşmesini (Overshooting) engelliyoruz.
            depth=6,               # Ağaç Derinliği: Her ağaç sadece 6 soru sorabilir. (Aşırı derin ağaç = Ezberleme / Overfitting)
            loss_function='RMSE',  # Optimizasyon Hedefi: Büyük hataları sert cezalandıran RMSE fonksiyonu.
            cat_features=self.cat_features,
            verbose=False          # Eğitim sırasında terminali kirletmesin (Sessiz mod).
        )

    def train(self, X, y, eval_set=None):
        
        #OVERFITTING (EZBERLEME) ENGELLEYİCİ]:
        #Eğer modele 1000 ağaç kur dersen ve onu denetlemezsen, 200. ağaçtan sonra 
        #öğrenecek bir şey bulamayıp verideki "gürültüyü/hataları" ezberlemeye başlar (Boosting Overfit).
        
        if eval_set is not None:
            self.model.fit(
                X, y,
                eval_set=eval_set, # Modele, kendini test etmesi için ayrılmış '%20'lik sınav kağıdını' veriyoruz.
                
                # ERKEN DURDURMA
                # "Eğer 50 ağaç boyunca test setindeki başarın gram artmıyorsa, eğitimi DERHAL KES!"
                # Böylece model asla veriyi ezberlemiyor, en optimum yerde durmayı biliyor.
                early_stopping_rounds=50,
                
                # Ve durduğu yerdeki değil, geçmişteki 'En Yüksek Başarıyı Sağladığı' ağacı sisteme kaydediyor.
                use_best_model=True
            )
        else:
            # Nihai model eğitilirken tüm veriyle çalışması için yedek blok.
            self.model.fit(X, y)

    def predict(self, X): 
        # Modeli canlıya (Production) aldığımızda MRP planlarını okuyup sipariş üreten fonksiyon.
        return self.model.predict(X)
        
    def get_best_iteration(self):
        # [SİSTEMATİK VERİMLİLİK]:
        # Üstteki "Early Stopping" sayesinde model örneğin 142. ağaçta eğitimi kestiyse, 
        # bu fonksiyon bize o "142" sayısını verir. Böylece final modelini eğitirken
        # körlemesine 1000 ağaç değil, tam olarak ihtiyaç duyulan '142' ağaç kurarız
        return self.model.get_best_iteration()

    def get_feature_importance(self):
        # ====================================================================================
        # "Model karar veriyor ama bu bir kapalı kutu, güvenemeyiz" geçerli bir soru değildir
        # Bu fonksiyon, modelin "Ben bu tahmini yaparken yüzde kaç ADIDA'ya, yüzde kaç TGE üretimine, 
        # yüzde kaç SBA'ya baktım?" sorusunun cevabını (Öznitelik Ağırlıklarını) dışarı verir. 
        # Sistemin "Açıklanabilir, Şeffaf ve Güvenilir" (White-Box) olduğunu kanıtlar!
        # ==================================================================================== 
        return self.model.get_feature_importance(prettified=True)