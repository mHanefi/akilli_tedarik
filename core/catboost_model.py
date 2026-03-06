import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# ========================================================================================================
# [SİSTEM MİMARİSİ]:
# Bu dosya, MAN Türkiye ERP verilerindeki karmaşık sinyalleri çözen CatBoost (Categorical Boosting)
# makine öğrenmesinin fabrika için özelleştirilmiş mimarisidir.
# 
# NEDEN CATBOOST? 
# 1. Simetrik Ağaçlar: XGBoost veya LightGBM gürültülü cıvata verilerinde
#    asimetrik dallanıp aykırı değerleri ezberlerken, CatBoost ağacın her seviyesinde 
#    aynı kuralı sorarak yapısal bir düzenlileştirme yapar. Ezberlemeye fiziksel olarak kapalıdır.
# 2. Kategorik Veri Gücü: Parça Ailesi gibi metin verilerini One-Hot Encoding ile matrisi şişirip 
#    belleği boğmak yerine, kendi içindeki "Target Encoding" mekanizmasıyla doğrudan işler.
# ========================================================================================================

class CatBoostModel:
    def __init__(self, cat_features=None, **kwargs):
        self.cat_features = cat_features if cat_features is not None else []
        
        # [KAYIP FONKSİYONU (LOSS FUNCTION)]:
        #  Neden Poisson kullanmadık?:
        # "Bizim amacımız sadece istatistiksel doğruluk değil. Fabrikada 100 adetlik bir 
        # tahmin sapması (Stock-out) üretim bandını aksatır. RMSE, doğası gereği hataların karesini 
        # aldığı için büyük sapmaları MAE veya Poisson'a göre çok daha sert cezalandırır. 
        # Modelimizi büyük risklerden kaçınmaya zorlamak için RMSE'yi stratejik olarak seçtik."
        default_params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'RMSE',
            'verbose': False
        }
        
        # Eğer dışarıdan Optuna ile optimize edilmiş parametre gelirse onu, gelmezse varsayılanı kullanır.
        self.params = {**default_params, **kwargs}
        self.params['cat_features'] = self.cat_features
        
        self.model = CatBoostRegressor(**self.params)

    def optimize_hyperparameters(self, X, y, n_trials=15):
        
        #====================================================================================
        #[ZAMAN SERİSİ ÇAPRAZ DOĞRULAMA (TIME-SERIES CROSS VALIDATION)]
        #"K-Fold kullanılmamasının sebebi": 
        #"Zaman serilerinde klasik K-Fold kullanmak, geleceği geçmişe karıştırmaktır (Data Leakage). 
        #Biz burada 'Genişleyen Pencere' mantığıyla çalışan Time-Series Split kullandık. 
        #Model sadece geçmişi görerek geleceği tahmin eder."
        
        #[BAYESYEN OPTİMİZASYON (OPTUNA)]
        #"Parametreleri kafamıza göre yazmadık. Optuna kullanarak Bayesyen Optimizasyon yaptık. 
        #Grid Search gibi körlemesine aramak yerine, her denemeden ders çıkaran bir arama sistematiği inşa ettik."
        #====================================================================================
        
        def objective(trial):
            # Modelin arayacağı optimum parametre uzayı (Ağaç derinliği, Öğrenme hızı, L2 Regülarizasyonu)
            param = {
                'iterations': trial.suggest_int('iterations', 300, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'depth': trial.suggest_int('depth', 4, 8),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'loss_function': 'RMSE',
                'verbose': False,
                'cat_features': self.cat_features
            }
            
            # Zamanın sırasını bozmadan 3 farklı kronolojik kesitte test (Time-Series Split)
            tscv = TimeSeriesSplit(n_splits=3)
            rmse_scores = []
            
            # Sadece Pandas DataFrame ise .iloc kullanılabilir, numpy ise doğrudan indekslenir.
            is_df = isinstance(X, pd.DataFrame)
            y_array = y.values if isinstance(y, pd.Series) else y

            for train_idx, val_idx in tscv.split(X):
                X_tr = X.iloc[train_idx] if is_df else X[train_idx]
                y_tr = y_array[train_idx]
                X_val = X.iloc[val_idx] if is_df else X[val_idx]
                y_val = y_array[val_idx]
                
                temp_model = CatBoostRegressor(**param)
                temp_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=False)
                
                preds = temp_model.predict(X_val)
                score = np.sqrt(mean_squared_error(y_val, preds))
                rmse_scores.append(score)
                
            # 3 farklı zaman kesitindeki hataların ortalamasını en aza indirmeye çalışır
            return np.mean(rmse_scores)

        # Optuna motorunu ateşle (Optimizasyon yönü: Minimize)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Bulunan en mükemmel parametreleri sınıfın kendisine kalıcı olarak ata
        best_params = study.best_params
        best_params['loss_function'] = 'RMSE'
        best_params['verbose'] = False
        best_params['cat_features'] = self.cat_features
        self.params = best_params
        self.model = CatBoostRegressor(**self.params)
        
        return best_params

    def train(self, X, y, eval_set=None):
        
        #[OVERFITTING (EZBERLEME) ENGELLEYİCİ]:
        #Eğer eval_set (validasyon) verilmişse, "Early Stopping" (Erken Durdurma) devreye girer.
        #Model test setindeki başarın 50 ağaç boyunca artmadığını görürse, eğitimi anında durdurur
        #Bu sayede model 'gürültüyü' ezberlemez, matematiği öğrenir.
        
        if eval_set is not None:
            self.model.fit(
                X, y,
                eval_set=eval_set,
                early_stopping_rounds=50,
                use_best_model=True
            )
        else:
            self.model.fit(X, y)

    def predict(self, X): 
        return self.model.predict(X)
        
    def get_best_iteration(self):

        # Modelin "Early Stopping" ile kaçıncı ağaçta durduğunu raporlar. (Sistematik Verimlilik)
        return self.model.get_best_iteration()

    def get_feature_importance(self):
        
        return self.model.get_feature_importance(prettified=True)