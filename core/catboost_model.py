from catboost import CatBoostRegressor

class CatBoostModel:
    def __init__(self, cat_features=None):
        # Kategorik (metin bazlı) kolonların listesini alıyoruz
        self.cat_features = cat_features if cat_features is not None else []
        
        self.model = CatBoostRegressor(
            iterations=500,        # Ağaç sayısı (Eski koda göre artırdık çünkü veri büyüyecek)
            learning_rate=0.05,
            depth=6,
            loss_function='RMSE',
            cat_features=self.cat_features,
            verbose=False          # Terminali kirletmemesi için kapalı
        )

    def train(self, X, y):
        # Modeli eğitiyoruz
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)