from catboost import CatBoostRegressor


class CatBoostModel:
    def __init__(self):
        self.model = CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            verbose=False
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)