from catboost import CatBoostRegressor

class CatBoostModel:
    def __init__(self, cat_features=None):
        self.cat_features = cat_features if cat_features is not None else []
        
        self.model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='RMSE',
            cat_features=self.cat_features,
            verbose=False          
        )

    def train(self, X, y, eval_set=None):
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
        return self.model.get_best_iteration()

    def get_feature_importance(self):
        return self.model.get_feature_importance(prettified=True)