import pandas as pd
from catboost import CatBoostClassifier
from sklearn.multioutput import MultiOutputClassifier


def train(X_train: pd.DataFrame, y_train: pd.DataFrame) -> MultiOutputClassifier:
    cat_model = CatBoostClassifier(n_estimators=100,
                                   depth=8,
                                   learning_rate=1, 
                                   loss_function='Logloss')
    model = MultiOutputClassifier(cat_model, n_jobs=1)
    model.fit(X_train, y_train)
    return model
