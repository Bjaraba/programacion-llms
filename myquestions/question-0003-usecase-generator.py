import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, confusion_matrix

def generador_caso_de_uso_classify_wine_quality():

    def make_data(n, seed):
        rng = np.random.RandomState(seed)
        alcohol   = rng.normal(10.5, 1.2, n).clip(8, 15)
        acidity   = rng.normal(7.5, 1.5, n).clip(4, 12)
        sugar     = rng.exponential(5, n).clip(1, 30)
        ph        = rng.normal(3.3, 0.2, n).clip(2.8, 4.0)
        sulphates = rng.normal(0.65, 0.17, n).clip(0.3, 1.5)
        density   = rng.normal(0.996, 0.003, n).clip(0.988, 1.005)

        log_odds = (
            +0.8  * (alcohol - 10.5)
            -0.3  * (acidity - 7.5)
            -0.1  * sugar
            +0.5  * (ph - 3.3)
            +1.0  * (sulphates - 0.65)
            -2.0  * (density - 0.996) * 100
        )
        prob    = 1 / (1 + np.exp(-log_odds))
        quality = (rng.rand(n) < prob).astype(int)

        df = pd.DataFrame({
            "alcohol": alcohol, "acidity": acidity, "sugar": sugar,
            "ph": ph, "sulphates": sulphates, "density": density,
            "quality": quality
        })
        for col in ["sugar", "sulphates", "acidity"]:
            idx = rng.choice(n, int(n * 0.04), replace=False)
            df.loc[idx, col] = np.nan
        return df

    df_train = make_data(900, seed=3)
    df_test  = make_data(200, seed=55)

    features = ["alcohol", "acidity", "sugar", "ph", "sulphates", "density"]
    X_train, y_train = df_train[features], df_train["quality"]
    X_test,  y_test  = df_test[features],  df_test["quality"]

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]

    expected_output = {
        "predictions":     preds,
        "probabilities":   probs,
        "f1_score":        f1_score(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds)
    }
    return (df_train, df_test), expected_output

