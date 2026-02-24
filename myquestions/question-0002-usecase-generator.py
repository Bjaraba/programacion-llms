import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score


def generador_casos_de_uso_predict_car_price():

    def make_data(n, seed):
        rng = np.random.RandomState(seed)
        km_driven = rng.exponential(60000, n).clip(1000, 300000)
        age_years = rng.randint(1, 20, n).astype(float)
        engine_cc = rng.choice([1000, 1400, 1600, 2000, 2500], n).astype(float)
        num_owners = rng.randint(1, 5, n).astype(float)
        fuel_efficiency = rng.normal(15, 4, n).clip(5, 30)

        price = (
            25000
            - 0.05 * km_driven
            - 800 * age_years
            + 3 * engine_cc
            - 1500 * num_owners
            + 200 * fuel_efficiency
            + rng.normal(0, 2000, n)
        ).clip(1000, None)

        df = pd.DataFrame(
            {
                "km_driven": km_driven,
                "age_years": age_years,
                "engine_cc": engine_cc,
                "num_owners": num_owners,
                "fuel_efficiency": fuel_efficiency,
                "price_usd": price,
            }
        )
        for col in ["km_driven", "fuel_efficiency", "engine_cc"]:
            idx = rng.choice(n, int(n * 0.05), replace=False)
            df.loc[idx, col] = np.nan
        return df

    df_train = make_data(1000, seed=42)
    df_test = make_data(250, seed=17)

    features = ["km_driven", "age_years", "engine_cc", "num_owners", "fuel_efficiency"]
    X_train, y_train = df_train[features], df_train["price_usd"]
    X_test, y_test = df_test[features], df_test["price_usd"]

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10)),
        ]
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    coefs = dict(zip(features, pipe.named_steps["model"].coef_))

    expected_output = {
        "predictions": preds,
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "r2": r2_score(y_test, preds),
        "coefficients": coefs,
    }
    return (df_train, df_test), expected_output
