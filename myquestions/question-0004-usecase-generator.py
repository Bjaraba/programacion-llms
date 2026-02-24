import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

def generador_caso_de_uso_segment_and_predict():

    def make_data(n, seed):
        rng = np.random.RandomState(seed)
        age = rng.normal(38, 12, n).clip(18, 75)
        income = rng.exponential(40000, n).clip(10000, 200000)
        visits = rng.poisson(6, n).astype(float).clip(1, 30)
        avg_basket = rng.normal(55, 20, n).clip(10, 150)
        loyalty_years = rng.exponential(3, n).clip(0, 20)

        monthly_spend = (
            0.002 * income
            + 12 * visits
            + 0.8 * avg_basket
            + 50 * loyalty_years
            + rng.normal(0, 80, n)
        ).clip(20, None)

        df = pd.DataFrame(
            {
                "age": age,
                "income_usd": income,
                "visits_per_month": visits,
                "avg_basket_usd": avg_basket,
                "loyalty_years": loyalty_years,
                "monthly_spend_usd": monthly_spend,
            }
        )
        for col in ["income_usd", "avg_basket_usd", "loyalty_years"]:
            idx = rng.choice(n, int(n * 0.05), replace=False)
            df.loc[idx, col] = np.nan
        return df

    df_train = make_data(1000, seed=8)
    df_test = make_data(250, seed=21)

    features = [
        "age",
        "income_usd",
        "visits_per_month",
        "avg_basket_usd",
        "loyalty_years",
    ]
    target = "monthly_spend_usd"

    # Imputar y escalar para clustering
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(df_train[features])
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_imp = imputer.transform(df_test[features])
    X_test_sc = scaler.transform(X_test_imp)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    train_segments = km.fit_predict(X_train_sc)
    test_segments = km.predict(X_test_sc)

    # Ridge por segmento
    predictions = np.zeros(len(df_test))
    rmse_per_seg = {}
    segment_sizes = {}

    for seg in range(3):
        tr_idx = np.where(train_segments == seg)[0]
        te_idx = np.where(test_segments == seg)[0]
        segment_sizes[seg] = len(tr_idx)

        if len(te_idx) == 0:
            rmse_per_seg[seg] = None
            continue

        model = Ridge(alpha=10)
        model.fit(X_train_sc[tr_idx], df_train[target].iloc[tr_idx])
        preds_seg = model.predict(X_test_sc[te_idx])
        predictions[te_idx] = preds_seg
        rmse_per_seg[seg] = np.sqrt(
            mean_squared_error(df_test[target].iloc[te_idx], preds_seg)
        )

    expected_output = {
        "segments_test": test_segments,
        "predictions": predictions,
        "rmse_per_segment": rmse_per_seg,
        "segment_sizes": segment_sizes,
    }
    return (df_train, df_test), expected_output
