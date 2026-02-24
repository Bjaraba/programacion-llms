import numpy as np
import pandas as pd

def generar_caso_de_uso_predict_churn():
    np.random.seed(7)
    n_train, n_test = 800, 200

    def make_data(n, seed):
        rng = np.random.RandomState(seed)
        tenure = rng.normal(24, 12, n).clip(1, 72)
        monthly = rng.normal(65, 20, n).clip(20, 120)
        complaints = rng.poisson(1.5, n).astype(float)
        products = rng.randint(1, 5, n).astype(float)
        last_login = rng.exponential(10, n).clip(1, 90)

        # Churn depende lógicamente de las features
        log_odds = (
            -0.05 * tenure
            + 0.02 * monthly
            + 0.30 * complaints
            - 0.40 * products
            + 0.04 * last_login
        )
        prob = 1 / (1 + np.exp(-log_odds))
        churn = (rng.rand(n) < prob).astype(int)

        df = pd.DataFrame(
            {
                "tenure_months": tenure,
                "monthly_charge": monthly,
                "num_complaints": complaints,
                "num_products": products,
                "last_login_days": last_login,
                "churn": churn,
            }
        )
        # Introducir nulos realistas
        for col in ["tenure_months", "monthly_charge", "last_login_days"]:
            idx = rng.choice(n, int(n * 0.05), replace=False)
            df.loc[idx, col] = np.nan
        return df

    df_train = make_data(n_train, seed=7)
    df_test = make_data(n_test, seed=99)

    # ── Salida esperada ──────────────────────────────────────────────
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score

    features = [
        "tenure_months",
        "monthly_charge",
        "num_complaints",
        "num_products",
        "last_login_days",
    ]

    X_train = df_train[features]
    y_train = df_train["churn"]
    X_test = df_test[features]
    y_test = df_test["churn"]

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    importances = dict(zip(features, pipe.named_steps["clf"].feature_importances_))

    expected_output = {
        "predictions": preds,
        "probabilities": probs,
        "accuracy": acc,
        "feature_importances": importances,
    }

    return (df_train, df_test), expected_output
