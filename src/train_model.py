
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student_data.csv"
MODEL_PATH = BASE_DIR / "models" / "student_performance_model.pkl"

FEATURES = [
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Assignment_Score",
    "Previous_Test_Average",
    "Assignment_Completion_Rate",
]
TARGET = "Final_Score"


def main():
    data = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]
    missing = [col for col in required_columns if col not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data[required_columns].apply(pd.to_numeric, errors="coerce")
    data = data.dropna()

    if len(data) < 10:
        raise ValueError("At least 10 complete records are required.")

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()

    folds = min(5, len(X_train))
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring="neg_mean_absolute_error"
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions) if len(y_test) >= 2 else float("nan")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "target": TARGET,
        },
        MODEL_PATH,
    )

    print("\nEduPredict AI - Model Evaluation")
    print("-" * 42)
    print(f"Valid records: {len(data)}")
    print(f"Training records: {len(X_train)}")
    print(f"Test records: {len(X_test)}")
    print(f"Cross-validation MAE: {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")
    print(f"Holdout MAE: {mae:.2f}")
    print(f"Holdout RMSE: {rmse:.2f}")
    print(f"Holdout R-squared: {r2:.3f}")
    print(f"\nModel saved to: {MODEL_PATH}")
    print(
        "\nImportant: The dataset contains only 30 illustrative records. "
        "These metrics are preliminary and do not establish real-world accuracy."
    )


if __name__ == "__main__":
    main()
