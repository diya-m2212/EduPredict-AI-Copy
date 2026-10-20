
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "student_performance_model.pkl"

FEATURES = [
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Assignment_Score",
    "Previous_Test_Average",
    "Assignment_Completion_Rate",
]


def predict_performance(
    study_hours,
    attendance,
    previous_score,
    assignment_score,
    previous_test_average,
    assignment_completion_rate,
):
    values = {
        "Study_Hours": float(study_hours),
        "Attendance": float(attendance),
        "Previous_Score": float(previous_score),
        "Assignment_Score": float(assignment_score),
        "Previous_Test_Average": float(previous_test_average),
        "Assignment_Completion_Rate": float(assignment_completion_rate),
    }

    if not 0 <= values["Study_Hours"] <= 24:
        raise ValueError("Study hours must be between 0 and 24.")

    for feature in FEATURES:
        if feature != "Study_Hours" and not 0 <= values[feature] <= 100:
            raise ValueError(f"{feature} must be between 0 and 100.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Train the model first using: python src/train_model.py"
        )

    bundle = joblib.load(MODEL_PATH)

    # Support both a saved model bundle and a directly saved model.
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        model_features = bundle.get("features", FEATURES)
    else:
        model = bundle
        model_features = FEATURES

    input_df = pd.DataFrame(
        [[values[feature] for feature in model_features]],
        columns=model_features,
    )

    prediction = float(model.predict(input_df)[0])

    return max(0.0, min(100.0, prediction))


if __name__ == "__main__":
    score = predict_performance(
        study_hours=4,
        attendance=80,
        previous_score=70,
        assignment_score=75,
        previous_test_average=70,
        assignment_completion_rate=85,
    )

    print(f"Predicted final score: {score:.2f}/100")
