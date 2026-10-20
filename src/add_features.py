
from pathlib import Path
import random
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student_data.csv"

# Load the existing dataset
data = pd.read_csv(DATA_PATH)

new_columns = [
    "Previous_Test_Average",
    "Assignment_Completion_Rate",
]

# Avoid adding the same columns twice
if all(column in data.columns for column in new_columns):
    print("Both new features already exist. No changes made.")
else:
    rng = random.Random(42)

    if "Previous_Test_Average" not in data.columns:
        data["Previous_Test_Average"] = [
            round(max(0, min(100, score + rng.uniform(-8, 8))), 1)
            for score in data["Previous_Score"]
        ]

    if "Assignment_Completion_Rate" not in data.columns:
        data["Assignment_Completion_Rate"] = [
            round(rng.uniform(55, 100), 1)
            for _ in range(len(data))
        ]

    data.to_csv(DATA_PATH, index=False)

    print("Dataset updated successfully!")
    print(f"Total records: {len(data)}")
    print("Columns:", list(data.columns))
    print("\nFirst five records:")
    print(data.head().to_string(index=False))
    print(
        "\nNote: The two added columns contain illustrative "
        "values, not real student observations."
    )
