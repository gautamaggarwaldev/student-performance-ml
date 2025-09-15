# src/data_preprocessing.py
"""
Load UCI student-mat.csv, create a categorical target (Good/Average/Poor),
select a compact set of features, and save cleaned CSV + metadata.
"""

import os
import json
import pandas as pd

# INPUT / OUTPUT
INPUT_CSV = "data/student-mat.csv"    # place the downloaded file here
CLEANED_CSV = "data/cleaned_student.csv"
META_JSON = "artifacts/metadata.json"

# Features we'll use (compact set for easy UI & good predictive power)
NUMERIC_FEATURES = ["age", "studytime", "failures", "absences", "G1", "G2"]
CATEGORICAL_FEATURES = [
    "sex", "schoolsup", "famsup", "paid",
    "activities", "higher", "internet", "romantic"
]

def grade_to_label(g):
    # Custom thresholds — adjust if you prefer different bins
    if g >= 15:
        return "Good"
    elif g >= 10:
        return "Average"
    else:
        return "Poor"

def main(input_path=INPUT_CSV):
    os.makedirs("data", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # read (UCI student csv uses ';' as separator)
    df = pd.read_csv(input_path, sep=";")
    # create target
    df["performance"] = df["G3"].apply(grade_to_label)

    # pick columns
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["performance"]
    df_selected = df[cols].copy()

    # save cleaned csv
    df_selected.to_csv(CLEANED_CSV, index=False)
    print(f"[+] Saved cleaned data -> {CLEANED_CSV}")

    # save metadata: unique values for each categorical feature + numeric names + target classes
    metadata = {}
    for c in CATEGORICAL_FEATURES:
        metadata[c] = sorted(df_selected[c].dropna().unique().tolist())
    metadata["numeric"] = NUMERIC_FEATURES
    metadata["target_classes"] = sorted(df_selected["performance"].unique().tolist())

    with open(META_JSON, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved metadata -> {META_JSON}")

if __name__ == "__main__":
    main()
