# src/train_model.py
"""
Train a RandomForest classifier pipeline (preprocessing + classifier),
evaluate it, save pipeline and some artifacts (confusion matrix & feature importances).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Paths
CLEANED_CSV = "data/cleaned_student.csv"
META_JSON = "artifacts/metadata.json"
MODEL_OUT = "models/student_performance_pipeline.joblib"
FI_OUT = "artifacts/feature_importances.csv"
CM_OUT = "artifacts/confusion_matrix.png"

# Make dirs
os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# Load metadata
with open(META_JSON, "r") as f:
    metadata = json.load(f)

numeric_features = metadata["numeric"]
# categorical feature names are keys in metadata excluding 'numeric' and 'target_classes'
categorical_features = [k for k in metadata.keys() if k not in ("numeric", "target_classes")]

# Load data
df = pd.read_csv(CLEANED_CSV)
X = df[numeric_features + categorical_features]
y = df["performance"]

# Train-test split with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Classifier
clf = RandomForestClassifier(n_estimators=200, random_state=42)

# Full pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", clf)
])

# Train
print("[*] Training model...")
pipeline.fit(X_train, y_train)
print("[+] Training complete.")

# Evaluate
y_pred = pipeline.predict(X_test)
print("\n=== Test accuracy ===")
print(accuracy_score(y_test, y_pred))
print("\n=== Classification report ===")
print(classification_report(y_test, y_pred))

# Confusion matrix (save figure)
clf_obj = pipeline.named_steps["clf"]
classes = clf_obj.classes_

cm = confusion_matrix(y_test, y_pred, labels=classes)
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(cm, interpolation="nearest")
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center")
fig.tight_layout()
plt.savefig(CM_OUT)
plt.close()
print(f"[+] Saved confusion matrix -> {CM_OUT}")

# Feature importances
ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
feature_names = numeric_features + cat_feature_names
importances = pipeline.named_steps["clf"].feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False)
fi_df.to_csv(FI_OUT, index=False)
print(f"[+] Saved feature importances -> {FI_OUT}")

# Save pipeline
joblib.dump(pipeline, MODEL_OUT)
print(f"[+] Saved pipeline model -> {MODEL_OUT}")
