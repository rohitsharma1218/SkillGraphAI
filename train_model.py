import json
from pathlib import Path
import numpy as np

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "processed" / "cleaned_career_dataset.csv"
MODEL_DIR = ROOT / "models"

TARGET_COLUMN = "Suggested_Career_Path"

TEXT_COLUMNS = [
    "Technical_Skills",
    "Soft_Skills",
    "Languages_Known",
    "Certifications",
    "Fields_of_Interest",
    "Skills_Gained",
]

NUMERIC_COLUMNS = [
    "Grade_CGPA_Percentage",
]

CATEGORICAL_COLUMNS = [
    "Highest_qualification",
    "Stream",
    "Current_Academic_Level",
    "Preferred_Work_Style",
    "Work_Type_Interest",
    "Past_Jobs_Internships",
    "Achievements",
    "Willing_to_Relocate",
]


def text_pipeline():
    return Pipeline(
        steps=[
            ("fill_missing", SimpleImputer(strategy="constant", fill_value="")),
            ("flatten", FunctionTransformer(flatten_column, validate=False)),
            ("vectorizer", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                token_pattern=r"(?u)\b[\w+#.]+\b",
            )),
        ]
    )


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE).fillna("")
    data = data[data[TARGET_COLUMN].astype(str).str.strip() != ""].copy()

    feature_columns = TEXT_COLUMNS + NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    X = data[feature_columns]
    y = data[TARGET_COLUMN].astype(str).str.strip()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    transformers = [
        (
            column.lower(),
            text_pipeline(),
            [column],
        )
        for column in TEXT_COLUMNS
    ]

    transformers.extend(
        [
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", ColumnTransformer(transformers=transformers)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    top_3_indexes = np.argsort(probabilities, axis=1)[:, -3:]
    top_3_predictions = model.classes_[top_3_indexes]

    top_3_correct = sum(
    actual in suggested
    for actual, suggested in zip(y_test, top_3_predictions)
    )

    top_3_accuracy = top_3_correct / len(y_test)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(y_test, predictions, zero_division=0)

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "target_column": TARGET_COLUMN,
        },
        MODEL_DIR / "career_model.joblib",
    )

    metrics = {
        "dataset_rows": len(data),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "career_labels": sorted(y.unique().tolist()),
        "accuracy": round(float(accuracy), 4),
        "weighted_precision": round(float(precision), 4),
        "weighted_recall": round(float(recall), 4),
        "weighted_f1_score": round(float(f1), 4),
        "top_3_accuracy": round(float(top_3_accuracy), 4),
    }

    with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with (MODEL_DIR / "classification_report.txt").open("w", encoding="utf-8") as file:
        file.write(report)

    print("Training complete.")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Weighted F1-score: {f1:.2%}")
    print(f"Saved model: {MODEL_DIR / 'career_model.joblib'}")
    print(f"Saved metrics: {MODEL_DIR / 'metrics.json'}")
    print(f"Top-3 accuracy: {top_3_accuracy:.2%}")

if __name__ == "__main__":
    main()

