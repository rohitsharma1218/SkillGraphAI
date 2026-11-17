import json
import sqlite3
from datetime import datetime
from pathlib import Path
from ml_utils import flatten_column
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "career_model.joblib"
DATABASE_FILE = ROOT / "skillgraph.db"

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        "Model not found. Run train_model.py before starting the Flask app."
    )

model_bundle = joblib.load(MODEL_FILE)
model = model_bundle["model"]
FEATURE_COLUMNS = model_bundle["feature_columns"]

TEXT_FIELDS = {
    "Technical_Skills",
    "Soft_Skills",
    "Languages_Known",
    "Certifications",
    "Skills_Gained",
}

REQUIRED_FIELDS = {
    "Highest_qualification",
    "Stream",
    "Current_Academic_Level",
    "Grade_CGPA_Percentage",
    "Fields_of_Interest",
    "Preferred_Work_Style",
    "Work_Type_Interest",
    "Past_Jobs_Internships",
    "Achievements",
    "Willing_to_Relocate",
}


def create_database():
    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                results_json TEXT NOT NULL
            )
            """
        )


def clean_value(value):
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())

    return str(value or "").strip()


def validate_profile(data):
    profile = {
        column: clean_value(data.get(column))
        for column in FEATURE_COLUMNS
    }

    missing = [
        field for field in REQUIRED_FIELDS
        if not profile.get(field)
    ]

    if missing:
        raise ValueError(
            "Missing required field(s): " + ", ".join(sorted(missing))
        )

    grade = float(profile["Grade_CGPA_Percentage"])

    if not 0 <= grade <= 100:
        raise ValueError("Grade_CGPA_Percentage must be between 0 and 100.")

    profile["Grade_CGPA_Percentage"] = grade

    if not profile["Technical_Skills"]:
        raise ValueError("Select at least one technical skill.")

    return profile


def predict_careers(profile):
    input_data = pd.DataFrame([profile], columns=FEATURE_COLUMNS)

    probabilities = model.predict_proba(input_data)[0]
    labels = model.classes_
    top_indexes = probabilities.argsort()[-3:][::-1]

    return [
        {
            "rank": rank,
            "career": labels[index],
            "confidence": round(float(probabilities[index] * 100), 2),
        }
        for rank, index in enumerate(top_indexes, start=1)
    ]


def save_prediction(profile, results):
    with sqlite3.connect(DATABASE_FILE) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (created_at, profile_json, results_json)
            VALUES (?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                json.dumps(profile),
                json.dumps(results),
            ),
        )
        return cursor.lastrowid


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        profile={
            "cgpa": "",
            "skills": "",
            "projects": "",
            "internships": "",
            "certifications": "",
        },
        error=None,
        submitted=False,
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not isinstance(data, dict):
            raise ValueError("Send profile data as JSON.")

        profile = validate_profile(data)
        results = predict_careers(profile)
        prediction_id = save_prediction(profile, results)

        return jsonify(
            {
                "prediction_id": prediction_id,
                "results": results,
            }
        )

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        return jsonify(
            {"error": "Prediction failed. Check the submitted profile data."}
        ), 500


@app.route("/api/predictions", methods=["GET"])
def prediction_history():
    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, created_at, results_json
            FROM predictions
            ORDER BY id DESC
            LIMIT 10
            """
        ).fetchall()

    history = [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "results": json.loads(row["results_json"]),
        }
        for row in rows
    ]

    return jsonify(history)


if __name__ == "__main__":
    create_database()
    app.run(debug=True)