
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from recommendation_engine import recommend_careers

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
DATABASE_FILE = ROOT / "skillgraph.db"


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


def validate_profile(data):
    if not isinstance(data, dict):
        raise ValueError("Send profile data as JSON.")

    profile = {
        key: value
        for key, value in data.items()
    }

    required_fields = [
        "Highest_qualification",
        "Current_Academic_Level",
        "Fields_of_Interest",
    ]

    missing = [
        field for field in required_fields
        if not profile.get(field)
    ]

    if missing:
        raise ValueError(
            "Please provide: " + ", ".join(missing)
        )

    if profile["Current_Academic_Level"] not in [
        "Student",
        "Graduate",
        "Working Professional"
    ]:
        raise ValueError("Please select a valid academic level.")

    skills = profile.get("Technical_Skills", [])

    if isinstance(skills, str):
        skills = [
            skill.strip()
            for skill in skills.split(",")
            if skill.strip()
        ]

    if not isinstance(skills, list):
        raise ValueError("Technical skills must be a list.")

    profile["Technical_Skills"] = skills

    return profile


def save_prediction(profile, results):
    with sqlite3.connect(DATABASE_FILE) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions
            (created_at, profile_json, results_json)
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
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        profile = validate_profile(data)

        results = recommend_careers(profile, top_n=5)

        prediction_id = save_prediction(profile, results)

        return jsonify({
            "prediction_id": prediction_id,
            "results": results,
        })

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception as error:
        app.logger.exception("Career recommendation failed")

        return jsonify({
            "error": "Recommendation failed. Please check the server logs."
        }), 500


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
