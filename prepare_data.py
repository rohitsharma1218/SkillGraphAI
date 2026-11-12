import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "data" / "raw" / "Career_Path_Dataset_with_Career.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
CLEAN_FILE = OUTPUT_DIR / "cleaned_career_dataset.csv"
OPTIONS_FILE = OUTPUT_DIR / "skill_options.csv"

TARGET = "Suggested_Career_Path"
DROP_COLUMNS = {"Age", "Gender", "Location"}

SKILL_FIELDS = {
    "Technical skill": "Technical_Skills",
    "Soft skill": "Soft_Skills",
    "Language": "Languages_Known",
    "Certification": "Certifications",
    "Skill gained": "Skills_Gained",
}

EMPTY_VALUES = {"", "none", "n/a", "na", "null", "nan"}


def clean_text(value):
    return " ".join((value or "").split())


def split_options(value):
    return [
        clean_text(item)
        for item in re.split(r"\s*,\s*", value or "")
        if clean_text(item).casefold() not in EMPTY_VALUES
    ]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_FILE.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []

        required = {TARGET, *SKILL_FIELDS.values()}
        missing = required.difference(columns)
        if missing:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

        cleaned_rows = []
        seen_rows = set()
        options = {}

        for row in reader:
            cleaned = {
                column: clean_text(row.get(column))
                for column in columns
            }

            if cleaned[TARGET].casefold() in EMPTY_VALUES:
                continue

            row_key = tuple(cleaned[column] for column in columns)
            if row_key in seen_rows:
                continue
            seen_rows.add(row_key)

            cleaned_rows.append(cleaned)

            for category, column in SKILL_FIELDS.items():
                for option in split_options(cleaned[column]):
                    key = (category, option.casefold())
                    options.setdefault(key, (category, option))

    output_columns = [
        column for column in columns
        if column not in DROP_COLUMNS
    ]

    with CLEAN_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_columns)
        writer.writeheader()
        for row in cleaned_rows:
            writer.writerow({key: row[key] for key in output_columns})

    with OPTIONS_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["category", "option"])
        for category, option in sorted(options.values()):
            writer.writerow([category, option])

    career_labels = {
        row[TARGET] for row in cleaned_rows
    }

    print(f"Cleaned rows: {len(cleaned_rows)}")
    print(f"Career labels: {len(career_labels)}")
    print(f"Checkbox options: {len(options)}")
    print(f"Saved: {CLEAN_FILE}")
    print(f"Saved: {OPTIONS_FILE}")


if __name__ == "__main__":
    main()