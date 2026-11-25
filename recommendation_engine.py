
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAREER_FILE = ROOT / "career_data" / "career_knowledge.json"


def load_careers():
    with open(CAREER_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["careers"]

def calculate_skill_match(user_skills, career_skills):
    user_skills = {
        skill.strip().lower() for skill in user_skills if skill.strip()
    }

    career_skills = {
        skill.strip().lower() for skill in career_skills
    }

    if not career_skills:
        return 0, []

    matched_skills = user_skills.intersection(career_skills)
    missing_skills = career_skills - user_skills

    match_percentage = round(
        (len(matched_skills) / len(career_skills)) * 100, 2
    )

    return match_percentage, sorted(missing_skills)

def calculate_interest_match(user_interest, career_category):
    if not user_interest:
        return 0

    user_interest = user_interest.strip().lower()
    career_category = career_category.strip().lower()

    if user_interest == career_category:
        return 100

    return 0

def calculate_education_match(user_qualification, career_qualifications):
    if not user_qualification:
        return 50

    user_qualification = user_qualification.strip().lower()

    career_qualifications = [
        qualification.lower()
        for qualification in career_qualifications
    ]

    if user_qualification in career_qualifications:
        return 100

    return 40

def convert_to_list(value):
    """Convert a string or list into a clean list of values."""
    if not value:
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    return [
        item.strip()
        for item in str(value).replace(";", ",").split(",")
        if item.strip()
    ]


def get_stage_guidance(academic_level):
    """Provide guidance based on the user's current career stage."""

    stage = str(academic_level).strip().lower()

    if "student" in stage:
        return [
            "Build strong programming and fundamental skills.",
            "Complete practical projects related to your target career.",
            "Look for internships, workshops and hackathons."
        ]

    elif "graduate" in stage:
        return [
            "Build a professional portfolio and resume.",
            "Work on industry-level projects.",
            "Apply for entry-level jobs and internships."
        ]

    elif "professional" in stage:
        return [
            "Strengthen advanced and specialized skills.",
            "Build experience with real-world projects.",
            "Explore certifications and career advancement opportunities."
        ]

    return [
        "Build relevant technical skills.",
        "Complete practical projects.",
        "Explore learning and career opportunities."
    ]


def recommend_careers(profile, top_n=5):
    """Generate career recommendations using profile matching."""

    careers = load_careers()

    user_skills = convert_to_list(
        profile.get("Technical_Skills", [])
    )

    user_interest = profile.get("Fields_of_Interest", "")

    user_qualification = profile.get(
        "Highest_qualification", ""
    )

    academic_level = profile.get(
        "Current_Academic_Level", ""
    )

    recommendations = []

    for career in careers:

        skill_score, missing_skills = calculate_skill_match(
            user_skills,
            career["technical_skills"]
        )

        interest_score = calculate_interest_match(
            user_interest,
            career["category"]
        )

        education_score = calculate_education_match(
            user_qualification,
            career["qualifications"]
        )

        # Weighted recommendation score
        final_score = (
            skill_score * 0.60
            + interest_score * 0.25
            + education_score * 0.15
        )

        matched_skills = sorted(
            set(skill.lower() for skill in user_skills)
            & set(skill.lower() for skill in career["technical_skills"])
        )

        recommendations.append({
            "career": career["name"],
            "category": career["category"],
            "description": career["description"],
            "match_percentage": round(final_score, 2),
            "skill_match": skill_score,
            "interest_match": interest_score,
            "education_match": education_score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "recommended_learning": career["recommended_learning"],
            "guidance": get_stage_guidance(academic_level)
        })

    recommendations.sort(
        key=lambda item: item["match_percentage"],
        reverse=True
    )

    return recommendations[:top_n]
