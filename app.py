from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    profile = {
        "cgpa": "",
        "skills": "",
        "projects": "",
        "internships": "",
        "certifications": "",
    }
    error = None
    submitted = False

    if request.method == "POST":
        profile = {
            "cgpa": request.form.get("cgpa", "").strip(),
            "skills": request.form.get("skills", "").strip(),
            "projects": request.form.get("projects", "").strip(),
            "internships": request.form.get("internships", "").strip(),
            "certifications": request.form.get("certifications", "").strip(),
        }

        try:
            cgpa = float(profile["cgpa"])
            projects = int(profile["projects"])
            internships = int(profile["internships"])
            certifications = int(profile["certifications"])

            if not 0 <= cgpa <= 10:
                raise ValueError("CGPA must be between 0 and 10.")
            if min(projects, internships, certifications) < 0:
                raise ValueError("Project, internship, and certification counts cannot be negative.")
            if not profile["skills"]:
                raise ValueError("Enter at least one technical skill.")

            submitted = True

        except ValueError as exc:
            error = str(exc) or "Enter valid numbers for CGPA and activity counts."

    return render_template(
        "index.html",
        profile=profile,
        error=error,
        submitted=submitted,
    )


if __name__ == "__main__":
    app.run(debug=True)