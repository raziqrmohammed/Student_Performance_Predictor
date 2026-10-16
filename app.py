
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("student_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        previous_score = float(request.form["previous_score"])
        assignment_score = float(request.form["assignment_score"])

        student = pd.DataFrame([{
            "Study Hours": study_hours,
            "Attendance": attendance,
            "Previous Score": previous_score,
            "Assignment Score": assignment_score
        }])

        prediction = round(float(model.predict(student)[0]), 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
