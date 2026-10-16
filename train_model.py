
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 1: Create sample student dataset
np.random.seed(42)
n = 200

study_hours = np.random.uniform(1, 8, n)
attendance = np.random.uniform(50, 100, n)
previous_score = np.random.uniform(30, 95, n)
assignment_score = np.random.uniform(35, 100, n)

noise = np.random.normal(0, 5, n)

final_marks = (
    2.5 * study_hours
    + 0.20 * attendance
    + 0.30 * previous_score
    + 0.25 * assignment_score
    + noise
)

final_marks = np.clip(final_marks, 0, 100)

df = pd.DataFrame({
    "Study Hours": study_hours.round(1),
    "Attendance": attendance.round(1),
    "Previous Score": previous_score.round(1),
    "Assignment Score": assignment_score.round(1),
    "Final Marks": final_marks.round(1)
})

# Save dataset as CSV
df.to_csv("student_data.csv", index=False)

print("Dataset created successfully!")
print(df.head())

# STEP 2: Check and handle missing values
print("\nMissing values:")
print(df.isnull().sum())

# STEP 3: Exploratory Data Analysis (EDA)

# Distribution of final marks
plt.figure(figsize=(7, 4))
sns.histplot(df["Final Marks"], bins=15, kde=True)
plt.title("Distribution of Final Marks")
plt.savefig("marks_distribution.png")
plt.show()

# Study hours vs final marks
plt.figure(figsize=(7, 4))
sns.scatterplot(data=df, x="Study Hours", y="Final Marks")
plt.title("Study Hours vs Final Marks")
plt.savefig("study_hours_vs_marks.png")
plt.show()

# Attendance vs final marks
plt.figure(figsize=(7, 4))
sns.scatterplot(data=df, x="Attendance", y="Final Marks")
plt.title("Attendance vs Final Marks")
plt.savefig("attendance_vs_marks.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Student Performance Correlation")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# STEP 4: Select input features and target
X = df[
    ["Study Hours", "Attendance",
     "Previous Score", "Assignment Score"]
]
y = df["Final Marks"]

# STEP 5: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 6: Build and compare ML models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=5, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}

results = []
trained_models = {}

for name, model in models.items():
    # Imputer handles missing feature values if present
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"), model
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "R2 Score": round(r2, 3)
    })

    trained_models[name] = pipeline

# STEP 7: Display evaluation results
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

# Choose the model with the lowest MAE
best_name = results_df.loc[
    results_df["MAE"].idxmin(), "Model"
]
best_model = trained_models[best_name]

print("\nBest model based on MAE:", best_name)

# STEP 8: Compare actual vs predicted marks
predictions = best_model.predict(X_test)

comparison = pd.DataFrame({
    "Actual Marks": y_test.values,
    "Predicted Marks": np.round(predictions, 1)
})

print("\nActual vs Predicted Marks:")
print(comparison.head(10).to_string(index=False))

comparison.to_csv("predictions.csv", index=False)

plt.figure(figsize=(6, 5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

# STEP 9: Predict marks for a new student
new_student = pd.DataFrame([{
    "Study Hours": 5,
    "Attendance": 90,
    "Previous Score": 75,
    "Assignment Score": 80
}])

predicted_marks = best_model.predict(new_student)[0]

print("\nNew Student Details:")
print(new_student.to_string(index=False))
print(f"Predicted Final Marks: {predicted_marks:.2f} / 100")

joblib.dump(model, "student_model.pkl")
print("Trained model saved successfully!")
