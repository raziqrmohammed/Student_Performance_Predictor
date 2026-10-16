

# Student Performance Predictor

A Python project that analyzes student data, compares three machine learning models, and provides a small Flask web app for estimating final marks.

## Project files

- `train_model.py` — generates sample student data, creates charts, compares models, and saves a trained model.
- `app.py` — runs the web app and predicts marks from student information.
- `templates/index.html` — the web app’s input form.
- `student_data.csv` — generated sample data.
- `predictions.csv` — actual and predicted marks for the test set.
- `student_model.pkl` — saved model loaded by the web app.
- PNG files — charts produced during training.

## Requirements

Install Python 3 and the required packages:

```bash
pip install flask pandas numpy matplotlib seaborn scikit-learn joblib
```

## Train the model

From the project folder, run:

```bash
python train_model.py
```

The script generates 200 sample student records, compares linear regression, decision tree, and random forest models, and saves charts, predictions, and a model file.

## Run the web app

After training, start the app:

```bash
python app.py
```

Open the local address printed in the terminal (usually `http://127.0.0.1:5000`). Enter study hours per day, attendance percentage, previous score, and assignment score to get a predicted mark.

## Model inputs

The model uses these student details to estimate final marks:

- Study hours per day
- Attendance percentage
- Previous score out of 100
- Assignment score out of 100

The prediction is an estimate based on the generated sample data; it should not be treated as a guaranteed result.

## Note

The training script selects the model with the lowest MAE, but currently saves the last model in its comparison loop instead of the selected best model. The web app loads that saved model.
