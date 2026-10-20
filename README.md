# EduPredict AI — Student Performance Prediction System

## 1. Project Overview

EduPredict AI is a machine learning-based web application that predicts a student's final academic score using study habits, attendance, previous examination performance, and assignment results.

The application provides an interactive dashboard for exploring student records, viewing academic analytics, and generating performance predictions.

## 2. Project Objectives

- Predict students' final scores using machine learning.
- Explore relationships between study habits and academic performance.
- Display student statistics through interactive charts.
- Provide a simple and user-friendly web interface.
- Demonstrate a machine learning workflow, from data preparation to prediction and evaluation.

## 3. Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas
- **Model Serialization:** Joblib
- **Web Application:** Streamlit
- **Data Visualization:** Plotly
- **Development Environment:** Visual Studio Code

## 4. Machine Learning Approach

EduPredict AI uses **Linear Regression** to estimate a student's final academic score.

### Input Features

The model uses six input features:

1. Study Hours
2. Attendance
3. Previous Score
4. Assignment Score
5. Previous Test Average
6. Assignment Completion Rate

### Target Variable

- Final Score

The dataset is divided into training and testing subsets. The model is evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²).

Cross-validation is also performed on the training data to provide an additional estimate of model performance.

## 5. Dataset

The demonstration dataset contains **30 student records and 8 columns**:

- `Student_ID`
- `Study_Hours`
- `Attendance`
- `Previous_Score`
- `Assignment_Score`
- `Final_Score`
- `Previous_Test_Average`
- `Assignment_Completion_Rate`

The last two features are illustrative values generated for this project. They should not be interpreted as verified observations from real students.

**Dataset limitation:** The dataset is small and intended for educational demonstration. Evaluation results may not represent performance on real students or unseen populations.

## 6. Application Features

- Interactive dashboard with student statistics
- Final-score distribution visualization
- Student performance prediction form
- Predicted score visualization
- Academic analytics and charts
- Student record search and filtering
- Dataset preview and CSV download
- Model training and evaluation

## 7. Project Structure

```text
EduPredict-AI/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   └── student_data.csv
├── models/
│   └── student_performance_model.pkl
└── src/
    ├── __init__.py
    ├── add_features.py
    ├── train_model.py
    └── predict.py
```

## 8. Installation and Execution

### Prerequisites

- Python 3.10 or a compatible Python version
- pip
- Visual Studio Code (recommended)

### Step 1: Install Dependencies

Open a terminal in the project directory and run:

```bash
python -m pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python src/train_model.py
```

The trained model is saved in the `models/` directory.

### Step 3: Run a Prediction Test

```bash
python src/predict.py
```

### Step 4: Launch the Web Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## 9. Model Evaluation

The model uses the following evaluation metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted scores.
- **Root Mean Squared Error (RMSE):** Measures prediction error while assigning greater weight to larger errors.
- **R-squared (R²):** Measures how well the model explains variation in the target values on the evaluated data.

Evaluation results from the current 30-record dataset are preliminary. A larger, representative dataset and independent testing are required before drawing conclusions about real-world predictive performance.

## 10. Future Enhancements

- Expand the dataset using reliable, representative student records.
- Compare Linear Regression with other regression algorithms.
- Introduce student performance categories.
- Improve input validation and error handling.
- Add model comparison charts and downloadable reports.
- Evaluate the model on a larger independent test dataset.

## 11. Conclusion

EduPredict AI demonstrates how machine learning can be integrated into a web application to estimate student academic performance. It combines data processing, regression modelling, model evaluation, and interactive visualization in one application.

The project is intended for educational and demonstration purposes and should not be used as the sole basis for academic decisions.
