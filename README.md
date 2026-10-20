

\# EduPredict AI — Student Performance Prediction System



\## 1. Project Overview



EduPredict AI is a machine learning-based web application that predicts a student's final academic score using study hours, attendance, previous examination scores, and assignment scores.



The application provides an interactive dashboard for viewing student data, exploring academic analytics, and predicting student performance.



\## 2. Project Objectives



\- Predict student final scores using machine learning.

\- Explore the relationship between study habits and academic performance.

\- Display student statistics through interactive charts.

\- Provide a simple and user-friendly web interface.

\- Demonstrate a complete machine learning workflow, from data preparation to prediction.



\## 3. Technologies Used



\- \*\*Programming Language:\*\* Python

\- \*\*Machine Learning:\*\* Scikit-learn

\- \*\*Data Processing:\*\* Pandas

\- \*\*Model Storage:\*\* Joblib

\- \*\*Web Application:\*\* Streamlit

\- \*\*Data Visualization:\*\* Plotly

\- \*\*Development Environment:\*\* Visual Studio Code



\## 4. Machine Learning Approach



EduPredict AI uses Linear Regression to estimate a student's final score.



\### Input Features



1\. Study Hours

2\. Attendance Percentage

3\. Previous Examination Score

4\. Assignment Score



\### Target Variable



\- Final Score



The dataset is divided into training and testing subsets. The model is trained on the training data and evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²).



Five-fold cross-validation is also used on the training subset to obtain an additional estimate of model performance.



\## 5. Dataset



The current dataset contains 30 student records and six columns:



\- Student\_ID

\- Study\_Hours

\- Attendance

\- Previous\_Score

\- Assignment\_Score

\- Final\_Score



The dataset has been checked for missing values and duplicate rows.



\*\*Dataset limitation:\*\* The current dataset is small and should be treated as a demonstration dataset. Evaluation results may not represent performance on real students or unseen populations.



\## 6. Application Features



\- Interactive dashboard with student statistics

\- Final-score distribution visualization

\- Student performance prediction form

\- Predicted score visualization

\- Academic analytics and charts

\- Dataset preview

\- Model evaluation during training



\## 7. Project Structure



```text

EduPredict-AI/

├── app.py

├── README.md

├── requirements.txt

├── data/

│   └── student\_data.csv

├── models/

│   └── student\_performance\_model.pkl

├── src/

│   ├── \_\_init\_\_.py

│   ├── train\_model.py

│   └── predict.py

├── assets/

└── venv/

```



\## 8. Installation and Execution



\### Prerequisites



\- Python 3.10 or a compatible Python version

\- pip

\- Visual Studio Code (recommended)



\### Install Dependencies



Open a terminal in the project directory and run:



```bash

python -m pip install -r requirements.txt

```



\### Train the Model



```bash

python src/train\_model.py

```



The trained model is saved to the `models/` directory.



\### Run a Prediction Test



```bash

python src/predict.py

```



\### Launch the Web Application



```bash

python -m streamlit run app.py

```



Open the local URL displayed in the terminal, usually:



```text

http://localhost:8501

```



\## 9. Model Evaluation



The model is evaluated using:



\- \*\*Mean Absolute Error (MAE):\*\* Average absolute difference between actual and predicted scores.

\- \*\*Root Mean Squared Error (RMSE):\*\* Measures prediction error while giving larger errors more weight.

\- \*\*R-squared (R²):\*\* Measures how well the model explains variation in the target values on the evaluated data.



Evaluation results from the current 30-record dataset are preliminary. A larger, representative dataset and further testing are needed before making claims about real-world accuracy.



\## 10. Future Enhancements



\- Expand the dataset with reliable, representative student records.

\- Compare Linear Regression with other regression algorithms.

\- Add student performance categories.

\- Improve input validation and error handling.

\- Add model comparison charts and downloadable reports.

\- Evaluate the model on a larger independent test dataset.



\## 11. Conclusion



EduPredict AI demonstrates how machine learning can be integrated into a web application to estimate student academic performance. It combines data processing, regression modelling, model evaluation, and interactive visualization in one application.



The project is intended for educational and demonstration purposes and should not be used as the sole basis for academic decisions.



