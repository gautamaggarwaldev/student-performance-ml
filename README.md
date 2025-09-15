## Student Performance Detection

A Machine Learning project to predict student academic performance based on demographics, study habits, and grades.
The app is built with Streamlit and provides:

Student performance prediction (Poor / Average / Good)

- Prediction probabilities for each class
- Model’s top feature importances
- AI-powered improvement tips (using Google Gemini)
- PDF report download
- Tech Stack
- Python (3.9+)
- Machine Learning: scikit-learn, pandas, numpy
- Model Persistence: joblib
- Web Framework: Streamlit
- AI Suggestions: Google Gemini API (google-generativeai)
- PDF Report Generation: ReportLab

```bash
📂 Project Structure
student-performance/
│── app/
│   └── streamlit_app.py       # Main Streamlit app
│
│── models/
│   └── student_performance_pipeline.joblib   # Trained ML model
│
│── artifacts/
│   ├── metadata.json          # Metadata (numeric + categorical features)
│   └── feature_importances.csv # Feature importance data
│
│── requirements.txt           # Dependencies
│── README.md                  # Project documentation
```

## Setup Instructions
- Clone the Repository

git clone https://github.com/gautamaggarwaldev/student-performance-ml.git
cd student-performance-ml

- Create Virtual Environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\activate
# Activate (Linux/Mac)
source .venv/bin/activate

- Install Dependencies
pip install -r requirements.txt

- Configure Gemini API Key

Option A: Using Streamlit Secrets (recommended)
Create a .streamlit folder inside your project.
Add a file named secrets.toml:
GEMINI_API_KEY = "your_google_gemini_api_key"

Option B: Using Environment Variable
# Windows PowerShell
setx GEMINI_API_KEY "your_google_gemini_api_key"

# Linux/Mac
export GEMINI_API_KEY="your_google_gemini_api_key"

- Run the App
streamlit run app/streamlit_app.py


Open in browser http://localhost:8501

## Features of the App

Input Form
Student’s age, study time, past failures, absences, grades (G1, G2)
Categorical inputs like school support, family support, etc.

Prediction Output
Performance class (Poor / Average / Good)

Probabilities for each class
Top 10 most important features affecting prediction

AI Suggestions

Personalized academic improvement tips powered by Google Gemini AI

PDF Report

Downloadable summary including prediction, probabilities, feature importance, and suggestions

## Example Output

✅ Prediction: Poor

📊 Probabilities: Poor (0.72), Average (0.21), Good (0.07)

🔎 Top Features: G2, G1, Absences, Age, Failures

💡 AI Suggestions:

Review class notes daily
Seek extra help for weak subjects
Reduce absences and attend study groups
Manage study time effectively

## Future Improvements

Add visualization (bar chart for feature importance)

Deploy on Streamlit Cloud or Heroku

Extend dataset for better accuracy

Include parent/teacher feedback as features

## Authors

Developed by Vedant Sharma.