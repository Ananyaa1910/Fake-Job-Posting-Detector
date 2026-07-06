# Fake-Job-Posting-Detector
Online job portals contain fraudulent postings that can lead to financial scams and identity theft.  This project builds a machine learning system to classify job postings as legitimate or fraudulent.

This project detects whether a job posting is likely to be genuine or fraudulent using machine learning and natural language processing techniques.

I built this project to understand how ML models can be used to solve real-world problems involving text data. Online job scams have become increasingly common, and this project explores how characteristics of fraudulent postings differ from legitimate ones.

## Dataset

I used the EMSCAD (Employment Scam Aegean Dataset), which contains around 18,000 job postings labeled as real or fake.

## What I did

* Cleaned and preprocessed textual data from job postings.
* Combined important fields such as title, description, and requirements.
* Extracted TF-IDF features from text.
* Engineered additional metadata features such as:

  * Presence of company logo
  * Missing salary information
  * Missing company profile
  * Length of descriptions
  * Excessive punctuation and urgency indicators
* Trained and compared multiple machine learning models:

  * Logistic Regression
  * Random Forest
  * XGBoost
* Used SHAP values to understand which features influenced predictions.
* Built a Streamlit application for interactive predictions.

## Model Performance

Among the models tested, XGBoost gave the best overall performance.

* Accuracy: ~98%
* Fraud Class F1 Score: ~0.84
* ROC-AUC: ~0.97

## Technologies Used

* Python
* Pandas
* Scikit-learn
* XGBoost
* SHAP
* Streamlit

## Running the Project

Clone the repository:

git clone https://github.com/Ananyaa1910/fake-job-detector.git

Install dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

## Project Structure

fake-job-detector/
├── app.py
├── requirements.txt
├── fake_job_artifacts.pkl
├── notebook.ipynb
├── README.md
├── LICENSE
└── screenshots/

## Future Improvements

* Experiment with transformer-based models such as BERT.
* Improve the suspicious phrase detection module.
* Extend the application to support browser-based scam detection.

## Author

Ananyaa Srivastava
