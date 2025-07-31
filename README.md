📧 Email Spam Classifier (FastAPI + ML)
This is a machine learning-based Email Spam Classifier built using Python's scikit-learn and deployed with FastAPI. It takes email text input and predicts whether it's Spam or Not Spam.

🚀 Features
Email preprocessing and text cleaning

Feature extraction using TF-IDF

Classification using LinearSVC

FastAPI backend for predictions

Dataset managed with Git LFS

Ready for cloud deployment

📊 Dataset Source
The dataset used was sourced from Kaggle and includes labeled emails as spam or ham (not spam).
link-[https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset]

Files:

email_spam_dataset.csv (raw dataset)

cleaned_email_dataset.csv (cleaned & preprocessed)

🧠 Model Info
Algorithm: Linear Support Vector Classifier (LinearSVC)

Vectorizer: TF-IDF

Files:

model.pkl – Trained spam classification model

vectorizer.pkl – TF-IDF vectorizer

main.py – FastAPI backend script

🛠️ Local Setup Instructions
Clone the repository:

git clone https://github.com/ShwetaPathak27/email-spam-classifier.git

Navigate into the folder:

cd email-spam-classifier

Install dependencies:

pip install -r requirements.txt

Start the FastAPI server:

uvicorn main:app --reload

Access in browser:

Swagger UI: http://127.0.0.1:8000/docs

ReDoc: http://127.0.0.1:8000/redoc

🔮 Prediction API
Method: POST

Endpoint: /predict

Payload Example:

bash

{
  "email": "Win $1000 now by clicking this link!"
}
Response:

json

{
  "prediction": "spam"
}

🌐 Live Demo
https://www.linkedin.com/posts/shweta-pathak-09a023295_machinelearning-fastapi-python-activity-7356644278590230528-yiqT?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEdfRrYB42UT_DC51AQyb2sW8fNnS7Ijmho

📁 Git LFS Note
CSV files are tracked using Git LFS. Ensure you have Git LFS installed before cloning:

https://git-lfs.github.com/

