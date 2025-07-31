import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load the cleaned dataset
df = pd.read_csv(r'D:/Email spam classifier/cleaned_email_dataset.csv')

# Drop rows with missing or empty cleaned text
df = df.dropna(subset=['cleaned'])
df = df[df['cleaned'].str.strip() != '']
print(f"✅ Dataset cleaned: {len(df)} rows remain.\n")

# Features and Labels
X = df['cleaned']
y = df['label']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train Linear SVC with probability calibration
svc = LinearSVC()
model = CalibratedClassifierCV(svc, cv=5)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("📄 Classification Report:\n", classification_report(y_test, y_pred))
print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model and vectorizer
pickle.dump(model, open(r'D:/Email spam classifier/svc_model.pkl', 'wb'))
pickle.dump(vectorizer, open(r'D:/Email spam classifier/tfidf_vectorizer.pkl', 'wb'))
print("\n✅ Model and vectorizer saved successfully!")

# ======== Live Terminal Prediction Loop ========
print("\n📬 Type a message to classify it as SPAM or HAM.")
print("🔴 Type 'exit' to quit.\n")

while True:
    user_input = input("📨 Enter Email Text: ").strip()
    if user_input.lower() == 'exit':
        print("👋 Exiting prediction loop.")
        break
    elif user_input == '':
        print("⚠️ Please enter valid non-empty text.\n")
        continue
    try:
        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0]
        label = "SPAM" if prediction == 1 else "HAM"
        confidence = round(proba[prediction] * 100, 2)
        print(f"🔍 Prediction: {label} (Confidence: {confidence}%)\n")
    except Exception as e:
        print(f"❌ Error during prediction: {e}\n")
