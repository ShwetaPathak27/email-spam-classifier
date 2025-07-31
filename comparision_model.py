import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# Load the cleaned data
df = pd.read_csv(r'D:/Email spam classifier/cleaned_email_dataset.csv')

# Drop rows with NaN or empty text
df = df.dropna(subset=['cleaned'])
df = df[df['cleaned'].str.strip() != '']

# Features and Labels
X = df['cleaned']
y = df['label']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42)

# Models to Compare
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVC": LinearSVC()
}

print("\n🔬 Comparing Models on Spam Detection Dataset:\n")

# Train & Evaluate Each Model
for name, model in models.items():
    print(f"\n================== {name} ==================")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {accuracy:.4f}")
    print("📄 Classification Report:")
    print(classification_report(y_test, y_pred))
