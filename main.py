from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle


# === Load model and vectorizer ===
model = pickle.load(open("D:/Email spam classifier/svc_model.pkl", "rb"))
vectorizer = pickle.load(open("D:/Email spam classifier/tfidf_vectorizer.pkl", "rb"))

# === FastAPI app ===
app = FastAPI(title="📬 Email Spam Classifier API")

# Mount static folder (for index.html and JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# === Request body ===
class EmailText(BaseModel):
    text: str

# === Serve index.html ===
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return FileResponse("static/index.html")

# === API Endpoint ===
@app.post("/predict")
def predict_spam(data: EmailText):
    input_text = data.text
    vector = vectorizer.transform([input_text])
    prediction = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]

    label = "SPAM" if prediction == 1 else "HAM"
    confidence = round(proba[prediction] * 100, 2)

    return {
        "label": label,
        "confidence": confidence,
        "raw_score": float(proba[prediction])
    }
