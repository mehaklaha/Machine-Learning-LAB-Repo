import os
import string
import joblib

from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# CREATE APP (VERY IMPORTANT)
# -----------------------------
app = FastAPI()

# -----------------------------
# LOAD MODEL SAFELY
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "model", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "..", "model", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# -----------------------------
# INPUT SCHEMA
# -----------------------------
class Review(BaseModel):
    text: str

# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def home():
    return {"message": "Fake Review Detection API running"}

@app.post("/predict")
def predict(review: Review):

    cleaned = clean_text(review.text)

    vec = vectorizer.transform([cleaned])

    pred = model.predict(vec)[0]

    prob = model.predict_proba(vec)[0]
    confidence = max(prob)

    return {
        "review": review.text,
        "prediction": pred,
        "confidence": float(confidence)
    }