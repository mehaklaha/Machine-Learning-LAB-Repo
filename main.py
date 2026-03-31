from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import string
import nltk
import os
import sys
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import TextVectorization

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nltk.corpus import stopwords

# Conditional NLTK
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

app = FastAPI(title="Fake Review Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model & artifacts...")
model_path = "model/fake_review_model.h5"
try:
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Loaded model: {model_path}")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    print("Run `python model/train_model.py` first!")
    raise

# Load vectorizer
with open("model/vectorizer_config.json", "r") as f:
    vectorizer_config = json.load(f)
vectorizer = TextVectorization.from_config(vectorizer_config)
with open("model/vocab.json", "r") as f:
    vocab = json.load(f)
vectorizer.set_vocabulary(vocab)
print("✅ Loaded vectorizer")

encoder = joblib.load("model/label_encoder.pkl")
print("✅ Loaded encoder")

MAX_LEN = 100
stop_words = set(stopwords.words("english"))

class Review(BaseModel):
    text: str

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

@app.get("/")
def root():
    return {"message": "Fake Review Detection API ready! POST to /predict"}

@app.post("/predict")
def predict(review: Review):
    try:
        cleaned = clean_text(review.text)
        # Vectorize
        vect_text = tf.constant([cleaned])
        vect_seq = vectorizer(vect_text)
        # Pad
        padded = pad_sequences(vect_seq, maxlen=MAX_LEN, padding='post')
        
        # Predict
        pred_proba = model.predict(padded, verbose=0)[0][0]
        pred_idx = 1 if pred_proba > 0.5 else 0  # 1=Fake, 0=Genuine
        pred_label = encoder.inverse_transform([pred_idx])[0]
        
        confidence = float(pred_proba) if pred_label == "Fake" else float(1 - pred_proba)
        
        return {
            "prediction": pred_label,
            "confidence": confidence,
            "raw_proba_fake": float(pred_proba)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
