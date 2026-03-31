import pandas as pd
import string
import nltk
import joblib
import os
import sys
import json
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Conditional NLTK download
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Create model directory
os.makedirs("model", exist_ok=True)

df = pd.read_csv("dataset/reviews.csv")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["clean_review"] = df["review"].apply(clean_text)

X = df["clean_review"]
y = df["label"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

vocab_size = 5000
max_len = 100
vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=max_len)
vectorizer.adapt(X)
X_sequences = vectorizer(X)
X_pad = pad_sequences(X_sequences, maxlen=max_len, padding='post')

X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y_encoded, test_size=0.2, random_state=42
)

model = Sequential([
    Embedding(vocab_size, 128, mask_zero=True),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("Training model...")
history = model.fit(
    X_train, y_train, 
    epochs=10, 
    batch_size=32, 
    validation_data=(X_test, y_test), 
    verbose=1
)

val_acc = history.history['val_accuracy'][-1]
print(f"Final Validation Accuracy: {val_acc:.4f}")

# Save model and artifacts to model/
model_path = "model/fake_review_model.h5"
model.save(model_path)

# Save vectorizer config and vocab
vectorizer_config = vectorizer.get_config()
with open("model/vectorizer_config.json", "w") as f:
    json.dump(vectorizer_config, f)

vocab = vectorizer.get_vocabulary()
with open("model/vocab.json", "w") as f:
    json.dump(vocab, f)

joblib.dump(encoder, "model/label_encoder.pkl")

print(f"✅ Training complete. Artifacts saved to model/:")
print(f"  - {model_path}")
print(f"  - model/vectorizer_config.json")
print(f"  - model/vocab.json") 
print(f"  - model/label_encoder.pkl")
print("Ready for backend deployment!")
