import pandas as pd
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("../dataset/reviews.csv")

print("Columns:", df.columns)

# -----------------------------
# CLEAN DATASET STRUCTURE
# -----------------------------

# Remove unwanted columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Rename review column
if "Review Text" in df.columns:
    df.rename(columns={"Review Text": "review"}, inplace=True)

# If already correct
if "review" not in df.columns:
    raise ValueError("No review column found in dataset")

# -----------------------------
# HANDLE LABELS
# -----------------------------

# If dataset does NOT have labels → create dummy labels
if "label" not in df.columns:
    print("⚠️ No label column found → creating dummy labels")
    df["label"] = "genuine"

# Keep only required columns
df = df[["review", "label"]]

# Remove missing values
df.dropna(inplace=True)

# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df["clean_review"] = df["review"].apply(clean_text)

# -----------------------------
# FEATURES & LABELS
# -----------------------------
X = df["clean_review"]
y = df["label"]

# Convert text → numerical (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# -----------------------------
# MODEL TRAINING
# -----------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
preds = model.predict(X_test)

print("\nModel Performance:\n")
print(classification_report(y_test, preds))

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n✅ Model trained and saved successfully!")