
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

FAKE_FILE = DATA_DIR / "Fake.csv"
REAL_FILE = DATA_DIR / "True.csv"
MODEL_FILE = MODEL_DIR / "fake_news_model.joblib"


def load_dataset():
    if not FAKE_FILE.exists() or not REAL_FILE.exists():
        raise FileNotFoundError(
            "Dataset missing! Put Fake.csv and True.csv inside the data folder."
        )

    fake = pd.read_csv(FAKE_FILE)
    real = pd.read_csv(REAL_FILE)

    for name, frame in [("Fake.csv", fake), ("True.csv", real)]:
        if "text" not in frame.columns:
            raise ValueError(f"{name} must contain a 'text' column.")
        if "title" not in frame.columns:
            frame["title"] = ""

    fake["label"] = 0
    real["label"] = 1

    data = pd.concat([fake, real], ignore_index=True)
    data["title"] = data["title"].fillna("").astype(str)
    data["text"] = data["text"].fillna("").astype(str)
    data["article"] = (
        data["title"].str.strip() + " " + data["text"].str.strip()
    ).str.strip()

    data = data[data["article"].str.len() >= 50]
    data = data.drop_duplicates(subset=["article"])

    if data["label"].nunique() != 2:
        raise ValueError("The dataset must contain both Fake and Real articles.")

    return data


def main():
    print("Loading dataset...")
    data = load_dataset()

    print(f"Articles available: {len(data):,}")
    print(data["label"].map({0: "Fake", 1: "Real"}).value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        data["article"],
        data["label"],
        test_size=0.20,
        random_state=42,
        stratify=data["label"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=100000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    print("\nTraining model. Please wait...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\n===== MODEL EVALUATION =====")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        zero_division=0,
    ))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "labels": {0: "Fake", 1: "Real"},
        "test_accuracy": float(accuracy),
    }, MODEL_FILE)

    print(f"\nModel saved successfully: {MODEL_FILE}")
    print("Training complete!")


if __name__ == "__main__":
    main()
