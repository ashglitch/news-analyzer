import re

SENSATIONAL_TERMS = {
    "shocking", "unbelievable", "explosive", "miracle",
    "secret", "exposed", "shocking truth", "must share",
    "wake up", "they don't want you to know",
    "100% guaranteed", "breaking", "bombshell",
    "conspiracy", "terrifying", "scandal", "jaw-dropping",
    "you won't believe"
}

POSITIVE_TERMS = {
    "achieve", "achievement", "benefit", "better",
    "breakthrough", "effective", "excellent", "good",
    "growth", "improve", "improved", "improvement",
    "innovative", "positive", "progress", "recover",
    "recovery", "safe", "success", "successful",
    "support", "win", "wonderful", "help", "hope",
    "advance", "advantage"
}

NEGATIVE_TERMS = {
    "accident", "anger", "angry", "attack", "bad",
    "crisis", "danger", "dangerous", "death", "decline",
    "destroy", "disaster", "failure", "fear", "fraud",
    "harm", "hate", "injury", "loss", "negative",
    "poor", "problem", "risk", "scam", "shortage",
    "threat", "tragedy", "violence", "warning",
    "damage", "corruption", "victim"
}

NEGATIONS = {
    "not", "no", "never", "neither", "hardly",
    "isn't", "wasn't", "weren't", "doesn't", "didn't",
    "cannot", "can't", "won't", "without"
}


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def get_sentiment(headline, article_text):
    tokens = tokenize(headline + ". " + article_text)
    positive_hits = []
    negative_hits = []
    score = 0

    for index, word in enumerate(tokens):
        if word not in POSITIVE_TERMS and word not in NEGATIVE_TERMS:
            continue

        previous = tokens[max(0, index - 3):index]
        negated = any(item in NEGATIONS for item in previous)

        if word in POSITIVE_TERMS:
            if negated:
                score -= 1
                negative_hits.append(word + " (negated)")
            else:
                score += 1
                positive_hits.append(word)
        else:
            if negated:
                score += 1
                positive_hits.append(word + " (negated)")
            else:
                score -= 1
                negative_hits.append(word)

    sentiment = (
        "Positive" if score > 0
        else "Negative" if score < 0
        else "Neutral"
    )

    total = len(positive_hits) + len(negative_hits)
    confidence = (
        50 if total == 0
        else min(95, round(50 + 45 * abs(score) / total))
    )

    return {
        "sentiment": sentiment,
        "sentiment_score": score,
        "sentiment_confidence": confidence,
        "positive_words": positive_hits[:8],
        "negative_words": negative_hits[:8],
    }


def get_credibility(headline, article_text):
    text = headline + " " + article_text
    lower = text.lower()
    words = tokenize(text)
    score = 100
    reasons = []

    matched_terms = sorted(
        term for term in SENSATIONAL_TERMS if term in lower
    )

    if matched_terms:
        score -= min(25, len(matched_terms) * 5)
        reasons.append(
            "Sensational wording detected: "
            + ", ".join(matched_terms[:4]) + "."
        )

    exclamations = text.count("!")
    if exclamations >= 3:
        score -= 10
        reasons.append(f"The text uses {exclamations} exclamation marks.")
    elif exclamations == 2:
        score -= 5
        reasons.append("The text uses repeated exclamation marks.")

    caps = re.findall(r"\b[A-Z]{3,}\b", text)
    if len(caps) >= 3:
        score -= 10
        reasons.append(f"Several all-caps words appear ({len(caps)} detected).")

    if len(words) < 80:
        score -= 15
        reasons.append("The article is short, so there is limited text to assess.")

    attribution = (
        "according to", "reported by", "said", "stated",
        "announced", "study", "research", "officials",
        "statement", "data", "report"
    )

    if not any(term in lower for term in attribution):
        score -= 10
        reasons.append("No common attribution or reporting indicators were detected.")

    score = max(0, min(100, score))

    rating = (
        "Lower text-risk signals" if score >= 80
        else "Some text-risk signals" if score >= 60
        else "Higher text-risk signals"
    )

    if not reasons:
        reasons.append("No major warning signals were detected.")

    reasons.append(
        "This rule-based score is not a fact-check or a probability of truth."
    )

    return {
        "credibility_score": score,
        "credibility_rating": rating,
        "credibility_reasons": reasons,
        "matched_terms": matched_terms,
    }


def analyze_article(headline, article_text):
    headline = (headline or "").strip()
    article_text = (article_text or "").strip()

    if not headline or not article_text:
        raise ValueError("Please enter both a headline and article text.")

    if len(article_text.split()) < 10:
        raise ValueError("Please enter at least 10 words in the article.")

    sentiment = get_sentiment(headline, article_text)
    credibility = get_credibility(headline, article_text)

    sentences = re.split(r"(?<=[.!?])\s+", article_text)
    summary = " ".join(sentences[:2]).strip()

    old_score = min(100, len(credibility["matched_terms"]) * 12)
    old_level = (
        "Low" if old_score < 20
        else "Moderate" if old_score < 50
        else "High"
    )

    return {
        "headline": headline,
        "article_text": article_text,
        "summary": summary,
        "description": summary,
        "word_count": len(article_text.split()),
        **sentiment,
        **credibility,
        **get_ml_prediction(headline, article_text),
        "score": old_score,
        "level": old_level,
    }

# --- Verity machine-learning model integration ---
from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "models" / "fake_news_model.joblib"
ML_MODEL = None
MODEL_LABELS = {0: "Fake", 1: "Real"}

try:
    saved_model = joblib.load(MODEL_PATH)
    ML_MODEL = saved_model["model"]
    MODEL_LABELS = saved_model.get("labels", MODEL_LABELS)
    print("Verity ML model loaded successfully.")
except Exception as error:
    print(f"Verity ML model could not be loaded: {error}")


def get_ml_prediction(headline, article_text):
    if ML_MODEL is None:
        return {
            "model_prediction": "Unavailable",
            "model_confidence": None,
            "model_note": "The trained model could not be loaded."
        }

    text = (headline.strip() + " " + article_text.strip()).strip()

    try:
        prediction = int(ML_MODEL.predict([text])[0])
        probabilities = ML_MODEL.predict_proba([text])[0]
        classes = list(ML_MODEL.classes_)
        confidence = round(
            float(probabilities[classes.index(prediction)]) * 100, 2
        )
        label = MODEL_LABELS.get(
            prediction, MODEL_LABELS.get(str(prediction), str(prediction))
        )

        return {
            "model_prediction": label,
            "model_confidence": confidence,
            "model_note": (
                "This is a machine-learning prediction, not a verified fact-check."
            )
        }
    except Exception as error:
        print(f"ML prediction error: {error}")
        return {
            "model_prediction": "Unavailable",
            "model_confidence": None,
            "model_note": "The model could not analyze this article."
        }

