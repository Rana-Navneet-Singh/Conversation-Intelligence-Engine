"""TF-IDF and logistic-regression sentiment model helpers."""

import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "sentiment.pkl"
LABELS = {"positive": "positive", "negative": "negative", "neutral": "neutral"}


def load_model(path=MODEL_PATH):
    if not Path(path).exists():
        return None
    with open(path, "rb") as file:
        return pickle.load(file)


def predict(messages, model=None):
    # sklearn rejects a zero-row matrix, so never call the pipeline without text.
    messages = list(messages)
    valid_indexes = [index for index, text in enumerate(messages)
                     if isinstance(text, str) and text.strip()]
    if not valid_indexes:
        return []
    model = model or load_model()
    if model is None:
        return ["unavailable"] * len(messages)
    predictions = model.predict([messages[index] for index in valid_indexes])
    results = ["unavailable"] * len(messages)
    for index, label in zip(valid_indexes, predictions):
        results[index] = label
    return results


def train_model(texts, labels, path=MODEL_PATH, c_value=1.0, min_df=2):
    """Train only from an explicitly supplied labelled dataset."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df,
                                                  strip_accents="unicode")),
                      ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced",
                                                         C=c_value, random_state=42))])
    model.fit(texts, labels)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(model, file)
    return model
