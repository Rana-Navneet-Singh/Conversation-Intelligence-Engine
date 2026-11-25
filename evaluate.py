"""Download the official TweetEval sentiment files and evaluate a simple model."""

import json
import hashlib
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn import __version__ as sklearn_version

from sentiment import train_model

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "raw" / "tweet_eval" / "sentiment"
RESULT_PATH = ROOT / "data" / "processed" / "sentiment_results.json"
BASE_URL = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment"
FILES = ("train_text.txt", "train_labels.txt", "val_text.txt", "val_labels.txt",
         "test_text.txt", "test_labels.txt", "mapping.txt")
LABEL_NAMES = {"0": "negative", "1": "neutral", "2": "positive"}
LABEL_ORDER = ["negative", "neutral", "positive"]


def download_dataset():
    """Download only the sentiment task files from the public original repository."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        with urlopen(f"{BASE_URL}/{name}", timeout=30) as response:
            (DATA_DIR / name).write_bytes(response.read())


def read_split(split):
    texts = (DATA_DIR / f"{split}_text.txt").read_text(encoding="utf-8").splitlines()
    raw_labels = (DATA_DIR / f"{split}_labels.txt").read_text(encoding="utf-8").splitlines()
    if len(texts) != len(raw_labels):
        raise ValueError(f"TweetEval {split} text and label counts do not match")
    return texts, [LABEL_NAMES[label] for label in raw_labels]


def evaluate_sentiment():
    train_x, train_y = read_split("train")
    val_x, val_y = read_split("val")
    test_x, test_y = read_split("test")

    validation_results = []
    best_c = None
    best_f1 = -1.0
    for c_value in (0.5, 1.0, 2.0):
        candidate = train_model(train_x, train_y, path=None, c_value=c_value)
        val_prediction = candidate.predict(val_x)
        val_report = classification_report(val_y, val_prediction, labels=LABEL_ORDER,
                                           output_dict=True, zero_division=0)
        macro_f1 = float(val_report["macro avg"]["f1-score"])
        validation_results.append({"C": c_value,
                                   "accuracy": float(accuracy_score(val_y, val_prediction)),
                                   "macro_f1": macro_f1})
        if macro_f1 > best_f1:
            best_c, best_f1 = c_value, macro_f1

    # The official validation split selects C; the official test split stays untouched.
    final_x, final_y = train_x + val_x, train_y + val_y
    model = train_model(final_x, final_y, c_value=best_c)
    predicted = model.predict(test_x)
    labels = LABEL_ORDER
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_y, predicted, average="weighted", zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        test_y, predicted, average="macro", zero_division=0)
    majority_accuracy = Counter(test_y).most_common(1)[0][1] / len(test_y)
    file_hashes = {name: hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
                   for name in FILES}
    result = {"dataset": "TweetEval sentiment, Cardiff NLP official public repository",
              "source": "https://github.com/cardiffnlp/TweetEval/tree/main/datasets/sentiment",
              "dataset_files_sha256": file_hashes,
              "scikit_learn_version": sklearn_version,
              "preprocessing": "Original text preserved; labels mapped 0=negative, 1=neutral, 2=positive; TF-IDF word uni/bi-grams",
              "split": "Official TweetEval train/validation/test files; validation selects C, final model refit on train+validation, test used once",
              "initial_train_rows": len(train_x), "validation_rows": len(val_x),
              "final_train_rows": len(final_x), "test_rows": len(test_x),
              "selected_C": best_c, "validation_trials": validation_results,
              "accuracy": accuracy_score(test_y, predicted),
              "majority_class_baseline_accuracy": majority_accuracy,
              "precision_weighted": precision, "recall_weighted": recall, "f1_weighted": f1,
              "precision_macro": float(macro_precision),
              "recall_macro": float(macro_recall),
              "f1_macro": float(macro_f1),
              "labels": labels,
              "confusion_matrix": confusion_matrix(test_y, predicted, labels=labels).tolist(),
              "report": classification_report(test_y, predicted, labels=labels,
                                              output_dict=True, zero_division=0),
              "limitation": "English tweets are not representative of Hinglish or private chat messages."}
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    if not (DATA_DIR / "train_text.txt").exists():
        print("Downloading official TweetEval sentiment files...")
        download_dataset()
    print(json.dumps(evaluate_sentiment(), indent=2))
