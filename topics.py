"""Semantic topic grouping for small conversations (CPU embeddings)."""

import re
import os
from pathlib import Path

FILLERS = {"yes", "yeah", "yep", "ok", "okay", "hmm", "lol", "haha", "haan", "acha", "thanks"}
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _message_rows(messages):
    rows = []
    for row in messages:
        text = row.get("message", "").strip() if isinstance(row, dict) else ""
        key = re.sub(r"\W+", " ", text.casefold()).strip()
        # Keep duplicate instances in the final topic, but embed each unique string once.
        if text and key not in FILLERS and len(re.findall(r"\w+", text)) >= 3:
            rows.append(row)
    return rows


def cluster_embeddings(vectors, distance_threshold=0.62):
    """Cluster normalized semantic vectors without choosing a fixed topic count."""
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        return []
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = values / np.maximum(norms, 1e-12)
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold,
                                    metric="cosine", linkage="average")
    return model.fit_predict(values).tolist()


def _title(keywords):
    words = set(keywords)
    themes = [("Planning and deadlines", {"plan", "project", "deadline", "finish", "schedule", "science", "presentation"}),
              ("Trip packing and supplies", {"lunch", "snacks", "snack", "hike", "water", "first aid", "pack"}),
              ("Travel and transport", {"train", "travel", "trip", "ticket", "station"}),
              ("Food and meals", {"food", "lunch", "dinner", "eat", "restaurant"}),
              ("Meetings and coordination", {"meet", "meeting", "time", "call", "tomorrow"})]
    for name, terms in themes:
        if words & terms:
            return name
    return "Discussion: " + ", ".join(keywords[:3]) if keywords else "General discussion"


def topics_from_embeddings(messages, vectors, distance_threshold=0.62):
    """Build UI-ready topic cards from rows and precomputed embedding vectors."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    rows = _message_rows(messages)
    if len(rows) < 2:
        return []
    vectors = list(vectors)
    if len(vectors) != len(rows):
        return []
    # Cluster each distinct text once, then restore repeats in the evidence list.
    unique_rows, unique_vectors, copies = [], [], []
    text_indexes = {}
    for row, vector in zip(rows, vectors):
        key = re.sub(r"\W+", " ", row["message"].casefold()).strip()
        if key not in text_indexes:
            text_indexes[key] = len(unique_rows)
            unique_rows.append(row)
            unique_vectors.append(vector)
            copies.append([row])
        else:
            copies[text_indexes[key]].append(row)
    labels = cluster_embeddings(unique_vectors, distance_threshold)
    if not labels:
        return []
    output = []
    for label in sorted(set(labels)):
        indexes = [i for i, value in enumerate(labels) if value == label]
        if len(indexes) < 2:
            continue
        cluster_unique_rows = [unique_rows[i] for i in indexes]
        cluster_rows = [row for i in indexes for row in copies[i]]
        try:
            stop_words = set(ENGLISH_STOP_WORDS) | {"hai", "hain", "haan", "aur", "kal", "mein", "theek"}
            vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2),
                                         stop_words=sorted(stop_words),
                                         token_pattern=r"(?u)\b\w\w+\b")
            matrix = vectorizer.fit_transform([r["message"] for r in cluster_unique_rows])
            average = matrix.mean(axis=0).A1
            terms = vectorizer.get_feature_names_out()
            keywords = [terms[i] for i in average.argsort()[::-1] if average[i] > 0][:5]
        except ValueError:
            keywords = []
        import numpy as np
        cluster_vectors = np.asarray([unique_vectors[i] for i in indexes], dtype=float)
        center = cluster_vectors.mean(axis=0)
        center /= max(np.linalg.norm(center), 1e-12)
        distances = 1 - (cluster_vectors @ center) / np.maximum(np.linalg.norm(cluster_vectors, axis=1), 1e-12)
        representatives = [cluster_unique_rows[i] for i in np.argsort(distances)[:3]]
        output.append({"id": len(output), "name": _title(keywords), "keywords": keywords,
                       "messages": cluster_rows, "representative_messages": representatives,
                       "participants": sorted({r.get("sender", "Unknown") for r in cluster_rows}),
                       "silhouette": None})
    return output


def find_topics(messages, distance_threshold=0.62):
    """Embed eligible messages with the multilingual model, then cluster on CPU."""
    rows = _message_rows(messages)
    if len(rows) < 2:
        return []
    cache_dir = Path(__file__).parent / "models" / "topic_embeddings"
    # Keep Hugging Face and Xet logs/cache within the project model directory.
    os.environ["HF_HOME"] = str(cache_dir / "hf_home")
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hf_hub")
    os.environ["HF_XET_CACHE"] = str(cache_dir / "hf_xet")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=MODEL_NAME,
                          cache_dir=str(cache_dir),
                          cuda=False, threads=2)
    unique_texts, text_to_vector = [], {}
    for row in rows:
        text = row["message"].strip()
        if text not in text_to_vector:
            text_to_vector[text] = None
            unique_texts.append(text)
    vectors = list(model.embed(unique_texts))
    text_to_vector.update(zip(unique_texts, vectors))
    return topics_from_embeddings(rows, [text_to_vector[r["message"].strip()] for r in rows],
                                  distance_threshold)
