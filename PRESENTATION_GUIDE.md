# Presentation Guide

## Project overview

The Conversation Intelligence Engine reads a text-only WhatsApp export and helps a user scan activity, sentiment, themes, a short summary, and possible decisions. Uploaded chats are analyzed locally and are not used to train the public sentiment model.

## Walkthrough

1. **Upload and parsing:** `app.py` accepts a TXT file. `chat_parser.py` reads timestamps and sender lines, joins multiline messages, and skips system notices and media placeholders.
2. **Overview:** `overview.py` counts messages and participants, sorts active participants, finds the date range, and groups messages by day.
3. **Sentiment:** `sentiment.py` applies TF-IDF word features and Logistic Regression. `evaluate.py` tuned the regularization value on TweetEval validation data, then measured final results on its untouched official test split.
4. **Topics:** `topics.py` filters out short filler messages and embeds eligible messages on CPU using FastEmbed (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` ONNX), then groups them with cosine-distance Agglomerative Clustering (`distance_threshold=0.62`) without fixing the cluster count. It derives keywords with TF-IDF and displays messages closest to each cluster center.
5. **Topic summary:** `summary.py` builds a chronological, source-grounded summary by ordering original messages with sender attribution, tagging message roles via `decisions.py`, and listing only explicit conclusions and tasks.
6. **Possible decisions:** `decisions.py` uses transparent regex cues to detect proposals, nearby explicit agreements/rejections (within 30 minutes and up to 4 messages), confirmations, self-assigned commitments, and explicit deadlines, returning the original messages as evidence. It does not infer who was absent or treat silence as disagreement.

## How to explain the sentiment model

TF-IDF turns words and word pairs into numbers. Words that help distinguish a message receive more weight than very common words. Logistic Regression learns how these values relate to the three training labels: negative, neutral, and positive. The model is small enough to train on a laptop CPU.

TweetEval's public sentiment split was used. We preserved tweet text and mapped labels 0/1/2 to negative/neutral/positive. We compared `C=0.5`, `1.0`, and `2.0` on the official validation split using macro F1. `C=2.0` scored best there. We refit on train plus validation and evaluated once on the official test set, containing 12,284 tweets.

Measured test results:

- Accuracy: **0.5981**
- Weighted precision: **0.6021**
- Weighted recall: **0.5981**
- Weighted F1: **0.5984**
- Macro precision: **0.5887**
- Macro recall: **0.6024**
- Macro F1: **0.5941**
- Majority-class accuracy baseline: **0.4833**

This is a modest result on English tweets. It does not prove accuracy on Hinglish, sarcasm, personal shorthand, or WhatsApp conversations. Do not claim that it does.

## How to explain topics and summaries

FastEmbed computes multilingual ONNX sentence embeddings on CPU (downloading ~240 MB into `models/topic_embeddings/` on first run), and Agglomerative Clustering merges messages within cosine distance `0.62` using average linkage. TF-IDF extracts keywords to name or describe each cluster, and representative messages closest to the cluster center are displayed so the class can judge the grouping directly.

The summary is chronological and source-grounded: it keeps original wording and speaker attribution in time order, tags roles from `decisions.py`, and reports conclusions or tasks only when stated explicitly. Dates, senders, and supporting messages are shown separately so a reader can check the source.

## How to explain decision flags

The decision detector is a transparent regex phrase search, not a trained classifier. It can miss implied decisions or flag a sentence that was only a suggestion. Every flag includes its original supporting message and confidence label (`low`, `medium`, `high`). Agreement is only attached when an explicit reply follows a proposal within 30 minutes and 4 messages; the app does not label silence as disagreement.

## Suggested presentation order

1. Describe the problem: long chats are hard to review.
2. Demonstrate parsing with a short synthetic example containing a multiline message.
3. Explain TF-IDF and Logistic Regression, then show the real held-out TweetEval results and confusion matrix.
4. Show the semantic topic clusters (FastEmbed + Agglomerative Clustering) and representative messages from a synthetic example.
5. Show the chronological source-grounded summary and decision evidence.
6. Finish with limitations, especially English-to-Hinglish transfer, clustering threshold errors, and phrase-rule errors.

## Commands for the demo

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Run tests with `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` and reproduce sentiment evaluation with `.\.venv\Scripts\python.exe evaluate.py`.
