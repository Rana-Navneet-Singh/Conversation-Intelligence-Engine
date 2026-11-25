# Conversation Intelligence Engine

A small local Streamlit app for exploring a new WhatsApp TXT export. It parses text messages and displays activity, sentiment, topic clusters, chronological source-grounded topic summaries, and possible decision statements with their supporting messages. It does not analyze media files or send chat contents to an external service.

## Features

- Parses common WhatsApp date and time formats, sender names, and multiline messages. System notices, empty messages, and media placeholders are skipped.
- Shows a parsed-message preview, date range, participant activity counts, and daily activity chart.
- Classifies messages with a trained TF-IDF and Logistic Regression sentiment pipeline.
- Groups eligible messages by multilingual semantic embeddings and variable-count agglomerative clustering. Short fillers are ignored; topic cards show keywords, participants, examples, and all assigned original messages.
- Summaries preserve the chronological messages and speaker attribution, then report only explicitly supported outcomes and commitments.
- Decision detection uses explainable cues and nearby explicit replies to label proposals, agreements, rejections, confirmations, and self-assigned commitments. Every item includes source messages and uncertainty.

## Project structure

```text
app.py                  Streamlit interface
chat_parser.py          WhatsApp TXT parsing
overview.py             Counts and activity data
sentiment.py            Sentiment model training and predictions
topics.py               Multilingual semantic embeddings and clustering
summary.py              Chronological source-based topic summaries
decisions.py            Explainable proposal and commitment extraction
evaluate.py             TweetEval download and evaluation
requirements.txt        Python packages used by the app
PRESENTATION_GUIDE.md   Student presentation notes
data/raw/               Public TweetEval files; no personal chats
data/processed/         Saved evaluation metrics
models/                 Saved sentiment model (`sentiment.pkl`); `topic_embeddings/` downloaded on first run
tests/                  Unit tests and synthetic chat fixture
outputs/                Generated presentation deck (`Navneet_Conversation_Intelligence_Engine.pptx`)
work/                   Local scratch directory (ignored by Git)
```

## Dataset and sentiment model

The model uses the public [TweetEval sentiment task from Cardiff NLP](https://github.com/cardiffnlp/TweetEval/tree/main/datasets/sentiment). The downloaded task files are in `data/raw/tweet_eval/sentiment/`. TweetEval maps labels `0`, `1`, and `2` to negative, neutral, and positive. Its English tweets are short social-media messages; they are not a representative test of Hinglish or private WhatsApp text.

`evaluate.py` preserves the original text, maps the labels, and uses the dataset's official train, validation, and test splits. It compares Logistic Regression `C` values on validation macro F1, refits the selected setting on train plus validation, and evaluates once on the untouched test split. The pipeline uses TF-IDF word unigrams and bigrams with Logistic Regression. The saved model is `models/sentiment.pkl`; metrics, validation trials, confusion matrix, package version, and SHA-256 hashes are saved in `data/processed/sentiment_results.json`.

## Topic, summary, and decision methods

Topic vectors use the multilingual `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` ONNX model through FastEmbed (`fastembed`), with CPU execution only (`cuda=False`, `threads=2`). Model files are automatically downloaded on first topic analysis into `models/topic_embeddings/` (excluded from Git because `model_optimized.onnx` is ~224 MB, exceeding GitHub's 100 MB file limit). Agglomerative clustering uses cosine distance (`linkage="average"`, `distance_threshold=0.62`) and does not require a fixed number of groups. Short filler messages and messages with fewer than 3 word tokens are skipped. A modest conversation may yield no clusters if messages are too short or dissimilar. Topic labels use simple keyword matching with TF-IDF fallback keywords and should be reviewed alongside the original messages.

Summaries (`summary.py`) are chronological, source-grounded evidence views, not generated prose: they order the original messages chronologically, attribute each to its sender, and tag message roles (`proposal`, `agreement`, `rejection`, `confirmation`, `commitment`, or `message`) using `decisions.py`. Outcome conclusions and tasks/commitments are populated only from explicit language. Decision detection is a rule-based regex system, not a trained classifier; its confidence labels (`low`, `medium`, `high`) are heuristic and it can miss indirect, sarcastic, or nuanced responses. Agreement is attached only when an explicit reply appears within up to 4 subsequent messages and 30 minutes of an open proposal. Deadlines and task owners are shown only when explicit.

Tests include small, manually labeled synthetic conversations. On a nine-message sample fixture and a separate six-message Hinglish/English set, the trained embedding model grouped the intended pairs/classes with pairwise precision 1.00, recall 1.00, F1 1.00, and 100% message coverage. These are tiny, curated checks used to validate the implementation, not a representative benchmark or a claim of general topic quality; the clustering threshold is heuristic and should be tested on broader annotated conversations.

### Recorded evaluation

These are measured on the public TweetEval test split (12,284 tweets), not on personal chats:

| Metric | Result |
| --- | ---: |
| Accuracy | 0.5981 |
| Weighted precision | 0.6021 |
| Weighted recall | 0.5981 |
| Weighted F1 | 0.5984 |
| Macro precision | 0.5887 |
| Macro recall | 0.6024 |
| Macro F1 | 0.5941 |
| Majority-class accuracy baseline | 0.4833 |

Confusion matrix, with rows as true labels and columns as predicted labels in the order negative, neutral, positive:

```text
[[2521, 1194,  257],
 [1636, 3402,  899],
 [ 250,  701, 1424]]
```

The best validation result was `C=2.0` (validation accuracy 0.6670, macro F1 0.6456). Test performance is modest and must not be presented as evidence of high accuracy on Hinglish. The dataset's English social-media distribution differs from private conversations, slang, sarcasm, and code-switching.

## Dependencies and automatic downloads

Python dependencies are listed in `requirements.txt` (`streamlit`, `pandas`, `scikit-learn`, `altair`, and `fastembed`; `numpy` is also used by `topics.py` and installed transitively).

- **Sentiment model (`models/sentiment.pkl`)**: Included in the repository for immediate use. Can be regenerated with `evaluate.py`.
- **TweetEval dataset (`data/raw/tweet_eval/sentiment/`)**: Included in the repository. If deleted or absent, `evaluate.py` downloads only those public task files from Cardiff NLP's GitHub repository.
- **Topic embedding model (`models/topic_embeddings/`)**: Not tracked in Git (~240 MB cached; `model_optimized.onnx` is ~224 MB). Automatically downloaded on CPU via FastEmbed the first time `find_topics()` runs in `app.py` or `tests/test_core.py`.

## Setup on Windows

The project uses Python 3.13 in an isolated `.venv` environment. From PowerShell in this folder, run the environment's Python directly so packages cannot be installed globally:

```powershell
# Run once only if the project does not already have a valid .venv.
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m streamlit run app.py
```

To reproduce sentiment training and evaluation, run:

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

If the TweetEval sentiment files are absent, `evaluate.py` downloads only those public task files from Cardiff NLP's repository. It does not download any model weights.

## Topic and decision evaluation

Topic clusters are unsupervised and generated for each uploaded chat; the current code does not compute or display a silhouette score (`silhouette` is `None`), so representative messages closest to each cluster center should be inspected manually. Decision detection uses regex phrase cues and has no labelled benchmark in this project, so it is an aid for review rather than a validated decision classifier.

## Limitations

- TweetEval sentiment scores describe an English tweet benchmark, not Hinglish chat performance.
- Embedding-based agglomerative clustering with a fixed cosine `distance_threshold=0.62` can split one discussion or merge different ones, and small chats may produce no clusters. Review the messages in every cluster.
- The chronological, source-grounded summary arranges original messages and surfaces only explicit outcomes or self-assigned commitments; it does not infer implicit narrative context.
- Phrase-based decision flags can miss implied decisions and can flag non-final discussion. The app shows evidence and does not infer disagreement or non-participation.
- Parsing supports common exports; unusual locale-specific date formats may need an added pattern.
- No pretrained generative summarization model has been downloaded or used.
