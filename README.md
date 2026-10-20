# Verity — News Analyzer

**Read beyond the headline.**

Verity is a web-based news analysis tool that examines an article's text and highlights patterns that may be worth a closer look. It combines machine learning, sentiment analysis, and text-based signals in one interface, so users can explore an article instead of relying on a single label.

Built with **Python, Flask, scikit-learn, HTML, CSS, and JavaScript**.

## What it does

- **News classification** — uses a trained TF-IDF and Logistic Regression pipeline to classify an article as *Fake* or *Real* based on patterns learned from its training data.
- **Sentiment analysis** — identifies the overall sentiment expressed in the text.
- **Text-risk signals** — highlights selected textual patterns that may deserve closer attention.
- **Article summaries and signals** — presents the analysis in a readable results page, including confidence, sentiment, and other text statistics.
- **Analysis history** — stores previous analyses locally so users can revisit results.

## How it works

Verity processes the article text through a machine-learning pipeline trained on labelled news articles. TF-IDF converts text into numerical features, and Logistic Regression uses those features to make a classification.

Sentiment and text-risk indicators provide additional context alongside the model's prediction. They are separate signals, not independent proof that a story is true or false.

## Tech stack

| Area | Tools |
|---|---|
| Web application | Python, Flask |
| Machine learning | scikit-learn, TF-IDF, Logistic Regression |
| Front end | HTML, CSS, JavaScript |
| Data storage | SQLite |
| Model storage | Joblib |

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/ashglitch/news-analyzer.git
cd news-analyzer
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the commands below using the virtual environment's Python directly, or adjust the execution policy for your current terminal session.

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the application

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

The application uses the trained model in `models/`. The CSV training datasets are not included in the repository.

## Project structure

```text
news-analyzer/
├── app.py                  # Flask routes and application logic
├── analyzer.py             # Text analysis and model predictions
├── database.py             # Local database operations
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── models/
│   └── fake_news_model.joblib
├── templates/              # HTML pages
└── static/
    └── css/style.css       # Interface styling
```

## Important limitations

Verity is an educational project for exploring news-text analysis, not a fact-checking authority.

- Predictions reflect patterns learned from the training dataset and may not generalise to every publisher, topic, or writing style.
- A model's confidence score is not a calibrated probability that an article is true.
- Sentiment and text-risk signals describe features of the text; they do not establish whether its claims are accurate.
- Important claims should still be checked against reliable sources and original evidence.

## Why I built it

The goal was to bring machine learning and web development together in a practical application: take a piece of news, process its text, and present the output in a way that is easier to inspect and understand.

Verity is a work in progress, and there is room to improve its evaluation, explainability, and ability to compare claims with external evidence.

---

**Repository:** [ashglitch/news-analyzer](https://github.com/ashglitch/news-analyzer)
