# Fake Review Detection Using NLP & Machine Learning

A complete end-to-end Machine Learning and Natural Language Processing (NLP) web application that classifies online product reviews as **Genuine** or **Fake**. Built with Python, Flask, Scikit-Learn, NLTK, HTML5, CSS3, and JavaScript.

---

## Overview

Online e-commerce platforms and review websites often suffer from inflated ratings and fake reviews written by bots, paid reviewers, or computer-generation tools. This project implements an NLP pipeline to analyze textual review patterns, convert text into numerical feature vectors using **TF-IDF**, and classify reviews using a **Logistic Regression** model.

The web application provides a clean, practical, user-friendly interface where users can paste custom reviews or select sample reviews to view real-time model predictions along with confidence probability scores.

---

## Problem Statement

Distinguishing genuine customer reviews from deceptive or computer-generated fake reviews is critical for maintaining user trust in e-commerce ecosystems. Manual moderation does not scale to millions of user-submitted reviews. This project builds a lightweight, transparent, and reproducible automated machine learning model to flag suspicious language patterns.

---

## Dataset

The model is trained on the benchmark **Fake Reviews Dataset** (derived from the Amazon Fake Reviews dataset / Deceptive Opinion Spam corpus).

* **Dataset Size:** 40,432 labeled reviews
* **Target Classes:** 
  * `OR` / `Genuine` (Original human-written reviews) - 20,216 samples
  * `CG` / `Fake` (Computer-generated / deceptive fake reviews) - 20,216 samples
* **Source:** [Kaggle Fake Reviews Dataset](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset)
* **File Location:** `data/reviews.csv`

---

## Technologies Used

* **Language:** Python 3.x
* **Backend Web Framework:** Flask
* **Machine Learning & NLP:** `scikit-learn`, `pandas`, `numpy`, `nltk`, `joblib`
* **Frontend UI:** HTML5, Vanilla CSS3 (Custom responsive layout), JavaScript (Fetch API)

---

## NLP Preprocessing

Raw text input is processed through a structured cleaning pipeline implemented in `src/preprocessing.py`:

1. **Lowercasing:** Converts text to lowercase to standardize word matching.
2. **HTML & URL Removal:** Strips web links, URLs, and HTML tags using regular expressions.
3. **Punctuation & Character Cleaning:** Removes punctuation, symbols, and special characters, retaining only alphanumeric characters and whitespace.
4. **Tokenization:** Splits review strings into individual word tokens.
5. **Stopwords Removal:** Filters out high-frequency non-informative English words using `nltk.corpus.stopwords`.
6. **Whitespace Normalization:** Strips excess whitespace and joins clean tokens.

---

## TF-IDF (Term Frequency-Inverse Document Frequency)

Cleaned reviews are transformed into numerical feature vectors using `TfidfVectorizer`:

* **Max Features:** 5,000 top n-gram vocabulary features.
* **N-gram Range:** `(1, 2)` (Unigrams and Bigrams to capture word pairs).
* **Sublinear TF:** Scaling applied to diminish the dominance of repetitive words.

---

## Machine Learning Model

* **Algorithm:** Logistic Regression (`C=1.0`, `max_iter=1000`, `random_state=42`)
* **Decision Output:** Binary probability output via Sigmoid decision boundary (`predict_proba`).

---

## Training

The model training pipeline is executed via `src/train_model.py` or the `notebooks/model_training.ipynb` Jupyter Notebook.

* **Data Split:** 80% Training Set (32,345 samples) / 20% Holdout Testing Set (8,087 samples), stratified by class label.
* **Artifacts Saved:** 
  * Trained Model: `model/model.pkl`
  * Vectorizer: `model/vectorizer.pkl`
  * Computed Metrics: `model/metrics.json`

---

## Evaluation & Results

Evaluated on 8,087 unseen holdout test reviews:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **88.88%** |
| **Precision** | **88.29%** |
| **Recall** | **89.66%** |
| **F1-Score** | **88.97%** |

### Confusion Matrix (Test Set)

| Actual \ Predicted | Predicted Fake | Predicted Genuine |
| :--- | :---: | :---: |
| **Actual Fake** | **3,562** (TN) | 481 (FP) |
| **Actual Genuine** | 418 (FN) | **3,626** (TP) |

---

## Project Structure

```
fake-review-detection/
│
├── app.py                  # Flask web application server
├── requirements.txt        # Python dependency specifications
├── README.md               # Project documentation
│
├── data/
│   └── reviews.csv         # Labeled reviews dataset
│
├── model/
│   ├── model.pkl           # Saved Logistic Regression model
│   ├── vectorizer.pkl      # Saved TF-IDF vectorizer
│   └── metrics.json        # Dynamic evaluation metrics output
│
├── notebooks/
│   └── model_training.ipynb # Data exploration & interactive training notebook
│
├── src/
│   ├── preprocessing.py    # Text cleaning and preprocessing module
│   └── train_model.py      # Retraining pipeline script
│
├── templates/
│   ├── index.html          # Detector home page template
│   ├── about.html          # Pipeline explanation page template
│   └── model.html          # Performance metrics page template
│
└── static/
    ├── style.css           # Styling & responsive design rules
    └── script.js            # Client-side JavaScript & AJAX handlers
```

---

## How to Run

### 1. Prerequisites
Ensure Python 3.8+ is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain Model
To retrain the model from scratch on `data/reviews.csv`:
```bash
python src/train_model.py
```

### 4. Launch the Web Application
```bash
python app.py
```

Open your browser and navigate to:
`http://127.0.0.1:5001`

---

## Limitations

* **Dataset Bias:** Model predictions depend on patterns learned from the specific training dataset.
* **Textual Scope:** Textual patterns alone cannot conclusively prove a review is fake; metadata (user IP, posting velocity, user history) is not incorporated.
* **Domain Adaptation:** The model may experience variation in performance when applied to review formats outside standard e-commerce products (e.g., short app store reviews).
* **Probabilistic Nature:** Confidence scores represent model statistical probabilities, not absolute certainty.

---

## Future Improvements

* **Algorithmic Comparison:** Benchmarking performance against Support Vector Machines (SVM), Naive Bayes, and Random Forests.
* **Feature Engineering:** Adding readability indices, uppercase ratio, punctuation frequency, and sentiment analysis scores.
* **Deep Learning & Transformers:** Experimenting with DistilBERT or RoBERTa fine-tuning for contextual language analysis.
* **Multi-Domain Data:** Expanding training data across food delivery, travel/hotel, and mobile application reviews.
