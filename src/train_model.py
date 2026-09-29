import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Import our custom preprocessing module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import clean_text

def train():
    # File paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'reviews.csv')
    model_dir = os.path.join(base_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    print("1. Loading dataset from:", data_path)
    df = pd.read_csv(data_path)
    print(f"   Raw dataset shape: {df.shape}")
    
    # 2. Inspect and clean data / Handle missing values
    df = df.dropna(subset=['text']).copy()
    
    # Ensure label column is clean. Map CG -> 0 (Fake), OR -> 1 (Genuine)
    # If label is already Fake/Genuine or CG/OR:
    label_map = {'CG': 0, 'OR': 1, 'Fake': 0, 'Genuine': 1, 0: 0, 1: 1}
    df['target'] = df['label'].map(label_map)
    
    # Drop rows where target couldn't be mapped
    df = df.dropna(subset=['target']).copy()
    df['target'] = df['target'].astype(int)
    
    print(f"   Cleaned dataset count: {len(df)}")
    print("   Label distribution:")
    print("   Fake (0):", (df['target'] == 0).sum())
    print("   Genuine (1):", (df['target'] == 1).sum())
    
    # 3. Clean text using preprocessing module
    print("2. Cleaning review text (lowercasing, punctuation, stopwords)...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Filter out empty cleaned strings
    df = df[df['clean_text'].str.strip() != ""].copy()
    
    X = df['clean_text']
    y = df['target']
    
    # 4. Train/test split (80% train, 20% test)
    print("3. Splitting dataset into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 5. Feature Extraction via TF-IDF Vectorizer
    print("4. Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 6. Model Training - Logistic Regression
    print("5. Training Logistic Regression Classifier...")
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # 7. Model Evaluation
    print("6. Evaluating model performance on test set...")
    y_pred = model.predict(X_test_tfidf)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary')
    rec = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)
    
    # Confusion matrix breakdown: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    print("\n--- MODEL EVALUATION RESULTS ---")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print("\nConfusion Matrix:")
    print(f"  True Negatives (Fake predicted as Fake):       {tn}")
    print(f"  False Positives (Fake predicted as Genuine):   {fp}")
    print(f"  False Negatives (Genuine predicted as Fake):   {fn}")
    print(f"  True Positives (Genuine predicted as Genuine): {tp}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Fake', 'Genuine']))
    
    # 8. Save model, vectorizer, and metrics
    model_path = os.path.join(model_dir, 'model.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    metrics_path = os.path.join(model_dir, 'metrics.json')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    metrics = {
        "dataset_size": int(len(df)),
        "num_classes": 2,
        "class_names": ["Fake Review", "Genuine Review"],
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "algorithm": "Logistic Regression (C=1.0, max_iter=1000)",
        "tfidf_config": {
            "max_features": 5000,
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "stop_words": "English (NLTK)"
        },
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist()
        }
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\nSaved trained model to: {model_path}")
    print(f"Saved vectorizer to: {vectorizer_path}")
    print(f"Saved evaluation metrics to: {metrics_path}")

if __name__ == "__main__":
    train()
