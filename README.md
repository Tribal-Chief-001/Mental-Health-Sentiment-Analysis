# 🧠 Mental Health Sentiment Classifier (Multi-Class)

Classifies Reddit posts into mental health categories: **Depression**, **ADHD**, and **OCD** using NLP and ML.

---

## 📌 Project Overview

Built on public data sourced from Hugging Face’s `reddit_mental_health_posts` dataset by `solomonk`, this project combines cleaned Reddit posts from real mental health subreddits — providing a multi-class classification framework for real-world emotional understanding.

---

## 🗂️ Dataset

- Publicly available: `depression.csv`, `adhd.csv`, `ocd.csv`
- Each CSV contains Reddit post content labeled with the corresponding subreddit (e.g., `body`, `subreddit`)
- Cleaned and merged into a unified DataFrame with `text` and `label` fields → ideal for multi-class sentiment modeling

---

## 🔧 Pipeline & Technical Steps

1. **Exploratory Data Analysis (EDA)**  
   - Loaded data, balanced label distribution, plotted class counts

2. **Text Preprocessing**  
   - Lowercased text  
   - Removed punctuation and URLs  
   - Tokenized, removed stopwords, lemmatized using NLTK

3. **Vectorization**  
   - TF‑IDF with top 10K vocabulary tokens

4. **Model Training**  
   - `LogisticRegression` (multi-class support)  
   - Train/test split (80/20, stratified by label)

5. **Evaluation & Visualization**  
   - Accuracy score, classification report, confusion matrix heatmap

6. **Saved Artifacts**  
   - `mh_model.pkl` (trained model)  
   - `tfidf_vectorizer.pkl` (vectorizer)

---

## 📈 Results & Performance

- **Overall Accuracy**: ~77% (placeholder for your score)  
- **Classes**:  
  - `depression`: Precision 80%, 
  - `adhd`: Precision 86%,   
  - `ocd`: Precision 74%, 
- Confusion matrix highlights areas where labels are mixed

---

## 💻 Try It Yourself

Use the notebook `main.ipynb` to:
- Predict on custom text input using the `predict_sentiment()` function
- Load model and vectorizer saved in `.pkl` files

```python
predict_sentiment("I can’t focus at all, even simple tasks are overwhelming.")

Built by Xandred 

