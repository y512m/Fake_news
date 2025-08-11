#  Fake News Detection with Python

##  Overview
This project detects whether a news article is **REAL** or **FAKE** using:
- **TF-IDF Vectorization** for text feature extraction.
- **PassiveAggressiveClassifier** for classification.

The dataset contains:
- `title` → Headline of the news article.
- `text` → Content of the news article.
- `label` → Either `REAL` or `FAKE`.

---

##  How It Works
1. **Load the Dataset** (`news.csv`).
2. **Combine** `title` + `text` into one field for better accuracy.
3. **Transform text to numeric features** using `TfidfVectorizer`.
4. **Train model** with `PassiveAggressiveClassifier`.
5. **Evaluate** using Accuracy, Classification Report, and Confusion Matrix.

---

