#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# # 1. Load Dataset

# In[3]:


df = pd.read_csv("news.csv")

# Combine title + text into a single field
df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)

print("Dataset Shape:", df.shape)
print("Label Distribution:\n", df['label'].value_counts())
print("Dataset Head" , df.head)


# # 2. Split into Train/Test

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(
    df['content'], df['label'], test_size=0.2, random_state=42)


# # 3. TF-IDF Vectorization

# In[5]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)


# # 4. Passive Aggressive Classifier
# 

# In[6]:


pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)


# # 5. Predictions & Evaluation
# 

# In[7]:


y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(score*100, 2)}%")

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# # 6. Save Model 

# In[9]:


import joblib
joblib.dump(pac, "fake_news_model.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved as 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl'")


# In[ ]:




