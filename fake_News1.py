import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

####################################################################################3
df = pd.read_csv("news.csv")

# Combine title + text into a single field
df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)

print("Dataset Shape:", df.shape)
print("Label Distribution:\n", df['label'].value_counts())
print("Dataset Head" , df.head)




