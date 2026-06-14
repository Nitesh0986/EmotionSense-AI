import pandas as pd
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import re

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load Dataset
df = pd.read_csv(
    "dataset/train.txt",
    sep=";",
    names=["text", "emotion"]
)
print("Dataset Loaded Successfully")
print(df.head())

# Text Cleaning
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [word for word in words
             if word not in stop_words]

    words = [lemmatizer.lemmatize(word)
             for word in words]

    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

# Features and Labels
X = df['clean_text']
y = df['emotion']

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Report
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# Sample Prediction
sample = ["I am feeling very happy today"]

sample_clean = clean_text(sample[0])

sample_vector = tfidf.transform([sample_clean])

prediction = model.predict(sample_vector)

print("\nSample Text:", sample[0])
print("Predicted Emotion:", prediction[0])
with open("emotion_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(tfidf, file)

print("Model Saved Successfully")