import pickle
import re

with open("emotion_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)

text = input("Enter text: ")

text = text.lower()
text = re.sub(r'[^a-zA-Z\\s]', '', text)

vector = tfidf.transform([text])

prediction = model.predict(vector)

print("Predicted Emotion:", prediction[0])