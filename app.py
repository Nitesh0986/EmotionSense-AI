import streamlit as st
import pickle
import re

# Load model
with open("emotion_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)

st.set_page_config(page_title="EmotionSense-AI")

st.title("🧠 EmotionSense-AI")
st.write("Detect emotions from text using NLP and Machine Learning")

text = st.text_area("Enter your text here")

if st.button("Predict Emotion"):

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    vector = tfidf.transform([text])

    prediction = model.predict(vector)

    st.success(f"Predicted Emotion: {prediction[0]}")