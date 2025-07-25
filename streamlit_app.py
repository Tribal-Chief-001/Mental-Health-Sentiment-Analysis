import streamlit as st
import joblib

model = joblib.load("mh_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("Mental Health Sentiment Classifier")
st.write("Classify Reddit-style posts into Depression, ADHD, or OCD")

text = st.text_area("Enter text here...", height=200)
if st.button("Predict"):
    vect = vectorizer.transform([text])
    pred = model.predict(vect)[0]
    st.write("Predicted Condition:", pred)
