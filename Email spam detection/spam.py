import streamlit as st
import tensorflow as tf
import pickle

# Load the trained model
model = tf.keras.models.load_model("spam_model.h5")

# Load the TF-IDF vectorizer
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Streamlit app UI
st.title("📩 Email Spam Detector")
st.markdown("Enter an email or message and detect if it's spam using a trained Neural Network.")

message = st.text_area("Enter the message:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a valid message.")
    else:
        vec_msg = tfidf.transform([message]).toarray()
        prediction = model.predict(vec_msg)[0][0]
        if prediction > 0.5:
            st.error("🚨 This message is **SPAM**.")
        else:
            st.success("✅ This message is **NOT SPAM**.")
