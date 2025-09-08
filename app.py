import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


#****************************************************** Load model and preprocessing tools ********************************************

model = tf.keras.models.load_model("Twitter-Sentiment-Analysis.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


#************************************************************ Preprocess function ********************************************************
# -----------------------
def preprocess_text(text):
    # Convert text to sequences
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post')  
    return padded  


#**************************************************************** Streamlit UI **************************************************************

st.title("🐦 Twitter Sentiment Analysis")
st.write("Enter a tweet below to predict its sentiment.")

# User input
user_input = st.text_area("Type a tweet here...")

if st.button("Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        processed = preprocess_text(user_input)
        prediction = model.predict(processed)
        predicted_class = np.argmax(prediction, axis=1)[0]
        sentiment = label_encoder.inverse_transform([predicted_class])[0]

        # Display results
        if sentiment.lower() == "positive":
            st.success(f"😊 Sentiment: {sentiment}")
        elif sentiment.lower() == "negative":
            st.error(f"😡 Sentiment: {sentiment}")
        else:
            st.info(f"😐 Sentiment: {sentiment}")
