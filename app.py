import streamlit as st
import joblib
import os

st.set_page_config(page_title="Spam Email Detector")

st.title("📧 Spam Email Detector")
st.markdown("Enter an email message below and click **Predict** to see if it's spam or not.")

# Debug log
st.info("Model loading...")

try:
    if os.path.exists("spam_detector_model.pkl") and os.path.exists("vectorizer.pkl"):
        model = joblib.load("spam_detector_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        st.success("Model and vectorizer loaded successfully.")
    else:
        st.error("Model or vectorizer file is missing!")
        st.stop()
except Exception as e:
    st.error(f"Error loading model/vectorizer: {e}")
    st.stop()

email_input = st.text_area("✍️ Type your email here:", height=200)

if st.button("🚀 Predict"):
    try:
        if email_input.strip() == "":
            st.warning("Please enter some text before predicting.")
        else:
            input_vector = vectorizer.transform([email_input])
            prediction = model.predict(input_vector)[0]
            if prediction == 1:
                st.error("🚫 This is SPAM!")
            else:
                st.success("✅ This is NOT spam.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

