import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit UI
st.set_page_config(page_title="Spam Email Detector")
st.title("📧 Spam Email Detector")
st.markdown("Enter an email message below and click **Predict** to see if it's spam or not.")

email_input = st.text_area("✍️ Type your email here:", height=200)

if st.button("🚀 Predict"):
    if email_input.strip() == "":
        st.warning("Please enter some text before predicting.")
    else:
        input_vector = vectorizer.transform([email_input])
        prediction = model.predict(input_vector)[0]
        if prediction == 1:
            st.error("🚫 This is SPAM!")
        else:
            st.success("✅ This is NOT spam.")
