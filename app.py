import streamlit as st
import joblib
import re
import string

# Page setup
st.set_page_config(page_title="Fake News Detector", layout="wide")

# Load the saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Optional: Minimal custom CSS for cleaner design
st.markdown(
    """
    <style>
    .stTextArea textarea {
        background-color: #f8f9fa;
        font-size: 16px;
        color: #212529;
    }
    .main-header {
        font-size: 36px;
        font-weight: 600;
        color: #1c1c1c;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 18px;
        color: #6c757d;
        margin-top: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Header
st.markdown('<p class="main-header">Fake News Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Use AI to identify whether a news article is Real or Fake.</p>', unsafe_allow_html=True)
st.write("---")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("Paste a news article or headline:", height=200)

with col2:
    st.markdown("#### Guidelines")
    st.markdown("- Use clear, concise text.")
    st.markdown("- Avoid emojis, links, or non-news content.")
    st.markdown("- Suitable for headlines or short articles.")

st.write("")

# Prediction
if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter some content to analyze.")
    else:
        cleaned = clean_text(text_input)
        vect = vectorizer.transform([cleaned])
        result = model.predict(vect)[0]

        if result == 1:
            st.success("Result: This article is likely **REAL**.")
        else:
            st.error("Result: This article is likely **FAKE**.")

# Clean footer line
st.write("---")
