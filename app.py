import streamlit as st
import joblib
import re
import string

# Page config
st.set_page_config(page_title="Fake News Detector", layout="wide")

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Custom CSS styling
st.markdown(
    """
    <style>
    .stTextArea textarea {
        background-color: #f1f3f5;
        font-size: 16px;
        color: #212529;
    }
    .title {
        font-size: 38px;
        font-weight: bold;
        color: #0c0c0c;
        margin-bottom: 0px;
    }
    .tagline {
        font-size: 18px;
        color: #333333;
        margin-top: 5px;
    }
    .footer {
        font-size: 13px;
        color: #555555;
        margin-top: 50px;
        text-align: center;
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

# --- UI Starts ---

# Logo + Header
st.image("flat-newspaper-news-round-icon-701751695035110ggaeu2amu9 (2).png", width=150)
st.markdown('<p class="title">Fake News Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Using machine learning to help verify the truth — instantly.</p>', unsafe_allow_html=True)
st.write("---")

# Layout: Input & Tips side by side
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📝 Paste your news article or headline here:", height=200)

with col2:
    st.markdown("#### 🧠 Quick Tips")
    st.markdown("- Use complete headlines or article excerpts.")
    st.markdown("- Avoid emojis, links, or gibberish.")
    st.markdown("- The model works best on English news content.")

st.write("")

# Prediction button + result
if st.button("🔍 Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter some content to analyze.")
    else:
        cleaned = clean_text(text_input)
        vect = vectorizer.transform([cleaned])
        result = model.predict(vect)[0]

        if result == 1:
            st.success("✅ This article is likely **REAL**.")
            st.balloons()
        else:
            st.error("🚨 This article is likely **FAKE**.")

# Footer
st.markdown('<div class="footer">This tool is for educational purposes only and may not be 100% accurate.</div>', unsafe_allow_html=True)
