import streamlit as st
import joblib
import re
import string

# Page setup
st.set_page_config(page_title="Fake News Detector", layout="wide")

# Load the saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Custom CSS styling (optional)
st.markdown(
    """
    <style>
    .stTextArea textarea {
        background-color: #f9f9f9;
        font-size: 16px;
        color: #333;
    }
    .big-font {
        font-size: 22px !important;
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
st.markdown("# 📰 Fake News Detection App")
st.markdown("Enter a news article below and we'll tell you whether it's likely **Real** or **Fake**.")
st.write("---")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📝 Paste your news content here:", height=200)

with col2:
    st.markdown("### 📌 Tips")
    st.markdown("- Use full news headlines or article snippets.")
    st.markdown("- Avoid emojis or URLs.")
    st.markdown("- Results are based on ML predictions — not 100% accurate.")

st.write("")

# Prediction
if st.button("🔍 Predict"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some content to analyze.")
    else:
        cleaned = clean_text(text_input)
        vect = vectorizer.transform([cleaned])
        result = model.predict(vect)[0]

        if result == 1:
            st.success("✅ This appears to be **REAL** news.")
            st.balloons()
        else:
            st.error("🚨 This appears to be **FAKE** news.")

# Footer
st.write("---")
st.markdown("Made with ❤️ by [Your Name](https://github.com/yourusername)")
