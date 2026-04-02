import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.stTextArea textarea {
    border-radius: 10px;
}
.big-title {
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    color: #2c3e50;
}
.subtitle {
    text-align: center;
    color: #7f8c8d;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
.fake {
    background-color: #ffe6e6;
    color: #c0392b;
}
.genuine {
    background-color: #e8f8f5;
    color: #27ae60;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="big-title">🤖 Fake Review Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze product reviews using AI</div>', unsafe_allow_html=True)

st.write("")

# Input box
review = st.text_area("✍️ Enter your review here:", height=150, placeholder="Type a product review...")

# Analyze button
if st.button("🔍 Analyze Review"):

    if review.strip() == "":
        st.warning("⚠️ Please enter a review first!")
    else:
        with st.spinner("Analyzing..."):

            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={"text": review}
                )

                result = response.json()

                prediction = result["prediction"]
                confidence = result["confidence"]

                st.write("")

                # Display result with styling
                if prediction.lower() == "fake":
                    st.markdown(f"""
                    <div class="result-box fake">
                        🚨 Fake Review Detected<br>
                        Confidence: {confidence:.2f}
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown(f"""
                    <div class="result-box genuine">
                        ✅ Genuine Review<br>
                        Confidence: {confidence:.2f}
                    </div>
                    """, unsafe_allow_html=True)

            except:
                st.error("❌ Backend not running. Please start FastAPI.")

# Divider
st.write("---")

# Extra Features Section
st.subheader("💡 Tips")

st.info("""
✔ Fake reviews often use exaggerated words like *'amazing', 'best ever'*  
✔ Genuine reviews are more balanced and descriptive  
✔ Try testing different types of reviews!
""")

# Footer
st.write("")
st.caption("Made with ❤️ using Machine Learning & NLP")