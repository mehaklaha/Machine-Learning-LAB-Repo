import streamlit as st
import requests
import time

st.title("🛡️ Fake Review Detection (LSTM NLP Model)")

st.markdown("Enter a review to check if it's **genuine** or **fake**.")

review = st.text_area("Enter Review", height=150, placeholder="e.g., This product is amazing...")

if st.button("🔍 Analyze Review", type="primary"):
    if review.strip():
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={"text": review},
                    timeout=10
                )
                response.raise_for_status()
                result = response.json()
                
                st.success("✅ Analysis Complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", result["prediction"])
                with col2:
                    conf = result["confidence"]
                    st.metric("Confidence", f"{conf:.1%}")
                st.caption(f"Raw fake probability: {result['raw_proba']:.4f}")
                
            except requests.exceptions.ConnectionError:
                st.error("❌ Backend not running! Run `uvicorn backend.main:app --reload --port 8000` first.")
            except requests.exceptions.Timeout:
                st.error("❌ Analysis timed out. Try again.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("Please enter a review.")

st.info("💡 **Backend must be running on port 8000.** Check terminal.")

# Update TODO
st.markdown("---")
st.markdown("**Progress:** Files updated. Next: Install deps & train.")
