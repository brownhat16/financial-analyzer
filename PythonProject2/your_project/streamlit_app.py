import streamlit as st
from PIL import Image
import requests

API_URL = "http://localhost:8000/analyze-image/"  # FastAPI should be running here

st.set_page_config(page_title="Financial Image Tagger", layout="centered")
st.title("📊 Financial Image Tag Analyzer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()
                st.success("Tags extracted:")
                for category, tags in result["results"].items():
                    st.subheader(category)
                    st.json(tags)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
