import streamlit as st
import numpy as np
import cv2
import re
import pandas as pd
from PIL import Image
import pytesseract

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fake Certificate Detection", layout="wide")

st.title("📜 Fake Coursera Certificate Detection")

# ---------------- LOAD DATASET ----------------
@st.cache_data
def load_registry():
    return pd.read_csv("synthetic_coursera_registry.csv")

registry = load_registry()

# ---------------- OCR FUNCTION ----------------
def run_ocr(image):
    # Convert PIL → OpenCV format
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve OCR accuracy
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    text = pytesseract.image_to_string(gray)

    return text

# ---------------- EXTRACT URL ----------------
def extract_url(text):
    urls = re.findall(r'https?://\S+', text)
    return urls[0] if urls else ""

# ---------------- MATCH FUNCTION ----------------
def match_with_registry(text, url):
    if not url:
        return None, "URL not found"

    match = registry[registry["credential_url"].str.contains(url, na=False)]

    if not match.empty:
        record = match.iloc[0]
        name = record["candidate_name"]

        if name.lower() in text.lower():
            return record, "Match"
        else:
            return record, "Name Mismatch"
    else:
        return None, "URL not found in database"

# ---------------- TAMPERING DETECTION ----------------
def detect_tampering(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    diff = cv2.absdiff(gray, blur)

    edges = cv2.Canny(diff, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x,y), (x+w,y+h), (0,0,255), 2)

    return output

# ---------------- UI ----------------
uploaded_file = st.file_uploader("Upload Certificate", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Certificate", use_container_width=True)

    # OCR
    text = run_ocr(image)

    # URL extraction
    url = extract_url(text)

    # Matching
    record, status = match_with_registry(text, url)

    # Tampering
    tampered = detect_tampering(image)

    with col2:
        if status == "Match":
            st.success("✅ REAL CERTIFICATE")
        elif status == "Name Mismatch":
            st.error("🚨 FAKE CERTIFICATE (Name Mismatch)")
        else:
            st.warning("⚠️ Unable to verify")

        st.write(f"🔗 URL: {url}")

    # Show tampered image
    st.subheader("🚨 Tampered Region Detection")
    st.image(tampered, use_container_width=True)

    # Extracted text
    with st.expander("📄 Extracted Text"):
        st.write(text)
