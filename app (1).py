
import io
import re

import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter

st.set_page_config(page_title="Fake Coursera Certificate Detection", layout="wide")

DATASET_PATH = "synthetic_coursera_registry.csv"
URL_RE = re.compile(r'https?://[^\s]+|www\.[^\s]+', re.IGNORECASE)


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r'[^a-z0-9:/._ -]+', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def normalize_url(s: str) -> str:
    s = normalize_text(s)
    s = s.rstrip('.,;:)')
    s = s.replace('coursera,org', 'coursera.org').replace('coursera org', 'coursera.org')
    s = s.replace('http //', 'http://').replace('https //', 'https://')
    s = s.replace('http:/', 'http://').replace('https:/', 'https://')
    if s.startswith('www.'):
        s = 'https://' + s
    return s


def token_set(s: str):
    return set(normalize_text(s).split())


def score_match(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa), len(sb))


def read_image_from_upload(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        doc = fitz.open(stream=data, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img, "PDF"
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img, "Image"


def run_ocr(image_pil):
    img_np = np.array(image_pil)
    data = pytesseract.image_to_data(
        img_np,
        output_type=pytesseract.Output.DATAFRAME,
        config="--oem 3 --psm 6"
    )
    data = data.fillna("")
    text = " ".join([str(x) for x in data["text"].tolist() if str(x).strip()])
    return text, data


def extract_url(text: str) -> str:
    matches = URL_RE.findall(text)
    if matches:
        return normalize_url(matches[0])
    t = normalize_text(text)
    m = re.search(r'(https?://)?(www\.)?coursera\.org/[a-z0-9/_\-?=&.%]+', t)
    if m:
        url = m.group(0)
        if not url.startswith("http"):
            url = "https://" + url.lstrip("/")
        return normalize_url(url)
    return ""


def find_text_bbox(ocr_df: pd.DataFrame, phrase: str):
    tokens = [t for t in normalize_text(phrase).split() if t]
    if not tokens or ocr_df.empty:
        return None
    df = ocr_df.copy()
    df["norm"] = df["text"].astype(str).apply(normalize_text)
    matched = df[df["norm"].isin(tokens)]
    if matched.empty:
        return None
    x1 = int(matched["left"].min())
    y1 = int(matched["top"].min())
    x2 = int((matched["left"] + matched["width"]).max())
    y2 = int((matched["top"] + matched["height"]).max())
    return (x1, y1, x2, y2)


def visual_tamper_heatmap(image_pil):
    rgb = np.array(image_pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = cv2.absdiff(gray, blur)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    heat = cv2.addWeighted(diff, 0.6, lap, 0.4, 0)
    heat = cv2.GaussianBlur(heat, (9, 9), 0)
    heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)

    mask = np.zeros_like(heat)
    mask[heat > 165] = 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr, 0.68, heatmap, 0.32, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), mask


def draw_bbox(image_pil, bbox, color=(255, 0, 0), width=4):
    img = image_pil.copy()
    if bbox:
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox, outline=color, width=width)
    return img


@st.cache_data
def load_registry():
    df = pd.read_csv(DATASET_PATH)
    df["credential_url_norm"] = df["credential_url"].apply(normalize_url)
    return df


def get_font(size=24):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def build_demo_certificate(row, tamper_name=False, wrong_url=False):
    w, h = 1400, 1000
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title = get_font(52)
    h1 = get_font(34)
    h2 = get_font(28)
    body = get_font(26)
    small = get_font(20)

    d.rectangle([18, 18, w - 18, h - 18], outline=(0, 87, 184), width=6)
    d.rectangle([40, 40, w - 40, h - 40], outline=(226, 190, 102), width=3)
    d.text((80, 90), "COURSERA CERTIFICATE", font=title, fill=(0, 70, 140))
    d.text((80, 190), "This certifies that", font=h1, fill="black")

    display_name = row["candidate_name"]
    if tamper_name:
        display_name = "Rahul Edited"

    d.text((80, 260), display_name, font=title, fill=(140, 20, 30))
    d.line((80, 330, 700, 330), fill=(140, 20, 30), width=2)
    d.text((80, 390), "successfully completed the Coursera course", font=h2, fill="black")
    d.text((80, 450), row["course_name"], font=h1, fill=(0, 90, 120))
    d.text((80, 560), f"Issue Date: {row['issue_date']}", font=body, fill="black")

    url = row["credential_url"]
    if wrong_url:
        url = url.replace(row["credential_id"], "BADLINK999")

    d.text((80, 680), "Credential URL:", font=body, fill="black")
    d.text((80, 730), url, font=small, fill=(10, 70, 150))
    d.text((1000, 840), "Coursera", font=h1, fill="black")

    if tamper_name:
        bbox = (70, 248, 720, 335)
        patch = img.crop(bbox).filter(ImageFilter.GaussianBlur(radius=1.4))
        img.paste(patch, bbox)

    return img


def evaluate_certificate(image_pil, text, ocr_df, registry):
    extracted_url = extract_url(text)
    official_record = None
    reasons = []
    extracted_fields = {
        "candidate_name_from_text": "",
        "course_name_from_text": "",
        "issue_date_from_text": "",
        "credential_url": extracted_url or "Not found",
    }

    name_bbox = None
    course_bbox = None

    if extracted_url:
        exact = registry[registry["credential_url_norm"] == normalize_url(extracted_url)]
        if not exact.empty:
            official_record = exact.iloc[0]
        else:
            reasons.append("Credential URL not found in the reference registry.")
    else:
        reasons.append("Credential URL could not be extracted from the certificate.")

    if official_record is not None:
        extracted_fields["candidate_name_from_text"] = official_record["candidate_name"]
        extracted_fields["course_name_from_text"] = official_record["course_name"]
        extracted_fields["issue_date_from_text"] = official_record["issue_date"]

        name_bbox = find_text_bbox(ocr_df, official_record["candidate_name"])
        course_bbox = find_text_bbox(ocr_df, official_record["course_name"])

        name_score = score_match(text, official_record["candidate_name"])
        course_score = score_match(text, official_record["course_name"])
        date_score = score_match(text, official_record["issue_date"])

        if name_score < 1.0:
            reasons.append("Name on the certificate does not fully match the credential record.")
        if course_score < 0.7:
            reasons.append("Course name does not match the credential record.")
        if date_score < 0.6:
            reasons.append("Issue date could not be confidently matched to the credential record.")
    else:
        best_name_idx = registry["candidate_name"].apply(lambda x: score_match(text, x)).idxmax()
        extracted_fields["candidate_name_from_text"] = registry.loc[best_name_idx, "candidate_name"]
        extracted_fields["course_name_from_text"] = ""
        extracted_fields["issue_date_from_text"] = ""

    overlay, mask = visual_tamper_heatmap(image_pil)
    mask_area_ratio = float((mask > 0).sum()) / float(mask.size)

    if mask_area_ratio > 0.02:
        reasons.append("Visual anomaly suggests possible tampering in a text region.")

    is_fake = any([
        "not found" in " ".join(reasons).lower(),
        "does not fully match" in " ".join(reasons).lower(),
        "does not match" in " ".join(reasons).lower(),
        mask_area_ratio > 0.02
    ])

    if official_record is not None:
        base = 0.55
        if not is_fake:
            confidence = min(0.98, base + 0.20)
        else:
            confidence = min(0.98, base + 0.15 + min(mask_area_ratio * 5, 0.20))
    else:
        confidence = 0.90 if is_fake else 0.60

    label = "Fake Certificate" if is_fake else "Real Certificate"

    highlight = image_pil.copy()
    draw = ImageDraw.Draw(highlight)
    if name_bbox and official_record is not None:
        if any("name on the certificate" in r.lower() for r in reasons):
            draw.rectangle(name_bbox, outline=(255, 0, 0), width=5)
        else:
            draw.rectangle(name_bbox, outline=(0, 180, 0), width=4)
    if course_bbox and official_record is not None and any("course name" in r.lower() for r in reasons):
        draw.rectangle(course_bbox, outline=(255, 165, 0), width=4)

    official = None
    if official_record is not None:
        official = {
            "candidate_name": official_record["candidate_name"],
            "course_name": official_record["course_name"],
            "issue_date": official_record["issue_date"],
            "credential_url": official_record["credential_url"],
            "credential_id": official_record["credential_id"],
        }

    return {
        "label": label,
        "confidence": confidence,
        "reasons": reasons if reasons else ["Credential URL and extracted details match the reference registry."],
        "overlay": overlay,
        "mask": mask,
        "highlight": highlight,
        "official_record": official,
        "extracted_fields": extracted_fields,
        "mask_area_ratio": mask_area_ratio,
    }


registry = load_registry()

st.title("Fake Coursera Certificate Detection")
st.write(
    "Upload a Coursera certificate image or PDF. The app extracts the credential URL, "
    "matches it against a synthetic reference registry, checks for name/course mismatch, "
    "and highlights suspicious regions."
)

tab1, tab2 = st.tabs(["Upload Certificate", "Demo Generator"])

selected_image = None
source_label = ""

with tab1:
    uploaded_file = st.file_uploader("Upload certificate image or PDF", type=["png", "jpg", "jpeg", "pdf"])
    if uploaded_file is not None:
        selected_image, source_kind = read_image_from_upload(uploaded_file)
        source_label = f"Uploaded {source_kind}"

with tab2:
    choice = st.selectbox(
        "Choose sample credential",
        registry["credential_id"] + " — " + registry["candidate_name"] + " — " + registry["course_name"]
    )
    demo_mode = st.radio("Demo type", ["Real sample", "Fake name sample", "Invalid link sample"], horizontal=True)
    if st.button("Generate demo certificate"):
        idx = choice.split(" — ")[0]
        row = registry[registry["credential_id"] == idx].iloc[0]
        selected_image = build_demo_certificate(
            row,
            tamper_name=(demo_mode == "Fake name sample"),
            wrong_url=(demo_mode == "Invalid link sample")
        )
        source_label = demo_mode

if selected_image is not None:
    ocr_text, ocr_df = run_ocr(selected_image)
    result = evaluate_certificate(selected_image, ocr_text, ocr_df, registry)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Uploaded / Generated Certificate")
        st.image(selected_image, use_container_width=True)
    with c2:
        st.subheader("Result")
        if result["label"].startswith("Fake"):
            st.error(result["label"])
        else:
            st.success(result["label"])
        st.metric("Confidence", f"{result['confidence']:.1%}")
        st.write(f"**Source:** {source_label}")
        st.write("**Reason(s):**")
        for r in result["reasons"]:
            st.write(f"- {r}")

    st.subheader("Extracted Fields")
    extracted_df = pd.DataFrame(
        [{"field": k, "value": v} for k, v in result["extracted_fields"].items()]
    )
    st.dataframe(extracted_df, use_container_width=True, hide_index=True)

    if result["official_record"] is not None:
        st.subheader("Reference Record Retrieved from Credential URL")
        official_df = pd.DataFrame(
            [{"field": k, "value": v} for k, v in result["official_record"].items()]
        )
        st.dataframe(official_df, use_container_width=True, hide_index=True)

    x1, x2 = st.columns(2)
    with x1:
        st.subheader("Tampered Region Heatmap")
        st.image(result["overlay"], use_container_width=True)
    with x2:
        st.subheader("Highlighted Suspicious / Mismatch Region")
        st.image(result["highlight"], use_container_width=True)

    with st.expander("View OCR Text"):
        st.text(ocr_text)

    with st.expander("View Binary Tamper Mask"):
        st.image(result["mask"], use_container_width=True, clamp=True)

st.caption(
    "Demo note: this app uses a synthetic Coursera-like registry and OCR-based verification. "
    "Core logic: if the certificate URL maps to an official record but the visible name differs, "
    "the certificate is flagged as fake."
)
