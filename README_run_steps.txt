
# Fake Coursera Certificate Detection

Files:
- app.py
- requirements.txt
- packages.txt
- synthetic_coursera_registry.csv

## Run in Google Colab
1. Upload all four files to Colab.
2. Run:
   !apt-get update -qq && apt-get install -y tesseract-ocr
   !pip install -q -r requirements.txt
   !streamlit run app.py & npx localtunnel --port 8501

## Deploy on Streamlit Cloud
1. Push all four files to the repo root.
2. In Streamlit Cloud, set main file path to `app.py`.
3. Streamlit Cloud will use `requirements.txt` and `packages.txt` automatically.
