# Air Quality Category Predictor

Predicts air quality category (Good / Moderate / Poor / Severe) from sensor
readings, trained on the UCI Air Quality dataset.

## Project structure

```
air_quality_project/
├── AirQualityUCI.csv     <- you add this (see step 1)
├── train_model.py        <- trains and saves the model
├── app.py                <- Streamlit web app
├── requirements.txt
└── README.md
```

## Step 1 — Get the dataset

1. Go to https://archive.ics.uci.edu/dataset/360/air+quality
2. Download the zip, extract `AirQualityUCI.csv`
3. Place it in this project folder

(Kaggle also mirrors this dataset if you prefer:
search "Air Quality UCI" on kaggle.com/datasets)

## Step 2 — Set up your environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 3 — Train the model

```bash
python train_model.py
```

This prints data cleaning info, class distribution, a classification
report, and feature importances — then saves three files:
`aqi_model.pkl`, `label_encoder.pkl`, `feature_cols.pkl`.

## Step 4 — Run the app locally

```bash
streamlit run app.py
```

Open the local URL it prints (usually http://localhost:8501) and try
some predictions.

## Step 5 — Push to GitHub

```bash
git init
git add .
git commit -m "Air quality category predictor"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

Make sure `aqi_model.pkl`, `label_encoder.pkl`, and `feature_cols.pkl`
are committed too (the app needs them at runtime). You can leave
`AirQualityUCI.csv` out of the repo if you want — it's only needed for
training, not for the deployed app.

## Step 6 — Deploy to the cloud (Streamlit Community Cloud, free)

1. Go to https://share.streamlit.io and sign in with GitHub
2. Click **"New app"**
3. Select your repo, branch (`main`), and set the main file path to `app.py`
4. Click **Deploy**

That's it — Streamlit installs `requirements.txt` automatically and gives
you a public URL like `https://<your-app-name>.streamlit.app` that you
can share or put on your resume/portfolio.

### Alternative: Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space
2. Choose **Streamlit** as the SDK
3. Push this same repo to the Space (git remote, same as GitHub)
4. It auto-builds and hosts your app at `https://huggingface.co/spaces/<you>/<space-name>`

## Notes on the AQI label

The UCI dataset only has raw pollutant/sensor readings — no PM2.5/PM10
and no official AQI. `train_model.py` creates a **simplified proxy AQI
category** using breakpoints on CO and NO2 concentrations. If you want
an accurate real-world AQI, swap in a dataset that includes PM2.5/PM10
and use the official EPA or CPCB breakpoint tables instead.
