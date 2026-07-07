Air Quality Index (AQI) Classifier 🌫️

A machine learning model that predicts how polluted the air is — Good,
Moderate, Poor, or Severe — using real air-quality sensor readings from
the UCI Air Quality dataset.

What this project does

Air quality monitoring stations record pollutant levels (CO, NO2, NOx,
benzene, etc.) throughout the day. Instead of manually checking a dozen
different pollutant numbers to figure out if the air is safe to breathe,
this project trains a Random Forest classifier to instantly label a
given set of sensor readings into one of four easy-to-understand
categories.

It's a simple, end-to-end supervised learning pipeline: raw sensor data →
cleaned data → labeled data → trained model → evaluated model → saved model
ready for reuse.

Why this is useful


Turns raw numbers into a human-readable label — instead of reading
"CO(GT) = 3.4, NO2(GT) = 95", you get "Poor".
Reusable model — once trained, the model is saved to disk and can be
loaded again anytime to classify new readings without retraining.
Good learning project — covers real-world data cleaning, feature
engineering, classification, and model evaluation in one compact script.


Dataset


Source: UCI Air Quality Dataset (AirQualityUCI.csv)
Contains hourly averaged readings from a gas multisensor device deployed
in an Italian city, including pollutant concentrations and weather
readings (temperature, humidity).
Missing values in the raw dataset are marked as -200, which this
project detects and cleans up.


How it works (pipeline overview)


Load the data
Reads AirQualityUCI.csv and drops any stray "Unnamed" columns that
come from formatting quirks in the raw file.
Clean the data

Replaces the placeholder value -200 (used in this dataset for
missing readings) with proper NaN.
Drops rows where the two most important pollutants — CO(GT) and
NO2(GT) — are missing, since these are used to create the labels.
Fills any remaining missing values in the other sensor columns with
that column's median value.



Create the target labels (AQI category)
Since the raw dataset doesn't come with a ready-made "how bad is the
air" label, this project builds one using standard pollutant
thresholds:
CO (mg/m³)NO2 (µg/m³)Category< 1< 40Good1–240–80Moderate2–1080–180Poor> 10> 180Severe

Each row gets the worse of its CO-based and NO2-based category,
so if either pollutant is high, the air is marked accordingly.
Prepare features & labels

Features (X): 13 sensor readings — CO, NMHC, Benzene (C6H6), NOx,
NO2, several metal-oxide sensor outputs (PT08.S1–S5), Temperature (T),
Relative Humidity (RH), and Absolute Humidity (AH).
Labels (y): the AQI category created in step 3, encoded into
numbers using LabelEncoder.



Split & train
80% of the data is used for training and 20% for testing (with
stratified sampling, so all four categories are fairly represented in
both sets). A Random Forest Classifier (300 trees) is trained on the
training data.
Evaluate
The model predicts on the unseen test set, and prints:

A classification report (precision, recall, F1-score per category)
A confusion matrix (which categories get mixed up with which)
Feature importances (which sensors matter most for the prediction)



Save the model
The trained model, label encoder, and the list of feature columns are
saved as .pkl files so they can be reloaded later without retraining.


Inputs

InputDescriptionAirQualityUCI.csvThe raw sensor dataset. Must be in the same folder as the script.13 feature columnsCO(GT), PT08.S1(CO), NMHC(GT), C6H6(GT), PT08.S2(NMHC), NOx(GT), PT08.S3(NOx), NO2(GT), PT08.S4(NO2), PT08.S5(O3), T, RH, AH — used to predict the AQI category.

Outputs

OutputDescriptionaqi_model.pklThe trained Random Forest model.label_encoder.pklEncoder that maps category names (Good/Moderate/Poor/Severe) to numbers and back.feature_cols.pklThe exact list/order of feature columns the model expects — needed so future predictions use the same input format.Console outputClass distribution, classification report, confusion matrix, and feature importance ranking
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
