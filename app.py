import sys
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Required fix: registers a module alias needed to unpickle the model on a
# fresh environment. Must run BEFORE pickle.load(). Do not remove.
# ---------------------------------------------------------------------------
import sklearn.ensemble  # noqa: F401
try:
    import sklearn._loss._loss as _sklearn_loss_ext
    sys.modules.setdefault("_loss", _sklearn_loss_ext)
except ImportError:
    pass

st.set_page_config(page_title="Screen-Time Addiction Risk Predictor", page_icon="📱")

GENDER_OPTIONS = ["Female", "Male", "Other"]
STRESS_OPTIONS = ["High", "Low", "Medium"]
IMPACT_OPTIONS = ["No", "Yes"]

FEATURES = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time",
    "screen_to_sleep_ratio", "social_share_of_screen", "gaming_share_of_screen",
    "weekend_vs_weekday", "notif_per_app_open", "leisure_vs_work",
    "gender", "stress_level", "academic_work_impact",
]


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()

st.title("📱 Screen-Time Addiction Risk Predictor")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 13, 90, 25)
        daily_screen_time_hours = st.number_input("Daily screen time (hrs)", 0.0, 24.0, 6.0, step=0.25)
        social_media_hours = st.number_input("Social media (hrs/day)", 0.0, 24.0, 2.0, step=0.25)
        gaming_hours = st.number_input("Gaming (hrs/day)", 0.0, 24.0, 1.0, step=0.25)
        work_study_hours = st.number_input("Work/study screen time (hrs/day)", 0.0, 24.0, 3.0, step=0.25)
    with col2:
        sleep_hours = st.number_input("Sleep (hrs/night)", 0.0, 24.0, 7.5, step=0.25)
        notifications_per_day = st.number_input("Notifications/day", 0, 1000, 100)
        app_opens_per_day = st.number_input("App opens/day", 0, 500, 50)
        weekend_screen_time = st.number_input("Weekend screen time (hrs)", 0.0, 24.0, 8.0, step=0.25)

    gender = st.selectbox("Gender", GENDER_OPTIONS)
    stress_level = st.selectbox("Stress level", STRESS_OPTIONS)
    academic_work_impact = st.selectbox("Impacting academic/work performance?", IMPACT_OPTIONS)

    submitted = st.form_submit_button("Predict risk")

if submitted:
    row = pd.DataFrame([{
        "age": age,
        "daily_screen_time_hours": daily_screen_time_hours,
        "social_media_hours": social_media_hours,
        "gaming_hours": gaming_hours,
        "work_study_hours": work_study_hours,
        "sleep_hours": sleep_hours,
        "notifications_per_day": notifications_per_day,
        "app_opens_per_day": app_opens_per_day,
        "weekend_screen_time": weekend_screen_time,
        "gender": gender,
        "stress_level": stress_level,
        "academic_work_impact": academic_work_impact,
    }])

    row["screen_to_sleep_ratio"] = row["daily_screen_time_hours"] / row["sleep_hours"].replace(0, np.nan)
    row["social_share_of_screen"] = row["social_media_hours"] / row["daily_screen_time_hours"].replace(0, np.nan)
    row["gaming_share_of_screen"] = row["gaming_hours"] / row["daily_screen_time_hours"].replace(0, np.nan)
    row["weekend_vs_weekday"] = row["weekend_screen_time"] - row["daily_screen_time_hours"]
    row["notif_per_app_open"] = row["notifications_per_day"] / row["app_opens_per_day"].replace(0, np.nan)
    row["leisure_vs_work"] = (row["social_media_hours"].fillna(0) + row["gaming_hours"].fillna(0)) - row["work_study_hours"]

    row["gender"] = pd.Categorical(row["gender"], categories=GENDER_OPTIONS)
    row["stress_level"] = pd.Categorical(row["stress_level"], categories=STRESS_OPTIONS)
    row["academic_work_impact"] = pd.Categorical(row["academic_work_impact"], categories=IMPACT_OPTIONS)

    proba = model.predict_proba(row[FEATURES])[0, 1]
    confidence = max(proba, 1 - proba)  # how sure the model is in whichever class it picked

    st.divider()
    if proba >= 0.5:
        st.error(f"⚠️ Higher addiction risk —("Confidence score", f"{confidence:.1%}") **")
    else:
        st.success(f"✅ Lower addiction risk — probability: **{proba:.1%}**")
    st.progress(min(max(proba, 0.0), 1.0))

    st.metric("Confidence score", f"{confidence:.1%}")
    
