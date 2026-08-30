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

# ---------------------------------------------------------------------------
# Dropdown category buckets. Each (label, value) pair: the label is shown to
# the user, and `value` (the bucket's midpoint) is what actually gets fed to
# the model. Ranges are based on the real distribution of the training data.
# ---------------------------------------------------------------------------
AGE_BUCKETS = [
    ("13 to 17", 15), ("18 to 21", 19.5), ("22 to 25", 23.5),
    ("26 to 29", 27.5), ("30 to 33", 31.5), ("34 to 40", 37),
    ("41 to 55", 48), ("56 to 90", 70),
]
SCREEN_TIME_BUCKETS = [
    ("0 to 2 hrs", 1), ("2 to 4 hrs", 3), ("4 to 6 hrs", 5),
    ("6 to 8 hrs", 7), ("8 to 10 hrs", 9), ("10 to 12 hrs", 11),
    ("12+ hrs", 14),
]
SOCIAL_MEDIA_BUCKETS = [
    ("0 to 1 hr", 0.5), ("1 to 2 hrs", 1.5), ("2 to 3 hrs", 2.5),
    ("3 to 4 hrs", 3.5), ("4 to 6 hrs", 5), ("6+ hrs", 7),
]
GAMING_BUCKETS = [
    ("0 to 0.5 hr", 0.25), ("0.5 to 1 hr", 0.75), ("1 to 1.5 hrs", 1.25),
    ("1.5 to 2 hrs", 1.75), ("2 to 3 hrs", 2.5), ("3+ hrs", 3.5),
]
WORK_STUDY_BUCKETS = [
    ("0 to 1 hr", 0.5), ("1 to 2 hrs", 1.5), ("2 to 3 hrs", 2.5),
    ("3 to 4 hrs", 3.5), ("4 to 5 hrs", 4.5), ("5+ hrs", 5.5),
]
SLEEP_BUCKETS = [
    ("Less than 4 hrs", 3.5), ("4 to 5 hrs", 4.5), ("5 to 6 hrs", 5.5),
    ("6 to 7 hrs", 6.5), ("7 to 8 hrs", 7.5), ("8 to 10 hrs", 9),
]
NOTIFICATIONS_BUCKETS = [
    ("0 to 25", 12), ("25 to 50", 37), ("50 to 100", 75),
    ("100 to 150", 125), ("150 to 200", 175), ("200+", 225),
]
APP_OPENS_BUCKETS = [
    ("0 to 20", 10), ("20 to 40", 30), ("40 to 60", 50),
    ("60 to 90", 75), ("90 to 120", 105), ("120+", 150),
]
WEEKEND_SCREEN_BUCKETS = [
    ("0 to 4 hrs", 2), ("4 to 6 hrs", 5), ("6 to 8 hrs", 7),
    ("8 to 10 hrs", 9), ("10 to 12 hrs", 11), ("12 to 15 hrs", 13.5),
    ("15+ hrs", 17),
]


def bucket_select(label, buckets, default_index=0):
    """Show a dropdown of range labels, return the numeric value behind the chosen one."""
    choice = st.selectbox(label, options=[b[0] for b in buckets], index=default_index)
    value = dict(buckets)[choice]
    return value


model_path = "model.pkl"


@st.cache_resource
def load_model():
    with open(model_path, "rb") as f:
        return pickle.load(f)


model = load_model()

st.title("📱 Screen-Time Addiction Risk Predictor")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = bucket_select("Age", AGE_BUCKETS, default_index=2)
        daily_screen_time_hours = bucket_select("Daily screen time", SCREEN_TIME_BUCKETS, default_index=3)
        social_media_hours = bucket_select("Social media use", SOCIAL_MEDIA_BUCKETS, default_index=1)
        gaming_hours = bucket_select("Gaming", GAMING_BUCKETS, default_index=1)
        work_study_hours = bucket_select("Work/study screen time", WORK_STUDY_BUCKETS, default_index=2)
    with col2:
        sleep_hours = bucket_select("Sleep", SLEEP_BUCKETS, default_index=4)
        notifications_per_day = bucket_select("Notifications per day", NOTIFICATIONS_BUCKETS, default_index=2)
        app_opens_per_day = bucket_select("App opens per day", APP_OPENS_BUCKETS, default_index=2)
        weekend_screen_time = bucket_select("Weekend screen time", WEEKEND_SCREEN_BUCKETS, default_index=2)

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
        st.error(f"⚠️ Higher addiction risk — confidence: **{confidence:.1%}**")
    else:
        st.success(f"✅ Lower addiction risk — confidence: **{confidence:.1%}**")
    st.progress(min(max(proba, 0.0), 1.0))


