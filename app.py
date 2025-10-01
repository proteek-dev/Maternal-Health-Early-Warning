import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import pickle
from datetime import date

# Local modules
from db import init_db, get_conn
from models.outbreak_utils import compute_anomaly_scores

st.set_page_config(page_title="Maternal Health & Outbreak Early Warning",
                   layout="wide")
init_db()

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "maternal_model.pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.warning("Model not trained yet. Click the 'Admin' tab to " \
        "train a synthetic model.")
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def insert_patient_row(payload: dict, risk_score: float, risk_label: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO patients(age, systolic_bp, diastolic_bp, hemoglobin, 
                gestational_weeks, previous_preeclampsia, risk_score, risk_label)
        VALUES(?,?,?,?,?,?,?,?)
    """, (
        payload["age"], payload["systolic_bp"], payload["diastolic_bp"],
        payload.get("hemoglobin"), payload.get("gestational_weeks"),
        1 if payload.get("previous_preeclampsia") else 0,
        risk_score, risk_label
    ))
    conn.commit()
    conn.close()

def upsert_symptoms(df: pd.DataFrame):
    conn = get_conn()
    cur = conn.cursor()
    for _, r in df.iterrows():
        cur.execute("""
            INSERT INTO symptoms(location, date, fever, cough, diarrhea)
            VALUES(?,?,?,?,?)
        """, (r["location"], r["date"], int(r["fever"]), int(r["cough"]),
              int(r["diarrhea"])))
    conn.commit()
    conn.close()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Maternal Risk", "Outbreak Dashboard",
                                  "Data Upload", "Admin"])

# -------- Maternal Risk Page --------
if page == "Maternal Risk":
    st.title("Maternal Risk Screening")
    model = load_model()

    with st.form("maternal_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=14, max_value=55, value=28)
            hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=18.0, value=11.5, step=0.1)
        with col2:
            systolic = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=220, value=120)
            diastolic = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=140, value=80)
        with col3:
            gest_weeks = st.number_input("Gestational Age (weeks)", min_value=4, max_value=42, value=24)
            prev_pre = st.checkbox("Previous preeclampsia")

        submitted = st.form_submit_button("Score Risk")

    if submitted:
        payload = {
            "age": int(age),
            "systolic_bp": int(systolic),
            "diastolic_bp": int(diastolic),
            "hemoglobin": float(hemoglobin),
            "gestational_weeks": int(gest_weeks),
            "previous_preeclampsia": bool(prev_pre)
        }
        if model is None:
            st.error("No model available. Please train one in the Admin tab.")
        else:
            X = pd.DataFrame([payload])
            proba = float(model.predict_proba(X)[0,1])
            label = "HIGH" if proba >= 0.5 else "LOW"
            st.metric("Risk probability", f"{proba:.2f}", help="Probability of risk (synthetic model)")
            st.success(f"Risk label: {label}")
            insert_patient_row(payload, proba, label)

    st.subheader("Recent screenings")
    conn = get_conn()
    dfp = pd.read_sql_query("SELECT created_at, age, systolic_bp, diastolic_bp, hemoglobin, gestational_weeks, previous_preeclampsia, risk_score, risk_label FROM patients ORDER BY id DESC LIMIT 50", conn)
    conn.close()
    st.dataframe(dfp)

# -------- Outbreak Dashboard --------
elif page == "Outbreak Dashboard":
    st.title("Early Outbreak Warning")
    conn = get_conn()
    dfs = pd.read_sql_query("SELECT location, date, fever, cough, diarrhea FROM symptoms", conn)
    conn.close()

    if dfs.empty:
        st.info("No symptom data yet. Upload via 'Data Upload' page.")
    else:
        window = st.slider("Rolling window (days)", 5, 14, 7)
        scores = compute_anomaly_scores(dfs, window=window)
        # Show today's z-scores by location
        today = pd.to_datetime(date.today())
        today_scores = scores[scores["date"] == today][["location", "total_symptoms", "rolling_mean", "rolling_std", "zscore"]]
        st.subheader("Today's anomaly scores")
        st.dataframe(today_scores)

        # Plot per location
        loc = st.selectbox("Select location", sorted(scores["location"].unique().tolist()))
        g = scores[scores["location"] == loc].sort_values("date")
        st.line_chart(g.set_index("date")[["total_symptoms", "rolling_mean"]])
        st.bar_chart(g.set_index("date")[["zscore"]])

# -------- Data Upload --------
elif page == "Data Upload":
    st.title("Upload Data")
    st.write("Upload a CSV for symptoms with columns: location,date,fever,cough,diarrhea")
    st.write("Which will Update the Outbreak Dashboard accordingly.")

    uploaded = st.file_uploader("symptoms.csv", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        required = {"location","date","fever","cough","diarrhea"}
        if not required.issubset(set(df.columns)):
            st.error(f"CSV must contain {required}")
        else:
            upsert_symptoms(df)
            st.success(f"Inserted {len(df)} rows into symptoms.")

    st.divider()
    st.write("Or load packaged synthetic data from /data/symptoms.csv")
    if st.button("Load sample symptoms"):
        sample_path = Path(__file__).resolve().parent / "data" / "symptoms.csv"
        if sample_path.exists():
            df = pd.read_csv(sample_path)
            upsert_symptoms(df)
            st.success(f"Inserted {len(df)} rows from sample.")
        else:
            st.error("Sample file not found.")
            st.error("1. Run data/make_synthetic.py first, before streamlit run app.py")
            st.error("2. Run data/make_synthetic_symptoms.py to generate more realistic symptoms data.")
            st.error("3. OR generate synthetic CSVs from the Admin tab.")

# -------- Admin --------
elif page == "Admin":
    st.title("Admin Utilities")
    st.write("- Train synthetic maternal model\n- Generate synthetic CSVs\n- Reset DB (dev only)")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Train maternal model (synthetic)"):
            from models.train_maternal import make_synthetic, train_and_save
            df = make_synthetic(1000000)
            models_dir = Path(__file__).resolve().parent / "models"
            out = models_dir / "maternal_model.pkl"
            metrics = train_and_save(df, out)
            st.success(f"Model trained. AUC={metrics['auc']:.3f}")
            st.text(metrics["report"])

    with c2:
        if st.button("Generate synthetic CSVs"):
            import subprocess, sys, os
            script = Path(__file__).resolve().parent / "data" / "make_synthetic.py"
            cmd = [sys.executable, str(script)]
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                st.code(out)
            except subprocess.CalledProcessError as e:
                st.error(e.output)

    with c3:
        if st.button("Reset database (dev only)"):
            db_file = Path(__file__).resolve().parent / "app.db"
            if db_file.exists():
                db_file.unlink()
            init_db()
            st.success("Database reset.")
