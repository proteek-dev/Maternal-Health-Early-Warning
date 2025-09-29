# AI for Maternal Health & Early Outbreak Warning

This repository hosts a **prototype system** that demonstrates how artificial intelligence can be applied to improve **maternal health risk screening** and **early outbreak detection** in low-resource settings.
The project is inspired by the UNCTAD 2021 white paper on **AI for Good**, focusing on practical, interpretable solutions that can be run on modest infrastructure.

⚠️ **Disclaimer**: This software is for **research and demonstration purposes only**. It is **not a medical device**, has not been clinically validated, and must not be used for patient care without regulatory approval.

---

## Features

* **Maternal Risk Screening**
  Logistic regression model (trained on synthetic data) predicts elevated maternal risk (e.g., preeclampsia).
  Input fields: age, blood pressure, hemoglobin, gestational age, and prior history.

* **Outbreak Early Warning**
  Tracks reported symptoms (fever, cough, diarrhea) from multiple locations.
  Uses rolling averages and z-score anomalies to flag unusual spikes.

* **Streamlit Web Application**

  * Tab for maternal risk screening with form input.
  * Dashboard for outbreak monitoring with interactive charts.
  * Data upload interface for CSV symptom data.
  * Admin utilities for model training, synthetic data generation, and DB reset.

* **SQLite Database**
  Stores patient screenings, symptom reports, and alert logs.

* **Synthetic Data Tools**
  Scripts to generate test data for development and demonstration.

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/maternal-outbreak-mvp.git
cd maternal-outbreak-mvp
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate sample data (optional)

```bash
python data/make_synthetic.py
```

### 5. Run the app

```bash
streamlit run app.py
```

Open the URL shown in your console (typically [http://localhost:8501](http://localhost:8501)).

---

## Usage

1. **Maternal Risk**

   * Fill out the screening form.
   * Get a probability score and “High/Low” label.
   * Screenings are stored in the database.

2. **Outbreak Dashboard**

   * Load sample symptom data or upload your own.
   * View anomaly scores and plots per location.
   * Adjust rolling window for sensitivity.

3. **Data Upload**

   * Upload CSV with columns: `location,date,fever,cough,diarrhea`.
   * Or load pre-packaged synthetic CSVs from `/data`.

4. **Admin**

   * Train a synthetic maternal model (if missing).
   * Generate synthetic CSVs.
   * Reset the database (development only).

---

## Project Structure

```
maternal-outbreak-mvp/
├── app.py                  # Main Streamlit app
├── db.py                   # SQLite schema and helpers
├── alerts.py               # Stub alert logger
├── app.db                  # SQLite database (created at runtime)
├── data/
│   ├── make_synthetic.py   # Script to generate sample data
│   ├── maternal_inputs.csv # Generated synthetic data
│   └── symptoms.csv        # Generated synthetic data
├── models/
│   ├── train_maternal.py   # Train logistic regression model
│   └── outbreak_utils.py   # Outbreak anomaly detection utils
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .env.example            # Example env vars for alerts (Twilio/SendGrid)
```

---

## Technology Stack

* **Python 3.9+**
* **Streamlit** — Web UI
* **SQLite** — Lightweight persistence
* **scikit-learn** — Machine learning (logistic regression)
* **pandas / numpy** — Data wrangling
* **Matplotlib/Streamlit Charts** — Visualization

---

## Data Sources

This MVP uses **synthetic data** only.
For real deployments, relevant datasets may include:

* WHO Global Health Observatory
* UNICEF maternal health statistics
* DHS (Demographic & Health Surveys)
* Local clinic reporting systems

---
