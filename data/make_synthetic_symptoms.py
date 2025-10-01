"""
Generate small synthetic CSVs:
- maternal_inputs.csv (for manual testing)
- symptoms.csv (daily per location)
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

rng = np.random.default_rng(2025)
out_dir = Path(__file__).resolve().parent

# -----------------------------
# Maternal input samples 
# -----------------------------
N = 50
num_days = 91

age = rng.integers(18, 42, size=N)
sys_bp = rng.normal(118, 14, size=N).clip(85, 190).astype(int)
dia_bp = (sys_bp * 0.66 + rng.normal(0, 5, size=N)).clip(50, 120).astype(int)
hb = rng.normal(11.7, 1.0, size=N).clip(7, 15)
weeks = rng.integers(10, 39, size=N)
prev = rng.integers(0, 2, size=N)

df_m = pd.DataFrame({
    "age": age,
    "systolic_bp": sys_bp,
    "diastolic_bp": dia_bp,
    "hemoglobin": hb,
    "gestational_weeks": weeks,
    "previous_preeclampsia": prev
})
df_m.to_csv(out_dir / "maternal_inputs.csv", index=False)

# -----------------------------
# Symptoms time series
# -----------------------------
# Settings you can tweak
LOCATIONS = ["Clinic A", "Clinic B", "Clinic C"]
FEVER_LAM = 2.0        # avg cases/day baseline
COUGH_LAM = 1.5
DIARRHEA_LAM = 0.8
WEEKEND_DROP = 0.15    # ~15% fewer cases on weekends
OUTBREAK_PROB = 0.07   # 7% chance a day is a spike

# Extra cases added during an outbreak (uniform integer ranges; upper bound exclusive)
FEVER_SPIKE = (5, 15)      # adds 5..14
COUGH_SPIKE = (3, 10)      # adds 3..9
DIARRHEA_SPIKE = (2, 8)    # adds 2..7

start = date.today() - timedelta(days=60)
rows = []

for loc in LOCATIONS:
    # Give each clinic its own variability (±30%)
    clinic_scale = rng.uniform(0.7, 1.3)
    lam_f = FEVER_LAM * clinic_scale
    lam_c = COUGH_LAM * clinic_scale
    lam_d = DIARRHEA_LAM * clinic_scale

    for i in range(num_days):
        d = start + timedelta(days=i)
        is_weekend = d.weekday() >= 5
        w_factor = (1.0 - WEEKEND_DROP) if is_weekend else 1.0

        # Independent Poisson draws
        fever = rng.poisson(lam=lam_f * w_factor)
        cough = rng.poisson(lam=lam_c * w_factor)
        diarrhea = rng.poisson(lam=lam_d * w_factor)

        # Occasional outbreak spike
        if rng.random() < OUTBREAK_PROB:
            fever += rng.integers(*FEVER_SPIKE)
            cough += rng.integers(*COUGH_SPIKE)
            diarrhea += rng.integers(*DIARRHEA_SPIKE)

        rows.append({
            "location": loc,
            "date": d.isoformat(),
            "fever": int(fever),
            "cough": int(cough),
            "diarrhea": int(diarrhea)
        })

df_s = pd.DataFrame(rows)
df_s.to_csv(out_dir / "symptoms.csv", index=False)

print("Wrote:", out_dir / "maternal_inputs.csv")
print("Wrote:", out_dir / "symptoms.csv")
