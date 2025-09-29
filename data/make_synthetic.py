"""
Generate small synthetic CSVs:
- maternal_inputs.csv (for manual testing)
- symptoms.csv (daily per location)
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
rng = np.random.default_rng(123)

out_dir = Path(__file__).resolve().parent

# Maternal input samples
N = 50
age = rng.integers(18, 42, size=N)
sys_bp = rng.normal(118, 14, size=N).clip(85, 190).astype(int)
dia_bp = (sys_bp * 0.66 + rng.normal(0, 5, size=N)).clip(50, 120).astype(int)
hb = rng.normal(11.7, 1.0, size=N).clip(7, 15)
weeks = rng.integers(10, 39, size=N)
prev = rng.integers(0, 2, size=N)
df_m = pd.DataFrame({"age": age, "systolic_bp": sys_bp, "diastolic_bp": dia_bp,
                     "hemoglobin": hb, "gestational_weeks": weeks, "previous_preeclampsia": prev})
df_m.to_csv(out_dir / "maternal_inputs.csv", index=False)

# Symptoms time series
start = date.today() - timedelta(days=60)
locs = ["Clinic A", "Clinic B", "Clinic C"]
rows = []
for loc in locs:
    baseline = rng.integers(1, 4)
    for i in range(61):
        d = start + timedelta(days=i)
        # occasional spikes
        spike = 0
        if rng.random() < 0.07:
            spike = rng.integers(4, 10)
        fever = max(0, baseline + int(spike/2) + rng.integers(-1, 2))
        cough = max(0, baseline + int(spike/3) + rng.integers(-1, 2))
        diarrhea = max(0, baseline + int(spike/4) + rng.integers(-1, 2))
        rows.append({"location": loc, "date": d.isoformat(), "fever": fever, "cough": cough, "diarrhea": diarrhea})
df_s = pd.DataFrame(rows)
df_s.to_csv(out_dir / "symptoms.csv", index=False)

print("Wrote:", out_dir / "maternal_inputs.csv")
print("Wrote:", out_dir / "symptoms.csv")
