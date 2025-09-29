"""
Utility to compute simple anomaly scores for outbreak detection:
Use z-score of rolling counts per location.
"""
import pandas as pd
import numpy as np

def compute_anomaly_scores(df: pd.DataFrame, window=7, min_periods=5):
    """
    df columns: date (YYYY-MM-DD), location, fever, cough, diarrhea
    Returns: dataframe with total_symptoms, rolling_mean, rolling_std, zscore
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["total_symptoms"] = df[["fever", "cough", "diarrhea"]].sum(axis=1)
    df = df.sort_values(["location", "date"])

    frames = []
    for loc, g in df.groupby("location"):
        g = g.set_index("date").asfreq("D", fill_value=0)
        g["location"] = loc
        g["rolling_mean"] = g["total_symptoms"].rolling(window=window, min_periods=min_periods).mean()
        g["rolling_std"] = g["total_symptoms"].rolling(window=window, min_periods=min_periods).std(ddof=0).fillna(0.0)
        g["zscore"] = (g["total_symptoms"] - g["rolling_mean"]) / (g["rolling_std"].replace(0, np.nan))
        frames.append(g.reset_index())

    out = pd.concat(frames, ignore_index=True)
    out["zscore"] = out["zscore"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out
